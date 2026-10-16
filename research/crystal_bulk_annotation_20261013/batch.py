"""Durable, resumable English annotation runner for the frozen subset."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[1]
sys.path.insert(0, str(ROOT / "research/crystal_preannotation_20261012"))
import annotation as helpers

DEFAULT_INPUTS = [STAGE / "subset_001/train-facts.jsonl.gz", STAGE / "subset_001/val-facts.jsonl.gz"]
DEFAULT_CONFIG = ROOT / "research/crystal_sharing_20261012/annotation-flash-config.json"
DEFAULT_PROMPT = ROOT / "research/crystal_sharing_20261012/description-prompt.md"
DEFAULT_RUN_DIR = STAGE / "run_001"
MAX_WORKERS = 8
MAX_ATTEMPTS = 4


class BatchError(RuntimeError):
    pass


class ManifestDrift(BatchError):
    pass


class GlobalStop(BatchError):
    pass


class LockBusy(BatchError):
    pass


class IncompleteResponse(BatchError):
    pass


class PermanentResponse(BatchError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_facts(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row["status"] != "ok":
                        raise BatchError(f"Non-success fact row in {path}")
                    yield row


class FileLock:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+")
        try:
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            self.stream = None
            raise LockBusy(f"Another batch process holds {self.path}") from None
        self.stream.seek(0)
        self.stream.truncate()
        self.stream.write(f"pid={os.getpid()}\n")
        self.stream.flush()
        return self

    def __exit__(self, *args):
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None


class DoneTrackingStream:
    def __init__(self, response):
        self.response = response
        self.saw_done = False

    def __iter__(self):
        for raw in self.response:
            line = raw.decode("utf-8").strip()
            if line.startswith("data:") and line[5:].strip() == "[DONE]":
                self.saw_done = True
            yield raw


class BatchRunner:
    def __init__(self, inputs=None, run_dir=DEFAULT_RUN_DIR, config=DEFAULT_CONFIG,
                 prompt=DEFAULT_PROMPT, env=ROOT / ".env", workers=MAX_WORKERS,
                 max_attempts=MAX_ATTEMPTS):
        self.inputs = [Path(p).resolve() for p in (inputs or DEFAULT_INPUTS)]
        self.run_dir = Path(run_dir).resolve()
        self.db_path = self.run_dir / "annotation.sqlite3"
        self.config_path = Path(config).resolve()
        self.prompt_path = Path(prompt).resolve()
        self.env_path = Path(env).resolve()
        self.workers = workers
        self.max_attempts = max_attempts
        self.run_manifest = None
        if not 1 <= workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")

    def lock(self):
        return FileLock(self.run_dir / "runner.lock")

    def _connect(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                material_id TEXT NOT NULL, split TEXT NOT NULL,
                facts_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                attempt_number INTEGER NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT, status TEXT NOT NULL, classification TEXT,
                http_status INTEGER, error_type TEXT, error_detail TEXT,
                elapsed_seconds REAL, receipt_json TEXT,
                UNIQUE(material_id, attempt_number));
            CREATE TABLE IF NOT EXISTS successes (
                material_id TEXT PRIMARY KEY, split TEXT NOT NULL,
                facts_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                attempt_number INTEGER NOT NULL, receipt_json TEXT NOT NULL);
        """)
        conn.commit()
        return conn

    def _connect_readonly(self):
        uri = self.db_path.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _manifest(self):
        return {
            "inputs": [{"path": str(path), "sha256": file_sha256(path)} for path in self.inputs],
            "config_sha256": file_sha256(self.config_path),
            "prompt_sha256": file_sha256(self.prompt_path),
            "helper_sha256": file_sha256(ROOT / "research/crystal_preannotation_20261012/annotation.py"),
            "runner_sha256": file_sha256(__file__),
        }

    def _ensure_manifest(self, conn):
        self.run_manifest = self._manifest()
        current = json.dumps(self.run_manifest, sort_keys=True)
        previous = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if previous is None:
            conn.execute("INSERT INTO metadata(key,value) VALUES('manifest',?)", (current,))
            conn.commit()
        elif previous["value"] != current:
            raise ManifestDrift("Input, config, prompt, helper, or runner changed; use a new run directory")

    def _load_config(self):
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        if (config["thinking"] != {"type": "enabled"} or config["reasoning_effort"] != "max"
                or config["stream"] is not True or "flash" not in config["model"].lower()):
            raise GlobalStop("Config must retain Flash, thinking enabled, max reasoning, and streaming")
        prompt_file = self.prompt_path.read_text(encoding="utf-8")
        template = prompt_file.split("```text\n", 1)[1].split("```", 1)[0]
        return config, template

    def _request_data(self, row, config, template, endpoint):
        payload = helpers.payload_for(row, config, template)
        manifest = self.run_manifest or self._manifest()
        facts_hash = helpers.sha(row)
        request_hash = helpers.sha({"facts": row, "endpoint": endpoint, "payload": payload,
            "config_sha256": manifest["config_sha256"], "prompt_sha256": manifest["prompt_sha256"],
            "helper_sha256": manifest["helper_sha256"], "runner_sha256": manifest["runner_sha256"]})
        return payload, facts_hash, request_hash

    @staticmethod
    def _counts(conn, material_id):
        rows = conn.execute("SELECT attempt_number,status,classification FROM attempts WHERE material_id=? ORDER BY attempt_number",
                            (material_id,)).fetchall()
        return len(rows), (rows[-1] if rows else None)

    @staticmethod
    def _trailing_transient_count(conn, material_id):
        rows = conn.execute("SELECT classification FROM attempts WHERE material_id=? ORDER BY attempt_number DESC",
                            (material_id,)).fetchall()
        count = 0
        for row in rows:
            if row["classification"] == "interrupted":
                break
            if row["classification"] != "transient":
                break
            count += 1
        return count

    @staticmethod
    def _recover_started(conn):
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE attempts SET status='error',classification='interrupted',finished_at=?,"
                     "error_type='InterruptedAttempt',error_detail='Process ended before the request receipt was committed.' "
                     "WHERE status='started'", (utc_now(),))
        conn.commit()

    def _start_attempt(self, conn, row, facts_hash, request_hash):
        count, _ = self._counts(conn, row["material_id"])
        number = count + 1
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO attempts(material_id,split,facts_sha256,request_sha256,attempt_number,started_at,status) "
                     "VALUES(?,?,?,?,?,?, 'started')",
                     (row["material_id"], row["split"], facts_hash, request_hash, number, utc_now()))
        conn.commit()
        return number

    def _request(self, row, payload, config, endpoint, key, started):
        request = urllib.request.Request(endpoint, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                tracking = DoneTrackingStream(response)
                parsed = helpers.read_stream(tracking, started)
                http_status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            if status in (401, 403):
                raise GlobalStop(f"HTTP {status}; check credentials or access") from None
            raise
        if parsed["returned_model"] and parsed["returned_model"] != config["model"]:
            raise GlobalStop(f"Returned model {parsed['returned_model']!r} did not match configured model")
        if not tracking.saw_done or not parsed["finish_reason"] or not parsed["description"] or not parsed["returned_model"]:
            raise IncompleteResponse("SSE ended without a complete nonempty answer")
        if parsed["finish_reason"] != "stop":
            raise PermanentResponse(f"Unexpected finish reason {parsed['finish_reason']!r}")
        parsed["http_status"] = http_status
        return parsed

    @staticmethod
    def _classification(exc):
        if isinstance(exc, GlobalStop):
            return "global"
        if isinstance(exc, urllib.error.HTTPError):
            if exc.code in (401, 403):
                return "global"
            return "transient" if exc.code in (408, 425, 429) or 500 <= exc.code <= 599 else "permanent"
        if isinstance(exc, (IncompleteResponse, http.client.IncompleteRead, json.JSONDecodeError,
                            TimeoutError, socket.timeout, urllib.error.URLError,
                            ConnectionError, ConnectionResetError)):
            return "transient"
        return "permanent"

    @staticmethod
    def _error_text(exc, key):
        return str(exc).replace(key, "[redacted]")

    def _save_failure(self, conn, material_id, attempt_number, exc, classification, started, key):
        http_status = exc.code if isinstance(exc, urllib.error.HTTPError) else None
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE attempts SET finished_at=?,status='error',classification=?,http_status=?,error_type=?,"
                     "error_detail=?,elapsed_seconds=? WHERE material_id=? AND attempt_number=?",
                     (utc_now(), classification, http_status, type(exc).__name__, self._error_text(exc, key),
                      round(time.monotonic() - started, 3), material_id, attempt_number))
        conn.commit()

    def _save_success(self, conn, row, facts_hash, request_hash, attempt_number, parsed, started):
        raw = parsed["description"]
        receipt = {
            "material_id": row["material_id"], "split": row["split"],
            "facts_sha256": facts_hash, "request_sha256": request_hash,
            "attempt_number": attempt_number, "status": "ok", "http_status": parsed["http_status"],
            "returned_model": parsed["returned_model"], "finish_reason": parsed["finish_reason"],
            "structure_description": raw, "description": helpers.render_description(raw, row),
            "property_labels": row["property_labels"], "usage": parsed["usage"],
            "reasoning_content_characters": parsed["reasoning_content_characters"],
            "time_to_first_content_seconds": parsed["time_to_first_content_seconds"],
            "content_stream_seconds": parsed["content_stream_seconds"],
            "content_chunks": parsed["content_chunks"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        encoded = json.dumps(receipt, ensure_ascii=False, separators=(",", ":"))
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE attempts SET finished_at=?,status='ok',classification='success',http_status=?,"
                     "elapsed_seconds=?,receipt_json=? WHERE material_id=? AND attempt_number=?",
                     (utc_now(), parsed["http_status"], receipt["elapsed_seconds"], encoded,
                      row["material_id"], attempt_number))
        conn.execute("INSERT INTO successes(material_id,split,facts_sha256,request_sha256,attempt_number,receipt_json) "
                     "VALUES(?,?,?,?,?,?)", (row["material_id"], row["split"], facts_hash, request_hash,
                                              attempt_number, encoded))
        conn.commit()
        return receipt

    def _submit(self, pool, pending, conn, row, payload, config, endpoint, key,
                facts_hash, request_hash, remaining, retry_index=0):
        attempt_number = self._start_attempt(conn, row, facts_hash, request_hash)
        started = time.monotonic()
        delay = min(30, 2 ** retry_index) if retry_index else 0
        future = pool.submit(self._request_after_delay, row, payload, config, endpoint, key, started, delay)
        pending[future] = (row, facts_hash, request_hash, attempt_number, remaining, retry_index, started)

    def _request_after_delay(self, row, payload, config, endpoint, key, started, delay):
        if delay:
            time.sleep(delay)
        return self._request(row, payload, config, endpoint, key, started)

    def run(self, limit=None, retry_failed=False):
        with self.lock():
            conn = self._connect()
            try:
                self._ensure_manifest(conn)
                self._recover_started(conn)
                config, template = self._load_config()
                try:
                    key = helpers.load_key(self.env_path)
                except (KeyError, FileNotFoundError):
                    raise GlobalStop("SCIENCEPAL_API_KEY is unavailable") from None
                endpoint = config["base_url"].rstrip("/") + "/chat/completions"
                facts = iter_facts(self.inputs)
                pending, active = {}, set()
                scanned = 0
                exhausted = False
                global_error = None
                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    while True:
                        while not exhausted and global_error is None and len(pending) < self.workers:
                            if limit is not None and scanned >= limit:
                                exhausted = True
                                break
                            try:
                                row = next(facts)
                            except StopIteration:
                                exhausted = True
                                break
                            scanned += 1
                            material_id = row["material_id"]
                            if material_id in active:
                                continue
                            payload, facts_hash, request_hash = self._request_data(row, config, template, endpoint)
                            success = conn.execute("SELECT facts_sha256,request_sha256 FROM successes WHERE material_id=?",
                                                   (material_id,)).fetchone()
                            if success:
                                if (success["facts_sha256"], success["request_sha256"]) != (facts_hash, request_hash):
                                    raise ManifestDrift(f"Stored receipt does not match current facts for {material_id}")
                                continue
                            count, latest = self._counts(conn, material_id)
                            if retry_failed:
                                if latest is None:
                                    continue
                                remaining = self.max_attempts
                            elif count == 0:
                                remaining = self.max_attempts
                            elif latest["classification"] == "interrupted":
                                remaining = self.max_attempts
                            elif latest["classification"] == "transient":
                                remaining = self.max_attempts - self._trailing_transient_count(conn, material_id)
                                if remaining <= 0:
                                    continue
                            else:
                                continue
                            active.add(material_id)
                            self._submit(pool, pending, conn, row, payload, config, endpoint, key,
                                         facts_hash, request_hash, remaining)

                        if not pending:
                            break
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        outcomes = []
                        for future in done:
                            state = pending.pop(future)
                            try:
                                outcomes.append((state, future.result(), None))
                            except Exception as exc:
                                outcomes.append((state, None, exc))
                        retries = []
                        for state, parsed, exc in outcomes:
                            row, facts_hash, request_hash, attempt_number, remaining, retry_index, started = state
                            if exc is None:
                                receipt = self._save_success(conn, row, facts_hash, request_hash,
                                                             attempt_number, parsed, started)
                                print(json.dumps({"material_id": row["material_id"], "status": "ok",
                                    "attempt": attempt_number, "elapsed_seconds": receipt["elapsed_seconds"],
                                    "usage": receipt["usage"]}), flush=True)
                                active.discard(row["material_id"])
                            else:
                                classification = self._classification(exc)
                                self._save_failure(conn, row["material_id"], attempt_number, exc,
                                                   classification, started, key)
                                if classification == "global":
                                    global_error = GlobalStop(self._error_text(exc, key))
                                elif classification == "transient" and remaining > 1:
                                    retries.append((state, retry_index + 1))
                                else:
                                    active.discard(row["material_id"])
                                    print(json.dumps({"material_id": row["material_id"], "status": "error",
                                        "attempt": attempt_number, "error_type": type(exc).__name__,
                                        "classification": classification}), flush=True)
                        if global_error is None:
                            for state, next_retry in retries:
                                row, facts_hash, request_hash, _, remaining, _, _ = state
                                self._submit(pool, pending, conn, row,
                                    helpers.payload_for(row, config, template), config, endpoint, key,
                                    facts_hash, request_hash, remaining - 1, next_retry)
                        else:
                            for state, _ in retries:
                                active.discard(state[0]["material_id"])
                if global_error:
                    raise global_error
                return self.status_locked(conn)
            finally:
                conn.close()

    def status_locked(self, conn):
        row = conn.execute("SELECT COUNT(*) AS successes FROM successes").fetchone()
        failures = conn.execute("SELECT COUNT(DISTINCT a.material_id) AS n FROM attempts a "
                                 "LEFT JOIN successes s USING(material_id) WHERE s.material_id IS NULL "
                                 "AND a.status='error'").fetchone()["n"]
        started = conn.execute("SELECT COUNT(DISTINCT a.material_id) AS n FROM attempts a "
                                "LEFT JOIN successes s USING(material_id) WHERE s.material_id IS NULL "
                                "AND a.status='started'").fetchone()["n"]
        return {"successes": row["successes"], "failed_materials": failures, "incomplete_attempts": started,
                "database": str(self.db_path)}

    def status(self):
        if not self.db_path.exists():
            return {"successes": 0, "failed_materials": 0, "incomplete_attempts": 0,
                    "database": str(self.db_path)}
        conn = self._connect_readonly()
        try:
            previous = conn.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if previous is None or previous["value"] != json.dumps(self._manifest(), sort_keys=True):
                raise ManifestDrift("Input, config, prompt, helper, or runner changed; use a new run directory")
            conn.execute("BEGIN")
            result = self.status_locked(conn)
            conn.commit()
            return result
        finally:
            conn.close()

    @staticmethod
    def _atomic_jsonl(path, rows):
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def export(self):
        with self.lock():
            conn = self._connect()
            try:
                self._ensure_manifest(conn)
                self._recover_started(conn)
                success_path = self.run_dir / "successes.jsonl"
                failure_path = self.run_dir / "failures.jsonl"

                def success_rows():
                    for row in iter_facts(self.inputs):
                        saved = conn.execute("SELECT receipt_json FROM successes WHERE material_id=?",
                                             (row["material_id"],)).fetchone()
                        if saved:
                            yield json.loads(saved["receipt_json"])

                def failure_rows():
                    for row in iter_facts(self.inputs):
                        material_id = row["material_id"]
                        if conn.execute("SELECT 1 FROM successes WHERE material_id=?", (material_id,)).fetchone():
                            continue
                        attempts = conn.execute("SELECT attempt_number,status,classification,http_status,error_type,"
                            "error_detail,elapsed_seconds FROM attempts WHERE material_id=? ORDER BY attempt_number",
                            (material_id,)).fetchall()
                        if attempts:
                            latest = dict(attempts[-1])
                            yield {"material_id": material_id, "split": row["split"],
                                "facts_sha256": helpers.sha(row), "attempt_count": len(attempts),
                                "classification": latest["classification"], "http_status": latest["http_status"],
                                "error_type": latest["error_type"], "error_detail": latest["error_detail"],
                                "attempts": [dict(a) for a in attempts]}

                self._atomic_jsonl(success_path, success_rows())
                self._atomic_jsonl(failure_path, failure_rows())
                return success_path, failure_path
            finally:
                conn.close()


def _parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "retry-failed", "status", "export"):
        command = commands.add_parser(name)
        command.add_argument("--input", action="append", dest="inputs", help="gzip JSONL input; repeat for each split")
        command.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
        command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        command.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
        command.add_argument("--env", type=Path, default=ROOT / ".env")
        command.add_argument("--workers", type=int, default=MAX_WORKERS)
        command.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
        if name in ("run", "retry-failed"):
            command.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    runner = BatchRunner(args.inputs, args.run_dir, args.config, args.prompt, args.env,
                         args.workers, args.max_attempts)
    try:
        if args.command == "run":
            result = runner.run(limit=args.limit)
        elif args.command == "retry-failed":
            result = runner.run(limit=args.limit, retry_failed=True)
        elif args.command == "status":
            result = runner.status()
        else:
            success_path, failure_path = runner.export()
            result = {"successes": str(success_path), "failures": str(failure_path)}
    except GlobalStop as exc:
        print(f"global stop: {exc}", file=sys.stderr)
        return 2
    except BatchError as exc:
        print(f"batch stopped: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
