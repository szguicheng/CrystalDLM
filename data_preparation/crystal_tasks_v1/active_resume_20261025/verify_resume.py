"""Verify the copied authoritative annotation journal before paid resumption."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
sys.path.insert(0, str(ROOT / "research/crystal_bulk_annotation_20261013"))
from batch import BatchRunner, file_sha256, iter_facts

INPUTS = [STAGE / "annotation_inputs" / f"{split}-facts.jsonl.gz"
          for split in ("train", "val", "test")]
DB_PATH = STAGE / "annotation_run/annotation.sqlite3"
ARCHIVED_DB = ROOT / "experiments/deprecated/design_understanding_full_20261025/annotation_run/annotation.sqlite3"
EXPECTED_SPLITS = {"train": 29090, "val": 3601, "test": 4602}
EXPECTED_ROWS = sum(EXPECTED_SPLITS.values())
EXPECTED_SUCCESSES = 1312


def digest_rows(conn, table, order):
    digest = hashlib.sha256()
    rows = conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
    for row in rows:
        digest.update(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {"rows": len(rows), "sha256": digest.hexdigest()}


def normalize_manifest(value):
    return {
        "inputs": [item["sha256"] for item in value["inputs"]],
        "config_sha256": value["config_sha256"],
        "prompt_sha256": value["prompt_sha256"],
        "helper_sha256": value["helper_sha256"],
        "runner_sha256": value["runner_sha256"],
    }


def write_atomic(path, content):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    runner = BatchRunner(
        inputs=INPUTS,
        run_dir=STAGE / "annotation_run",
        config=ROOT / "research/crystal_sharing_20261012/annotation-flash-config.json",
        prompt=ROOT / "research/crystal_sharing_20261012/description-prompt.md",
        env=ROOT / ".env",
        workers=8,
        max_attempts=4,
    )
    current_manifest = runner._manifest()
    config, template = runner._load_config()
    assert config["base_url"] == "https://llm.sciencepal.ai/v1"
    assert config["model"] == "deepseek-v4-flash"
    assert config["thinking"] == {"type": "enabled"}
    assert config["reasoning_effort"] == "max" and config["stream"] is True
    assert config["max_tokens"] == 16384

    env_path = ROOT / ".env"
    key_lines = [line.partition("=")[2].strip().strip("\"'")
                 for line in env_path.read_text().splitlines()
                 if line.startswith("SCIENCEPAL_API_KEY=")]
    assert key_lines and key_lines[0], "SCIENCEPAL_API_KEY is absent from the configured local credential file"

    snapshot_sha256 = file_sha256(DB_PATH)
    copied = sqlite3.connect(DB_PATH)
    copied.row_factory = sqlite3.Row
    assert copied.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    original_manifest = json.loads(copied.execute(
        "SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    assert normalize_manifest(original_manifest) == normalize_manifest(current_manifest)
    assert original_manifest != current_manifest
    for key in current_manifest:
        if key != "inputs":
            assert original_manifest[key] == current_manifest[key], key
    old_input_paths = [item["path"] for item in original_manifest["inputs"]]
    new_input_paths = [str(path.resolve()) for path in INPUTS]
    assert len(old_input_paths) == len(new_input_paths) == 3
    assert [item["sha256"] for item in original_manifest["inputs"]] == [
        file_sha256(path) for path in INPUTS]

    archived = sqlite3.connect(ARCHIVED_DB.as_uri() + "?mode=ro", uri=True)
    archived.row_factory = sqlite3.Row
    archived_manifest = json.loads(archived.execute(
        "SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
    assert normalize_manifest(original_manifest) == normalize_manifest(archived_manifest)
    db_identity = {
        table: digest_rows(copied, table, order)
        for table, order in (("attempts", "id"), ("successes", "material_id"))
    }
    assert db_identity == {
        table: digest_rows(archived, table, order)
        for table, order in (("attempts", "id"), ("successes", "material_id"))
    }

    successes = {row["material_id"]: row for row in copied.execute("SELECT * FROM successes")}
    attempts_by_id = defaultdict(list)
    for row in copied.execute("SELECT * FROM attempts ORDER BY material_id,attempt_number"):
        attempts_by_id[row["material_id"]].append(row)
    assert len(successes) == EXPECTED_SUCCESSES
    assert copied.execute("SELECT COUNT(*) FROM attempts WHERE status='started'").fetchone()[0] == 0
    runner.run_manifest = current_manifest
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    input_counts = Counter()
    input_ids = set()
    request_lines = []
    for row in iter_facts(INPUTS):
        material_id = row["material_id"]
        assert material_id not in input_ids, f"duplicate frozen input ID: {material_id}"
        input_ids.add(material_id)
        input_counts[row["split"]] += 1
        _, facts_hash, request_hash = runner._request_data(row, config, template, endpoint)
        saved = successes.get(material_id)
        if saved:
            assert (saved["facts_sha256"], saved["request_sha256"]) == (facts_hash, request_hash)
        for attempt in attempts_by_id.get(material_id, []):
            assert (attempt["facts_sha256"], attempt["request_sha256"]) == (facts_hash, request_hash)
        request_lines.append(json.dumps({
            "material_id": material_id,
            "split": row["split"],
            "facts_sha256": facts_hash,
            "request_sha256": request_hash,
        }, ensure_ascii=False, separators=(",", ":")) + "\n")
    assert dict(input_counts) == EXPECTED_SPLITS
    assert len(input_ids) == EXPECTED_ROWS
    assert set(successes).issubset(input_ids)
    assert set(attempts_by_id).issubset(input_ids)
    unresolved = {mid: rows[-1]["classification"] for mid, rows in attempts_by_id.items()
                  if mid not in successes}
    assert Counter(unresolved.values()) == Counter({"interrupted": 8})
    assert EXPECTED_ROWS - len(successes) == 35981

    rebound_manifest = dict(original_manifest)
    rebound_manifest["inputs"] = [
        {"path": path, "sha256": item["sha256"]}
        for path, item in zip(new_input_paths, original_manifest["inputs"])
    ]
    assert {k: v for k, v in rebound_manifest.items() if k != "inputs"} == {
        k: v for k, v in original_manifest.items() if k != "inputs"}
    copied.execute("UPDATE metadata SET value=? WHERE key='manifest'",
                   (json.dumps(rebound_manifest, sort_keys=True),))
    copied.commit()
    copied.close()
    archived.close()
    assert runner.status() == {
        "successes": EXPECTED_SUCCESSES,
        "failed_materials": 8,
        "incomplete_attempts": 0,
        "database": str(DB_PATH.resolve()),
    }

    request_bytes = "".join(request_lines).encode()
    request_path = STAGE / "request-manifest.jsonl"
    write_atomic(request_path, request_bytes)
    success_ids = sorted(successes)
    ids_bytes = ("".join(item + "\n" for item in success_ids)).encode()
    ids_path = STAGE / "success_ids_at_resume.txt"
    write_atomic(ids_path, ids_bytes)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authority": "/data1/users/liugang/crystalDLM/data_preparation/crystal_tasks_v1/annotation_run/annotation.sqlite3",
        "sqlite_backup_snapshot": "/data1/users/liugang/crystalDLM/data_preparation/crystal_tasks_v1/resume_artifacts/annotation_snapshot_20261026T0500Z.sqlite3",
        "source_snapshot_sha256": snapshot_sha256,
        "local_archive_journal": str(ARCHIVED_DB),
        "local_archive_and_remote_snapshot_rows_equal": db_identity,
        "metadata_rebind": [{"old_path": old, "new_path": new}
                            for old, new in zip(old_input_paths, new_input_paths)],
        "input_files": [{"path": str(path.resolve()), "sha256": file_sha256(path),
                         "rows": input_counts[split]}
                        for split, path in zip(("train", "val", "test"), INPUTS)],
        "input_rows": EXPECTED_ROWS,
        "reused_previous_captions": 8172,
        "saved_new_captions_at_resume": len(successes),
        "remaining_at_resume": EXPECTED_ROWS - len(successes),
        "success_ids_file": ids_path.name,
        "success_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "attempts": copied_attempt_count(),
        "latest_unresolved_failures": dict(Counter(unresolved.values())),
        "settings": {key: config[key] for key in
                     ("base_url", "model", "thinking", "reasoning_effort", "max_tokens", "stream")},
        "workers": 8,
        "max_attempts": 4,
        "config_sha256": current_manifest["config_sha256"],
        "prompt_sha256": current_manifest["prompt_sha256"],
        "helper_sha256": current_manifest["helper_sha256"],
        "runner_sha256": current_manifest["runner_sha256"],
        "launcher_sha256": file_sha256(STAGE / "annotate.py"),
        "verifier_sha256": file_sha256(__file__),
        "request_manifest_file": request_path.name,
        "request_manifest_rows": len(request_lines),
        "request_manifest_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "credential_source": "local .env variable SCIENCEPAL_API_KEY; present, value not recorded",
        "dataset_finalization_run": False,
    }
    write_atomic(STAGE / "resume-record.json", (json.dumps(record, indent=2) + "\n").encode())
    print(json.dumps({"verified": True, "successes": len(successes),
                      "remaining": EXPECTED_ROWS - len(successes),
                      "request_manifest_sha256": record["request_manifest_sha256"]}))


def copied_attempt_count():
    conn = sqlite3.connect(DB_PATH.as_uri() + "?mode=ro", uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    finally:
        conn.close()


if __name__ == "__main__":
    main()
