"""Run the frozen NatureCry first-pass pilot on the first 30 selected cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import api_runner

PILOT_PREFIX = 30
PILOT_REQUEST_LIMIT = 2 * PILOT_PREFIX
PRIOR_ATTEMPT_BREAKDOWN = {
    "legacy_firstpass_001": 7,
    "compatible_http400_pilot": 30,
    "passthrough_diagnostic": 1,
}
PRIOR_ATTEMPTS = sum(PRIOR_ATTEMPT_BREAKDOWN.values())
GLOBAL_REQUEST_LIMIT = 200


def normalize_endpoint(base_url):
    parts = urlsplit(base_url.strip())
    if (parts.scheme not in ("http", "https") or not parts.netloc or parts.username or parts.password or
            parts.query or parts.fragment):
        raise RuntimeError("SCIENCEPAL_BASE_URL must be an HTTP(S) base URL")
    path = parts.path.rstrip("/")
    if path.endswith("/chat/completions"):
        pass
    elif path.endswith("/v1"):
        path += "/chat/completions"
    else:
        path += "/v1/chat/completions"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


class CompatibleFirstPassRunner(api_runner.FirstPassRunner):
    def request(self, messages, _temperature, case_id, stage):
        if self.request_count >= self.max_requests or self.request_count >= api_runner.MAX_REQUESTS:
            raise api_runner.RequestFailure("First-pass request budget exhausted", "budget_exhausted", "budget_exhausted")
        payload = {"model": api_runner.MODEL, "messages": messages,
                   "reasoning_effort": "max", "max_completion_tokens": 8192,
                   "allowed_openai_params": ["reasoning_effort"]}
        request_path = self.output / "requests" / f"{case_id}.{stage}.request.json"
        api_runner.write_json(request_path, payload, self.key)
        request = Request(api_runner.ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                          method="POST")
        self.request_count += 1
        try:
            with self.opener.open(request, timeout=180) as response:
                raw = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            api_runner.write_json(self.output / "requests" / f"{case_id}.{stage}.http_error.json",
                                  {"status": error.code, "body": body}, self.key)
            error.close()
            raise api_runner.RequestFailure(f"HTTP {error.code}; no retry",
                                            f"{stage}_transport", "http_error") from None
        except (URLError, TimeoutError, OSError) as error:
            raise api_runner.RequestFailure(f"Transport {type(error).__name__}; no retry",
                                            f"{stage}_transport", "transport_error") from None
        try:
            result = json.loads(raw)
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
            response_path = self.output / "requests" / f"{case_id}.{stage}.response.raw"
            response_path.write_text(raw.decode("utf-8", "replace").replace(self.key, "[REDACTED]"))
            raise api_runner.RequestFailure("Malformed Chat Completions response; no retry",
                                            f"{stage}_format", "response_format") from None
        usage = result.get("usage")
        if isinstance(usage, dict):
            self.usage.append(usage)
        api_runner.write_json(self.output / "requests" / f"{case_id}.{stage}.response.json", result, self.key)
        if not isinstance(content, str) or finish_reason != "stop":
            raise api_runner.RequestFailure("Model response is incomplete or not plain text; no retry",
                                            f"{stage}_incomplete", "incomplete_response")
        if self.key and self.key in content:
            raise api_runner.RequestFailure("Model response contains a credential; redacted",
                                            f"{stage}_secret_redacted", "credential_redacted")
        return content, usage, finish_reason


def execute(prepared, output, key, network_opener=None, max_requests=PILOT_REQUEST_LIMIT):
    prepared, output = Path(prepared), Path(output)
    if max_requests < 1 or max_requests > PILOT_REQUEST_LIMIT:
        raise ValueError(f"Pilot request budget must be between 1 and {PILOT_REQUEST_LIMIT}")
    if PRIOR_ATTEMPTS + max_requests > GLOBAL_REQUEST_LIMIT:
        raise ValueError("Pilot plus prior attempts exceed the global request limit")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot output: {output}")

    hashes, full_manifest, selection, checks = api_runner.verify_prepared(prepared, include_preflight=True)
    expected_ids = full_manifest["case_ids_in_original_order"][:PILOT_PREFIX]
    pilot_manifest = dict(full_manifest)
    pilot_manifest["case_ids_in_original_order"] = expected_ids
    real_verify = api_runner.verify_prepared
    real_runner = api_runner.FirstPassRunner

    def verify_first30(path, include_preflight=False):
        if Path(path).resolve() != prepared.resolve():
            raise RuntimeError("API execution switched away from the verified prepared directory")
        return hashes, pilot_manifest, selection, checks

    try:
        api_runner.verify_prepared = verify_first30
        api_runner.FirstPassRunner = CompatibleFirstPassRunner
        summary = api_runner.execute(prepared, output, key, max_requests, network_opener)
    finally:
        api_runner.verify_prepared = real_verify
        api_runner.FirstPassRunner = real_runner

    records = json.loads((output / "records.json").read_text())
    if [record["case"] for record in records] != expected_ids:
        raise RuntimeError("Pilot records do not match the first 30 frozen selection cases")
    if any(record["encoder_requests"] > 1 or record["decoder_requests"] > 1 for record in records):
        raise RuntimeError("A case/stage was attempted more than once")
    attempts = sum(record["encoder_requests"] + record["decoder_requests"] for record in records)
    if attempts != summary["total_requests"] or attempts > max_requests:
        raise RuntimeError("Pilot request ledger does not reconcile to case records")

    passed = sum(record["status"] == "pass" for record in records)
    summary.update(selected_denominator=PILOT_PREFIX, completed=PILOT_PREFIX,
                   strict_firstpass_matches=passed, firstpass_match_rate=passed / PILOT_PREFIX,
                   failed=PILOT_PREFIX - passed, total_requests=attempts,
                   requests_reconciled_to_case_records=True)
    summary["pilot_controller"] = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_cohort_size": 100,
        "evaluated_prefix": PILOT_PREFIX,
        "pilot_request_limit": max_requests,
        "pilot_attempts": attempts,
        "prior_attempts": PRIOR_ATTEMPTS,
        "prior_attempt_breakdown": PRIOR_ATTEMPT_BREAKDOWN,
        "global_attempts_with_prior": PRIOR_ATTEMPTS + attempts,
        "global_attempt_limit": GLOBAL_REQUEST_LIMIT,
        "prior_attempts_included_in_pilot_result": False,
        "legacy_reference": {
            "directory": str((HERE / "api_runs" / "firstpass_001").resolve()),
            "attempts": PRIOR_ATTEMPT_BREAKDOWN["legacy_firstpass_001"],
            "model_alias": api_runner.MODEL,
            "encoder_temperature": 0.4,
            "decoder_temperature": 0.0,
            "reasoning_effort": "not explicitly set",
            "max_tokens": 8192,
            "attempts_reused": 0,
        },
        "compatible_http400_reference": {
            "directory": str((HERE / "api_runs" / "pilot30_k3_compatible_001").resolve()),
            "attempts": PRIOR_ATTEMPT_BREAKDOWN["compatible_http400_pilot"],
            "outcome": "HTTP 400 before model execution; excluded from pilot score",
        },
        "passthrough_diagnostic_reference": {
            "directory": str((HERE / "diagnostics" / "k3_http400_20261004").resolve()),
            "attempts": PRIOR_ATTEMPT_BREAKDOWN["passthrough_diagnostic"],
            "outcome": "separate gateway passthrough diagnostic; excluded from pilot score",
        },
        "request_parameters": {
            "model_alias": api_runner.MODEL,
            "reasoning_effort": "max",
            "temperature": "omitted",
            "max_completion_tokens": 8192,
            "allowed_openai_params": ["reasoning_effort"],
            "response_format": "omitted",
        },
    }
    api_runner.write_json(output / "summary.json", summary, key)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared_final")
    parser.add_argument("--out", type=Path, default=HERE / "api_runs" / "pilot30_k3_passthrough_001")
    parser.add_argument("--execute-api", action="store_true", help="Explicitly enable the fresh 30-case API pilot")
    args = parser.parse_args()
    if not args.execute_api:
        parser.error("network disabled; pass --execute-api after protocol review")

    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
    key = os.environ.get("SCIENCEPAL_API_KEY")
    base_url = os.environ.get("SCIENCEPAL_BASE_URL")
    if not key or not base_url:
        parser.error("SCIENCEPAL_BASE_URL and SCIENCEPAL_API_KEY must be set; no credential prompt or fallback is used")
    try:
        endpoint = normalize_endpoint(base_url)
    except RuntimeError as error:
        parser.error(str(error))
    if endpoint != api_runner.ENDPOINT:
        parser.error("Configured SciencePal endpoint differs from the frozen first-pass endpoint")

    summary = execute(args.prepared.resolve(), args.out.resolve(), key)
    print(json.dumps(api_runner.redact(summary, key), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
