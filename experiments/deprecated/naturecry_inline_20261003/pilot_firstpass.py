"""Run the frozen NatureCry first-pass pilot on the first 30 selected cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from openai import APIConnectionError, APIStatusError

import api_runner

PILOT_PREFIX = 30
PILOT_REQUEST_LIMIT = 2 * PILOT_PREFIX
REASONING_EFFORTS = ("low", "high", "max")
DEFAULT_REASONING_EFFORT = "low"
PRIOR_ATTEMPT_BREAKDOWN = {
    "legacy_firstpass_001": 7,
    "compatible_http400_pilot": 30,
    "passthrough_diagnostic": 1,
    "passthrough_disconnect_pilot": 14,
    "max_effort_stream_aborted": 1,
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


def make_sdk_client(api_key, base_url, openai_factory=None, httpx_client_factory=None):
    if openai_factory is None:
        from openai import OpenAI
        openai_factory = OpenAI
    if httpx_client_factory is None:
        import httpx
        httpx_client_factory = httpx.Client
    transport = httpx_client_factory(timeout=180, follow_redirects=False)
    return openai_factory(api_key=api_key, base_url=base_url, timeout=180,
                          max_retries=0, http_client=transport)


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value


def _chunk_data(chunk):
    data = _jsonable(chunk)
    for choice in getattr(chunk, "choices", []) or []:
        reasoning = getattr(choice.delta, "reasoning_content", None)
        if not reasoning:
            continue
        index = getattr(choice, "index", 0)
        for saved_choice in data.get("choices", []):
            if saved_choice.get("index", 0) == index:
                saved_choice.setdefault("delta", {})["reasoning_content"] = reasoning
                break
    return data


class StreamingFirstPassRunner(api_runner.FirstPassRunner):
    client = None
    reasoning_effort = DEFAULT_REASONING_EFFORT

    def __init__(self, key, output, max_requests, opener=None):
        super().__init__(key, output, max_requests, opener)
        self.transport_abort_reason = None
        self.stream_stats = []
        self.client = type(self).client
        self.reasoning_effort = type(self).reasoning_effort
        snapshot = self.output / "pilot_firstpass.py"
        if not snapshot.exists():
            snapshot.write_text(Path(__file__).read_text())

    def _abort(self, case_id, stage, failure_kind, detail, error=None, status_code=None):
        if self.transport_abort_reason is None:
            self.transport_abort_reason = api_runner.redact({
                "case_id": case_id,
                "stage": stage,
                "failure_kind": failure_kind,
                "error_type": type(error).__name__ if error is not None else "StreamIncomplete",
                "detail": detail,
                "status_code": status_code,
            }, self.key)
        return self.transport_abort_reason

    def _write_transport_error(self, case_id, stage, failure_kind, detail, error=None):
        api_runner.write_json(self.output / "requests" / f"{case_id}.{stage}.transport_error.json", {
            "failure_kind": failure_kind,
            "error_type": type(error).__name__ if error is not None else "StreamIncomplete",
            "detail": detail,
        }, self.key)

    def request(self, messages, _temperature, case_id, stage):
        if self.transport_abort_reason is not None:
            raise api_runner.RequestFailure("Not attempted after an earlier transport failure",
                                            "not_attempted_after_transport_abort", "transport_aborted")
        if self.request_count >= self.max_requests or self.request_count >= api_runner.MAX_REQUESTS:
            raise api_runner.RequestFailure("First-pass request budget exhausted", "budget_exhausted", "budget_exhausted")

        allowed = ["reasoning_effort"]
        payload = {"model": api_runner.MODEL, "messages": messages, "reasoning_effort": self.reasoning_effort,
                   "max_completion_tokens": 8192, "allowed_openai_params": allowed,
                   "stream": True, "stream_options": {"include_usage": True}}
        request_path = self.output / "requests" / f"{case_id}.{stage}.request.json"
        api_runner.write_json(request_path, payload, self.key)
        self.request_count += 1
        stream_path = self.output / "requests" / f"{case_id}.{stage}.stream.jsonl"
        content_parts, usage = [], None
        response_id = observed_model = created = finish_reason = None
        chunk_count = reasoning_chunks = reasoning_characters = 0
        iterator_ended_normally = False
        stats = {"case_id": case_id, "stage": stage, "chunks": 0,
                 "reasoning_chunks": 0, "reasoning_characters": 0,
                 "finish_reason": None, "iterator_ended_normally": False}
        try:
            with stream_path.open("w", encoding="utf-8") as stream_log:
                stream = self.client.chat.completions.create(
                    model=api_runner.MODEL,
                    messages=messages,
                    reasoning_effort=self.reasoning_effort,
                    max_completion_tokens=8192,
                    stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"allowed_openai_params": allowed},
                )
                with stream:
                    for chunk in stream:
                        saved_chunk = api_runner.redact(_chunk_data(chunk), self.key)
                        stream_log.write(json.dumps(saved_chunk, ensure_ascii=False, allow_nan=False) + "\n")
                        stream_log.flush()
                        chunk_count += 1
                        response_id = getattr(chunk, "id", None) or response_id
                        observed_model = getattr(chunk, "model", None) or observed_model
                        created = getattr(chunk, "created", None) or created
                        if getattr(chunk, "usage", None) is not None:
                            usage = _jsonable(chunk.usage)
                        for choice in getattr(chunk, "choices", []) or []:
                            if getattr(choice, "index", 0) != 0:
                                continue
                            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
                            delta = choice.delta
                            text = getattr(delta, "content", None)
                            if text is not None:
                                if not isinstance(text, str):
                                    raise api_runner.RequestFailure("Stream content was not plain text",
                                                                    f"{stage}_format", "response_format")
                                content_parts.append(text)
                            reasoning = getattr(delta, "reasoning_content", None)
                            if reasoning:
                                reasoning_chunks += 1
                                reasoning_characters += len(reasoning)
                iterator_ended_normally = True
        except APIStatusError as error:
            body = error.body
            try:
                json.dumps(body, allow_nan=False)
            except (TypeError, ValueError):
                body = str(body)
            api_runner.write_json(self.output / "requests" / f"{case_id}.{stage}.http_error.json",
                                  {"status": error.status_code, "body": body}, self.key)
            detail = api_runner.redact(str(error), self.key)
            self._abort(case_id, stage, "http_error", detail, error, error.status_code)
            raise api_runner.RequestFailure(f"HTTP {error.status_code}; no retry",
                                            f"{stage}_transport", "http_error") from None
        except APIConnectionError as error:
            detail = api_runner.redact(str(error), self.key)
            self._write_transport_error(case_id, stage, "connection_error", detail, error)
            self._abort(case_id, stage, "connection_error", detail, error)
            raise api_runner.RequestFailure(f"{type(error).__name__}; no retry",
                                            f"{stage}_transport", "connection_error") from None
        except api_runner.RequestFailure:
            raise
        except Exception as error:
            detail = api_runner.redact(str(error), self.key)
            self._write_transport_error(case_id, stage, "stream_transport_error", detail, error)
            self._abort(case_id, stage, "stream_transport_error", detail, error)
            raise api_runner.RequestFailure(f"{type(error).__name__}; no retry",
                                            f"{stage}_transport", "stream_transport_error") from None
        finally:
            if usage is not None:
                self.usage.append(usage)
            stats.update(chunks=chunk_count, reasoning_chunks=reasoning_chunks,
                         reasoning_characters=reasoning_characters,
                         finish_reason=finish_reason,
                         iterator_ended_normally=iterator_ended_normally)
            self.stream_stats.append(stats)

        content = "".join(content_parts)
        if finish_reason is None:
            detail = "SDK stream ended without a finish_reason"
            self._write_transport_error(case_id, stage, "stream_transport_incomplete", detail)
            self._abort(case_id, stage, "stream_transport_incomplete", detail)
            raise api_runner.RequestFailure(detail, f"{stage}_stream_incomplete", "stream_incomplete")

        response = {"id": response_id, "object": "chat.completion", "created": created,
                    "model": observed_model, "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish_reason,
                    }],
                    "reasoning_stream": {"chunks": reasoning_chunks, "characters": reasoning_characters}}
        if usage is not None:
            response["usage"] = usage
        api_runner.write_json(self.output / "requests" / f"{case_id}.{stage}.response.json", response, self.key)
        if finish_reason != "stop":
            raise api_runner.RequestFailure("Model response is incomplete or not plain text; no retry",
                                            f"{stage}_incomplete", "incomplete_response")
        if self.key and self.key in content:
            raise api_runner.RequestFailure("Model response contains a credential; redacted",
                                            f"{stage}_secret_redacted", "credential_redacted")
        return content, usage, finish_reason


def execute(prepared, output, key, client, max_requests=PILOT_REQUEST_LIMIT,
            reasoning_effort=DEFAULT_REASONING_EFFORT):
    prepared, output = Path(prepared), Path(output)
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}")
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
    runner_holder = {}
    sdk_client = client
    selected_effort = reasoning_effort

    def verify_first30(path, include_preflight=False):
        if Path(path).resolve() != prepared.resolve():
            raise RuntimeError("API execution switched away from the verified prepared directory")
        return hashes, pilot_manifest, selection, checks

    class PilotRunner(StreamingFirstPassRunner):
        client = sdk_client
        reasoning_effort = selected_effort

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runner_holder["runner"] = self

    try:
        api_runner.verify_prepared = verify_first30
        api_runner.FirstPassRunner = PilotRunner
        summary = api_runner.execute(prepared, output, key, max_requests)
    finally:
        api_runner.verify_prepared = real_verify
        api_runner.FirstPassRunner = real_runner

    runner = runner_holder["runner"]
    records = json.loads((output / "records.json").read_text())
    if [record["case"] for record in records] != expected_ids:
        raise RuntimeError("Pilot records do not match the first 30 frozen selection cases")
    if any(record["encoder_requests"] > 1 or record["decoder_requests"] > 1 for record in records):
        raise RuntimeError("A case/stage was attempted more than once")
    attempts = sum(record["encoder_requests"] + record["decoder_requests"] for record in records)
    if attempts != summary["total_requests"] or attempts > max_requests:
        raise RuntimeError("Pilot request ledger does not reconcile to case records")

    passed = sum(record["status"] == "pass" for record in records)
    actual_cases_attempted = sum(record["encoder_requests"] + record["decoder_requests"] > 0 for record in records)
    not_attempted_cases = PILOT_PREFIX - actual_cases_attempted
    summary.update(selected_denominator=PILOT_PREFIX, completed=PILOT_PREFIX,
                   strict_firstpass_matches=passed, firstpass_match_rate=passed / PILOT_PREFIX,
                   failed=PILOT_PREFIX - passed, total_requests=attempts,
                   requests_reconciled_to_case_records=True,
                   actual_cases_attempted=actual_cases_attempted,
                   not_attempted_cases=not_attempted_cases,
                   transport_abort_reason=runner.transport_abort_reason)
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
        "passthrough_disconnect_reference": {
            "directory": str((HERE / "api_runs" / "pilot30_k3_passthrough_001").resolve()),
            "attempts": PRIOR_ATTEMPT_BREAKDOWN["passthrough_disconnect_pilot"],
            "outcome": "13 transport failures plus one interrupted request; excluded from pilot score",
        },
        "max_effort_stream_reference": {
            "directory": str((HERE / "api_runs" / "pilot30_k3_stream_001").resolve()),
            "attempts": PRIOR_ATTEMPT_BREAKDOWN["max_effort_stream_aborted"],
            "outcome": "reasoning-only stream interrupted without finish_reason; excluded from pilot score",
            "reasoning_chunks": 2212,
            "answer_chunks": 0,
        },
        "request_parameters": {
            "model_alias": api_runner.MODEL,
            "reasoning_effort": runner.reasoning_effort,
            "temperature": "omitted",
            "max_completion_tokens": 8192,
            "allowed_openai_params": ["reasoning_effort"],
            "allowed_openai_params_via": "OpenAI SDK extra_body",
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": "omitted",
            "sdk_max_retries": 0,
            "timeout_seconds": 180,
            "follow_redirects": False,
        },
        "stream_summary": {
            "streams_started": len(runner.stream_stats),
            "streams_ended_normally": sum(item["iterator_ended_normally"] for item in runner.stream_stats),
            "chunks_received": sum(item["chunks"] for item in runner.stream_stats),
            "reasoning_chunks": sum(item["reasoning_chunks"] for item in runner.stream_stats),
            "reasoning_characters": sum(item["reasoning_characters"] for item in runner.stream_stats),
        },
    }
    api_runner.write_json(output / "summary.json", summary, key)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared_final")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default=DEFAULT_REASONING_EFFORT)
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
    sdk_base_url = endpoint.removesuffix("/chat/completions")
    client = make_sdk_client(key, sdk_base_url)
    try:
        output = args.out or HERE / "api_runs" / f"pilot30_k3_stream_{args.reasoning_effort}_001"
        summary = execute(args.prepared.resolve(), output.resolve(), key, client,
                          reasoning_effort=args.reasoning_effort)
    finally:
        client.close()
    print(json.dumps(api_runner.redact(summary, key), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
