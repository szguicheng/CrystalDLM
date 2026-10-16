import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    import httpx
    from openai import APIStatusError
    import pilot_firstpass as pilot
except ModuleNotFoundError:
    pilot = None
    httpx = None
    APIStatusError = None


PREPARED = HERE / "prepared_final"


class FakeChunk:
    def __init__(self, data, choices, usage=None):
        self.data = data
        self.id = data["id"]
        self.model = data["model"]
        self.created = data["created"]
        self.choices = choices
        self.usage = usage

    def model_dump(self, mode="json", exclude_none=True):
        return self.data


def make_chunk(content=None, reasoning=None, finish=None, usage=None):
    choices, raw_choices = [], []
    if content is not None or reasoning is not None or finish is not None:
        delta = {"content": content, "reasoning_content": reasoning}
        choices.append(type("Choice", (), {
            "index": 0,
            "delta": type("Delta", (), delta)(),
            "finish_reason": finish,
        })())
        raw_choices.append({"index": 0, "delta": {key: value for key, value in delta.items() if value is not None},
                            "finish_reason": finish})
    data = {"id": "chunk-id", "object": "chat.completion.chunk", "created": 1700000000,
            "model": "tokenhub/kimi-k3", "choices": raw_choices}
    if usage is not None:
        data["usage"] = usage
    return FakeChunk(data, choices, usage)


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeClient:
    def __init__(self, stream_factory):
        self.stream_factory = stream_factory
        self.calls = []
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {"create": self.create})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.stream_factory(kwargs)
        if isinstance(result, Exception):
            raise result
        return result


class FreshPilotTests(unittest.TestCase):
    def test_controller_module_exists(self):
        self.assertIsNotNone(pilot, "pilot_firstpass.py has not been implemented")

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_offline_pilot_uses_streaming_payload_and_first_30_denominator(self):
        usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}
        client = FakeClient(lambda _: FakeStream([
            make_chunk(content="invalid encoder document", finish="stop"),
            make_chunk(usage=usage),
        ]))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pilot"
            summary = pilot.execute(PREPARED, output, "DUMMY_TEST_KEY", client=client)
            records = json.loads((output / "records.json").read_text())
            selected = json.loads((PREPARED / "selection.json").read_text())["cases"]
            expected_ids = [case["id"] for case in selected[:30]]

            self.assertEqual([record["case"] for record in records], expected_ids)
            self.assertFalse((output / "requests" / f"{selected[30]['id']}.encoder.request.json").exists())
            self.assertEqual(len(records), 30)
            self.assertEqual(len(client.calls), 30)
            self.assertTrue(all(call["stream"] is True for call in client.calls))
            self.assertTrue(all(call["stream_options"] == {"include_usage": True} for call in client.calls))
            self.assertTrue(all(call["reasoning_effort"] == "low" for call in client.calls))
            self.assertTrue(all(call["max_completion_tokens"] == 8192 and "temperature" not in call
                                for call in client.calls))
            self.assertTrue(all(call["extra_body"] == {"allowed_openai_params": ["reasoning_effort"]}
                                for call in client.calls))
            payloads = [json.loads(path.read_text()) for path in (output / "requests").glob("*.request.json")]
            self.assertEqual(len(payloads), 30)
            self.assertTrue(all(payload["stream"] is True and
                                payload["stream_options"] == {"include_usage": True} and
                                payload["allowed_openai_params"] == ["reasoning_effort"]
                                for payload in payloads))
            self.assertTrue(all(record["encoder_requests"] <= 1 and record["decoder_requests"] <= 1
                                for record in records))
            self.assertEqual(summary["selected_denominator"], 30)
            self.assertEqual(summary["completed"], 30)
            self.assertEqual(summary["strict_firstpass_matches"], 0)
            self.assertEqual(summary["failed"], 30)
            self.assertEqual(summary["firstpass_match_rate"], 0.0)
            self.assertEqual(summary["actual_cases_attempted"], 30)
            self.assertEqual(summary["not_attempted_cases"], 0)
            self.assertIsNone(summary["transport_abort_reason"])
            self.assertEqual(summary["pilot_controller"]["frozen_cohort_size"], 100)
            self.assertEqual(summary["pilot_controller"]["evaluated_prefix"], 30)
            self.assertEqual(summary["pilot_controller"]["pilot_request_limit"], 60)
            self.assertEqual(summary["pilot_controller"]["pilot_attempts"], 30)
            self.assertEqual(summary["pilot_controller"]["prior_attempts"], 53)
            self.assertEqual(summary["pilot_controller"]["prior_attempt_breakdown"], {
                "legacy_firstpass_001": 7,
                "compatible_http400_pilot": 30,
                "passthrough_diagnostic": 1,
                "passthrough_disconnect_pilot": 14,
                "max_effort_stream_aborted": 1,
            })
            self.assertEqual(summary["pilot_controller"]["global_attempts_with_prior"], 83)
            self.assertLessEqual(53 + summary["pilot_controller"]["pilot_request_limit"], 200)
            self.assertTrue(summary["requests_reconciled_to_case_records"])
            self.assertEqual(summary["pilot_controller"]["request_parameters"]["allowed_openai_params"],
                             ["reasoning_effort"])
            self.assertEqual(summary["pilot_controller"]["request_parameters"]["reasoning_effort"], "low")
            for path in [output / "records.json", output / "summary.json", *sorted((output / "requests").glob("*"))]:
                self.assertNotIn("DUMMY_TEST_KEY", path.read_text())
            self.assertEqual((output / "pilot_firstpass.py").read_text(),
                             (HERE / "pilot_firstpass.py").read_text())

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_stream_separates_reasoning_and_captures_empty_choice_usage(self):
        secret = "DUMMY_TEST_KEY"
        usage = {"prompt_tokens": 4, "completion_tokens": 9, "total_tokens": 13}
        client = FakeClient(lambda _: FakeStream([
            make_chunk(reasoning=f"private thought {secret}"),
            make_chunk(content="crystal "),
            make_chunk(content="shape", finish="stop"),
            make_chunk(usage=usage),
        ]))
        with tempfile.TemporaryDirectory() as tmp:
            runner = pilot.StreamingFirstPassRunner(secret, Path(tmp), 2)
            runner.client = client
            content, reported_usage, finish = runner.request([], 0.4, "mp-test", "encoder")
            response = json.loads((Path(tmp) / "requests" / "mp-test.encoder.response.json").read_text())
            chunks = [json.loads(line) for line in
                      (Path(tmp) / "requests" / "mp-test.encoder.stream.jsonl").read_text().splitlines()]

            self.assertEqual(content, "crystal shape")
            self.assertEqual(reported_usage, usage)
            self.assertEqual(finish, "stop")
            self.assertEqual(response["choices"][0]["message"]["content"], "crystal shape")
            self.assertNotIn("reasoning_content", response["choices"][0]["message"])
            self.assertEqual(response["reasoning_stream"],
                             {"chunks": 1, "characters": len(f"private thought {secret}")})
            self.assertEqual(chunks[0]["choices"][0]["delta"]["reasoning_content"],
                             "private thought [REDACTED]")
            self.assertNotIn(secret, (Path(tmp) / "requests" / "mp-test.encoder.stream.jsonl").read_text())
            self.assertEqual(chunks[-1]["choices"], [])
            self.assertEqual(chunks[-1]["usage"], usage)
            self.assertEqual(runner.request_count, 1)
            self.assertEqual(len(client.calls), 1)

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_missing_finish_aborts_before_any_following_request(self):
        client = FakeClient(lambda _: FakeStream([make_chunk(content="partial")]))
        with tempfile.TemporaryDirectory() as tmp:
            runner = pilot.StreamingFirstPassRunner("DUMMY_TEST_KEY", Path(tmp), 4)
            runner.client = client
            with self.assertRaises(pilot.api_runner.RequestFailure) as caught:
                runner.request([], 0.4, "mp-first", "encoder")
            with self.assertRaises(pilot.api_runner.RequestFailure) as blocked:
                runner.request([], 0.4, "mp-next", "encoder")

            self.assertEqual(caught.exception.failure_kind, "stream_incomplete")
            self.assertEqual(blocked.exception.failure_kind, "transport_aborted")
            self.assertEqual(runner.request_count, 1)
            self.assertEqual(len(client.calls), 1)
            self.assertFalse((Path(tmp) / "requests" / "mp-next.encoder.request.json").exists())
            self.assertEqual(runner.transport_abort_reason["stage"], "encoder")

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_length_is_a_model_truncation_and_does_not_abort_queue(self):
        streams = [
            FakeStream([make_chunk(content="truncated", finish="length")]),
            FakeStream([make_chunk(content="second", finish="stop")]),
        ]
        client = FakeClient(lambda _: streams.pop(0))
        with tempfile.TemporaryDirectory() as tmp:
            runner = pilot.StreamingFirstPassRunner("DUMMY_TEST_KEY", Path(tmp), 4)
            runner.client = client
            with self.assertRaises(pilot.api_runner.RequestFailure) as caught:
                runner.request([], 0.4, "mp-first", "encoder")
            self.assertEqual(caught.exception.failure_kind, "incomplete_response")
            self.assertIsNone(runner.transport_abort_reason)
            self.assertEqual(runner.request([], 0.0, "mp-second", "decoder"), ("second", None, "stop"))
            self.assertEqual(runner.request_count, 2)
            self.assertEqual(len(client.calls), 2)

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_execute_marks_remaining_cases_unattempted_after_first_transport_failure(self):
        client = FakeClient(lambda _: OSError("RemoteDisconnected"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "pilot"
            summary = pilot.execute(PREPARED, output, "DUMMY_TEST_KEY", client=client)
            records = json.loads((output / "records.json").read_text())
            selected = json.loads((PREPARED / "selection.json").read_text())["cases"]

            self.assertEqual(len(client.calls), 1)
            self.assertEqual(summary["total_requests"], 1)
            self.assertEqual(summary["actual_cases_attempted"], 1)
            self.assertEqual(summary["not_attempted_cases"], 29)
            self.assertEqual(summary["transport_abort_reason"]["stage"], "encoder")
            self.assertEqual(summary["pilot_controller"]["global_attempts_with_prior"], 54)
            self.assertEqual(records[1]["case"], selected[1]["id"])
            self.assertEqual(records[1]["stage"], "not_attempted_after_transport_abort")
            self.assertEqual(records[1]["encoder_requests"], 0)
            self.assertFalse((output / "requests" / f"{selected[1]['id']}.encoder.request.json").exists())

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_http_error_body_is_redacted_and_attempt_is_not_retried(self):
        secret = "DUMMY_TEST_KEY"
        body = {"message": f"unsupported; key={secret}"}
        request = httpx.Request("POST", pilot.api_runner.ENDPOINT)
        error = APIStatusError("Bad Request", response=httpx.Response(400, request=request), body=body)
        client = FakeClient(lambda _: error)
        with tempfile.TemporaryDirectory() as tmp:
            runner = pilot.StreamingFirstPassRunner(secret, Path(tmp), 2)
            runner.client = client
            with self.assertRaises(pilot.api_runner.RequestFailure) as caught:
                runner.request([], 0.4, "mp-test", "encoder")
            error_file = json.loads((Path(tmp) / "requests" / "mp-test.encoder.http_error.json").read_text())

            self.assertEqual(error_file["status"], 400)
            self.assertIn("[REDACTED]", json.dumps(error_file["body"]))
            self.assertNotIn(secret, json.dumps(error_file))
            self.assertEqual(caught.exception.failure_kind, "http_error")
            self.assertEqual(runner.request_count, 1)
            self.assertEqual(len(client.calls), 1)

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_sdk_client_disables_retries_and_redirects(self):
        captured = {}

        def httpx_factory(**kwargs):
            captured["httpx"] = kwargs
            return object()

        def openai_factory(**kwargs):
            captured["openai"] = kwargs
            return object()

        pilot.make_sdk_client("DUMMY_TEST_KEY", "https://llm.sciencepal.ai/v1",
                              openai_factory=openai_factory, httpx_client_factory=httpx_factory)
        self.assertEqual(captured["openai"]["max_retries"], 0)
        self.assertEqual(captured["openai"]["timeout"], 180)
        self.assertEqual(captured["openai"]["base_url"], "https://llm.sciencepal.ai/v1")
        self.assertEqual(captured["httpx"], {"timeout": 180, "follow_redirects": False})

    @unittest.skipIf(pilot is None, "pilot controller not implemented")
    def test_base_url_normalizes_only_to_the_frozen_endpoint(self):
        self.assertEqual(pilot.normalize_endpoint("https://llm.sciencepal.ai"), pilot.api_runner.ENDPOINT)
        self.assertEqual(pilot.normalize_endpoint("https://llm.sciencepal.ai/v1/"), pilot.api_runner.ENDPOINT)
        self.assertNotEqual(pilot.normalize_endpoint("https://example.invalid"), pilot.api_runner.ENDPOINT)


if __name__ == "__main__":
    unittest.main()
