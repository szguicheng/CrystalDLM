from io import BytesIO
from pathlib import Path
import json
import tempfile
import pytest
import json

from urllib.error import URLError

from api_runner import FirstPassRunner, RequestFailure, execute


class FailingOpener:
    def __init__(self):
        self.calls = 0

    def open(self, request, timeout):
        self.calls += 1
        raise URLError("offline fake failure")


class TruncatedOpener:
    def __init__(self, body):
        self.body = body

    def open(self, request, timeout):
        return BytesIO(self.body)


def test_transport_failure_is_counted_and_preserves_full_denominator():
    root = Path(__file__).resolve().parent
    opener = FailingOpener()
    secret = "test-only-never-save-this-key"
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        output = Path(temporary) / "api-failure"
        summary = execute(root / "prepared_final", output, secret, 1, opener=opener)

        records = json.loads((output / "records.json").read_text())
        assert len(records) == 100
        assert records[0]["encoder_requests"] == 1
        assert records[0]["decoder_requests"] == 0
        assert all(record["encoder_requests"] == 0 for record in records[1:])
        assert summary["total_requests"] == 1
        assert summary["requests_reconciled_to_case_records"] is True
        assert summary["failed"] == 100
        assert summary["automatic_retries"] == 0
        assert opener.calls == 1
        for path in output.rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                assert secret.encode() not in data
                assert b"Authorization" not in data


def test_non_stop_completion_is_a_failure_and_response_is_redacted():
    root = Path(__file__).resolve().parent
    secret = "test-only-never-save-this-key"
    body = json.dumps({"choices": [{"message": {"content": f"echo {secret}"}, "finish_reason": "length"}]}).encode()
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        output = Path(temporary)
        runner = FirstPassRunner(secret, output, 1, opener=TruncatedOpener(body))
        with pytest.raises(RequestFailure) as error:
            runner.request([{"role": "user", "content": "inline"}], 0, "case", "decoder")
        assert error.value.stage == "decoder_incomplete"
        assert runner.request_count == 1
        response = (output / "requests/case.decoder.response.json").read_text()
        assert secret not in response
        assert "[REDACTED]" in response
        assert "Authorization" not in response
