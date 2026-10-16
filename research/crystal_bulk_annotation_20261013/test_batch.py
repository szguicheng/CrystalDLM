import gzip
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from research.crystal_bulk_annotation_20261013 import batch


def fact(material_id, split, value):
    return {
        "status": "ok", "material_id": material_id, "split": split,
        "core_facts": {"cell_formula": "Si", "reduced_formula": "Si", "elements": ["Si"],
            "element_count": 1, "cell_atom_count": 1, "space_group_number": 1,
            "space_group_symbol": "P1", "crystal_system": "triclinic", "independent_orbit_count": 1},
        "local_structure_facts": {"coordination_environments": [], "polyhedra_sharing": {"status": "not_computed"},
            "connectivity": {"dimension": 0}},
        "property_labels": {"energy_above_hull": value, "unit": "eV/atom", "provenance": "archived MP structure"},
    }


class Response:
    status = 200

    def __init__(self, model, text="A concise crystal description.", done=True):
        self.lines = [
            "data: " + json.dumps({"model": model, "choices": [{"delta": {"content": text}, "finish_reason": None}]}),
            "data: " + json.dumps({"model": model, "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                                   "choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ] + (["data: [DONE]"] if done else [])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return (line.encode() + b"\n" for line in self.lines)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = []
        for split, value in (("train", "0.123456789"), ("val", "0.456789123")):
            path = self.root / f"{split}-facts.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(fact("mp-" + split, split, value)) + "\n")
            self.inputs.append(path)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"base_url": "https://example.invalid/v1", "model": "deepseek-v4-flash",
            "thinking": {"type": "enabled"}, "reasoning_effort": "max", "stream": True, "max_tokens": 16384}))
        self.prompt = self.root / "prompt.md"
        self.prompt.write_text("```text\nDescribe {structure_version}: {core_facts}; {local_structure_facts}; {property_labels}\n```\n")
        self.env = self.root / ".env"
        self.env.write_text("SCIENCEPAL_API_KEY=must-not-leak\n")
        self.run_dir = self.root / "run"

    def tearDown(self):
        self.temp.cleanup()

    def runner(self, max_attempts=2, workers=2):
        return batch.BatchRunner(self.inputs, self.run_dir, self.config, self.prompt, self.env,
                                 workers=workers, max_attempts=max_attempts)

    def test_success_resume_exports_exact_label_and_no_credential(self):
        calls = []
        def fake(request, timeout):
            calls.append(request)
            return Response("deepseek-v4-flash")
        with patch("urllib.request.urlopen", side_effect=fake):
            self.runner().run(limit=1)
            self.runner().run()
        self.assertEqual(len(calls), 2)
        paths = self.runner().export()
        receipts = [json.loads(line) for line in paths[0].read_text().splitlines()]
        self.assertEqual(len(receipts), 2)
        self.assertIn("0.123456789 eV/atom", receipts[0]["description"])
        self.assertEqual(receipts[0]["structure_description"], "A concise crystal description.")
        self.assertEqual(receipts[0]["usage"]["prompt_tokens"], 11)
        self.assertNotIn("must-not-leak", paths[0].read_text() + paths[1].read_text())

    def test_transient_exhaustion_is_queued_and_next_material_continues(self):
        calls = {"mp-train": 0, "mp-val": 0}
        def fake(request, timeout):
            body = request.data.decode("utf-8")
            material_id = "mp-train" if "material mp-train" in body else "mp-val"
            calls[material_id] += 1
            if material_id == "mp-train" and calls[material_id] <= 2:
                raise urllib.error.HTTPError(request.full_url, 429, "busy", None, None)
            return Response("deepseek-v4-flash")
        with patch("urllib.request.urlopen", side_effect=fake):
            self.runner().run()
        self.assertEqual(calls, {"mp-train": 2, "mp-val": 1})
        successes, failures = self.runner().export()
        self.assertEqual(len(successes.read_text().splitlines()), 1)
        failed = json.loads(failures.read_text().splitlines()[0])
        self.assertEqual(failed["material_id"], "mp-train")
        self.assertEqual(failed["attempt_count"], 2)
        self.assertEqual(failed["error_type"], "HTTPError")

    def test_auth_error_stops_before_next_record(self):
        calls = []
        def fake(request, timeout):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", None, None)
        with patch("urllib.request.urlopen", side_effect=fake):
            with self.assertRaises(batch.GlobalStop):
                self.runner(workers=1).run()
        self.assertEqual(len(calls), 1)

    def test_interrupted_request_is_resumed_from_durable_attempt(self):
        runner = self.runner(workers=1)
        with patch("urllib.request.urlopen", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                runner.run(limit=1)
        with patch("urllib.request.urlopen", return_value=Response("deepseek-v4-flash")):
            runner.run(limit=1)
        receipt = json.loads(runner.export()[0].read_text().splitlines()[0])
        self.assertEqual(receipt["attempt_number"], 2)

    def test_fake_provider_secret_is_not_logged(self):
        calls = []
        def fake(request, timeout):
            body = request.data.decode("utf-8")
            calls.append(body)
            if "material mp-train" in body:
                raise RuntimeError("provider echoed must-not-leak")
            return Response("deepseek-v4-flash")
        with patch("urllib.request.urlopen", side_effect=fake):
            self.runner(max_attempts=1, workers=1).run()
        success_path, failure_path = self.runner().export()
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(success_path.read_text().splitlines()), 1)
        self.assertNotIn("must-not-leak", failure_path.read_text())

    def test_incomplete_sse_retried_and_model_mismatch_stops(self):
        calls = []
        def fake(request, timeout):
            calls.append(request.full_url)
            return Response("deepseek-v4-flash", done=len(calls) > 1)
        with patch("urllib.request.urlopen", side_effect=fake):
            self.runner().run(limit=1)
        self.assertEqual(len(calls), 2)
        mismatch = lambda request, timeout: Response("other-model")
        with patch("urllib.request.urlopen", side_effect=mismatch):
            with self.assertRaises(batch.GlobalStop):
                self.runner().run()

    def test_manifest_drift_and_exclusive_lock(self):
        runner = self.runner()
        with runner.lock():
            with self.assertRaises(batch.LockBusy):
                with self.runner().lock():
                    pass
        with patch("urllib.request.urlopen", return_value=Response("deepseek-v4-flash")):
            runner.run(limit=1)
        self.prompt.write_text(self.prompt.read_text() + "changed\n")
        with self.assertRaises(batch.ManifestDrift):
            self.runner().run()


if __name__ == "__main__":
    unittest.main()
