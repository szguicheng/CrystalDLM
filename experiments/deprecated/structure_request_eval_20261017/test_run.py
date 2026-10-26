import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


RUNNER = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("structure_request_eval", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RequestEvaluationTests(unittest.TestCase):
    def test_first_decoded_structure_creates_its_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts" / "sg-001" / "attempt-000.cif"
            receipt = runner._write_cif(path, "data_test\n")
            self.assertEqual(path.read_text(), "data_test\n")
            self.assertEqual(receipt["bytes"], 10)

    def test_fixed_plan_is_reproducible_and_preflight_limit_covers_all_groups(self):
        plan = runner.attempt_plan()
        self.assertEqual(len(plan), 700)
        self.assertEqual(plan, runner.attempt_plan())
        self.assertEqual(len({item["seed"] for item in plan}), 700)
        first_seven = runner.attempt_plan(1)
        self.assertEqual([item["space_group_number"] for item in first_seven],
                         list(runner.SG_NUMBERS))
        self.assertEqual([item["attempt_id"] for item in first_seven], [0] * 7)

    def test_resume_rejects_changed_checkpoint_or_sampling_identity(self):
        identity = {"checkpoint": "trained-state-a", "temperature": 0.8, "top_p": 0.95}
        runner.validate_manifest(identity, identity.copy())
        with self.assertRaisesRegex(ValueError, "does not match"):
            runner.validate_manifest(identity, {**identity, "checkpoint": "trained-state-b"})

    def test_resume_accepts_json_roundtrip_of_real_request_contract(self):
        contract = {"requests": runner.REQUESTS, "attempt_plan": runner.attempt_plan(1)}
        runner.validate_manifest(json.loads(json.dumps(contract)), contract)

    def test_failed_slots_remain_in_raw_attempt_denominator(self):
        plan = runner.attempt_plan(1)
        receipts = [
            {"request_id": plan[0]["request_id"], "attempt_id": 0,
             "lead_status": "failed", "crystal_status": "skipped",
             "structural_valid": False, "requested_sg_match": False},
            {"request_id": plan[1]["request_id"], "attempt_id": 0,
             "lead_status": "ok", "crystal_status": "ok",
             "structural_valid": True, "requested_sg_match": True},
        ]
        summary = runner.summarize(plan, receipts)
        self.assertEqual(summary["scheduled_attempts"], 7)
        self.assertEqual(summary["raw_attempt_denominator"], 7)
        self.assertEqual(summary["completed_attempts"], 2)
        self.assertEqual(summary["pending_attempts"], 5)
        self.assertEqual(summary["invalid_structure_attempts"], 1)
        self.assertEqual(summary["ehull_evaluation_status"], "not_run")

    def test_condition_uses_decoded_space_group_number(self):
        self.assertTrue(runner.requested_sg_matches(194, 194))
        self.assertFalse(runner.requested_sg_matches(194, 143))
        self.assertFalse(runner.requested_sg_matches(None, 194))

    def test_time_embeddings_match_the_ar_crystal_region_boundary(self):
        self.assertEqual(
            runner.attention_time_values([1, 2, 3, 4, 5, 2, 6, 4, 7], 2, 4),
            [-1, -1, 0, 0, -1, -1, 0, 0, -1])


if __name__ == "__main__":
    unittest.main()
