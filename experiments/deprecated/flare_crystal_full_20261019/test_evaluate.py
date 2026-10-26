"""CPU-only contracts for block generation and quality receipts."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("flare_full_evaluate", HERE / "evaluate.py")
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


class BlockSamplerContracts(unittest.TestCase):
    def test_each_flare_commit_reduces_mask_count_by_one(self):
        current = [99, 99, 99, 99]
        next_state, position, token = EVALUATE._commit_highest_confidence(
            current, [0, 1, 2, 3], [10, 11, 12, 13], [.1, .8, .2, .3])
        self.assertEqual(next_state, [99, 11, 99, 99])
        self.assertEqual((position, token), (1, 11))

    def test_equal_confidence_commits_leftmost_mask_deterministically(self):
        state, position, token = EVALUATE._commit_highest_confidence(
            [99, 99], [0, 1], [7, 8], [.5, .5])
        self.assertEqual((state, position, token), ([7, 99], 0, 7))

    def test_end_truncates_current_block_and_reports_missing_end(self):
        self.assertEqual(EVALUATE._truncate_at_end([4, 5, 2, 8, 9], 2), ([4, 5, 2], True))
        self.assertEqual(EVALUATE._truncate_at_end([4, 5], 2), ([4, 5], False))


class ReceiptContracts(unittest.TestCase):
    def test_resume_checks_arm_checkpoint_code_and_request_fingerprint(self):
        item = {"request_id": "sg-001", "attempt_id": 0, "seed": 7,
                "prompt": "request", "space_group_number": 1}
        manifest = {"arm": "flare", "checkpoint_fingerprint_sha256": "checkpoint",
                    "trainable_state_sha256": "state", "evaluation_source_sha256": {"x": "y"}}
        receipt = {"request_id": item["request_id"], "attempt_id": item["attempt_id"],
                   "seed": item["seed"], "arm": "flare",
                   "checkpoint_fingerprint_sha256": "checkpoint",
                   "trainable_state_sha256": "state", "evaluation_source_sha256": {"x": "y"},
                   "request_fingerprint_sha256": EVALUATE._request_fingerprint(item)}
        EVALUATE._validate_receipt(receipt, item, manifest)
        for key, value in (("arm", "ar"), ("checkpoint_fingerprint_sha256", "other"),
                           ("evaluation_source_sha256", {"x": "other"}),
                           ("request_fingerprint_sha256", "other")):
            changed = {**receipt, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                EVALUATE._validate_receipt(changed, item, manifest)

    def test_failures_remain_in_scheduled_denominator_and_invalid_readback_is_separate(self):
        plan = [{"request_id": "sg-001", "attempt_id": 0, "seed": 1}]
        receipt = {"request_id": "sg-001", "attempt_id": 0, "seed": 1,
                   "lead_status": "failed", "crystal_status": "skipped",
                   "structural_valid": False, "requested_sg_match": False,
                   "readback": {"status": "skipped", "reason": "lead failed"}}
        summary = EVALUATE.RUNNER.summarize(plan, [receipt])
        EVALUATE._augment_summary(summary, [receipt])
        self.assertEqual(summary["raw_attempt_denominator"], 1)
        self.assertEqual(summary["strict_valid_structures"], 0)
        self.assertEqual(summary["readback_denominator"], 0)
        self.assertEqual(summary["readback_skipped_invalid_structures"], 1)

    def test_readback_scores_decoded_space_group_and_only_the_four_requested_fields(self):
        class Score:
            @staticmethod
            def parse(body):
                return {"formula": "Mg2O2", "sg": 1, "atoms": 4, "orbits": 2}

        graded = EVALUATE._grade_readback(
            Score, "The formula is Mg2 O2. The space group number is 194. The unit cell contains 4 atoms and 2 independent orbits.",
            "body", ["Mg", "Mg", "O", "O"], 194)
        self.assertEqual(tuple(graded["fields"]), EVALUATE.READBACK_FIELDS)
        self.assertTrue(graded["fields"]["space_group"])
        self.assertTrue(graded["all_four_correct"])


if __name__ == "__main__":
    unittest.main()
