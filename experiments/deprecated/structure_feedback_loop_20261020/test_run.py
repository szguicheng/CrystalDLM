import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("structure_feedback_loop_run", HERE / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class PilotContracts(unittest.TestCase):
    def test_retry_seeds_are_deterministic_and_paired_and_arm_stops_at_success(self):
        seed = 17531
        independent_seeds, feedback_seeds = [], []

        def retry(seeds, success_round):
            def generate(round_index, retry_seed, _previous):
                seeds.append((round_index, retry_seed))
                return {"success": round_index == success_round}
            return generate

        independent = RUN._continue_arm([{"success": False}], seed,
                                        retry(independent_seeds, 1), lambda x: x["success"])
        feedback = RUN._continue_arm([{"success": False}], seed,
                                     retry(feedback_seeds, 2), lambda x: x["success"])
        self.assertEqual(independent_seeds[0], feedback_seeds[0])
        self.assertEqual(independent_seeds[0][1], RUN.retry_seed(seed, 1))
        self.assertEqual([x[0] for x in independent_seeds], [1])
        self.assertEqual([x[0] for x in feedback_seeds], [1, 2])
        self.assertTrue(independent[-1]["success"])
        self.assertTrue(feedback[-1]["success"])

    def test_failed_arm_uses_at_most_three_candidates(self):
        calls = []
        attempts = RUN._continue_arm(
            [{"success": False}], 91,
            lambda round_index, seed, previous: calls.append((round_index, seed)) or {"success": False},
            lambda receipt: receipt["success"])
        self.assertEqual(len(attempts), 3)
        self.assertEqual([round_index for round_index, _ in calls], [1, 2])

    def test_feedback_context_preserves_original_and_closes_only_context_copy(self):
        original = [1, 2]
        prior = {"lead_token_ids": [10, 90], "crystal_raw_token_ids": [11, 12]}
        context, synthetic_end = RUN._feedback_context(original, prior, [20, 21], 90, 91)
        self.assertEqual(context, [1, 2, 10, 90, 11, 12, 91, 20, 21])
        self.assertTrue(synthetic_end)
        self.assertEqual(prior["crystal_raw_token_ids"], [11, 12])
        no_start, synthetic_end = RUN._feedback_context(
            original, {"lead_token_ids": [10], "crystal_raw_token_ids": [11]}, [20], 90, 91)
        self.assertEqual(no_start, [1, 2, 10, 11, 20])
        self.assertFalse(synthetic_end)

    def test_success_requires_strict_audit_and_requested_detected_sg(self):
        audit = {"valid": True}
        self.assertTrue(RUN._candidate_success(audit, 194, 194))
        self.assertFalse(RUN._candidate_success({"valid": False}, 194, 194))
        self.assertFalse(RUN._candidate_success(audit, 143, 194))
        self.assertFalse(RUN._candidate_success(audit, None, 194))

    def test_resume_rejects_group_with_changed_plan_or_code_identity(self):
        item = {"request_id": "sg-001", "attempt_id": 0, "plan_index": 0, "seed": 7}
        code = {"run.py": "code-hash"}
        attempt = {"request_id": "sg-001", "plan_index": 0, "round": 0, "seed": 7,
                   "policy": "shared_initial", "success": False,
                   "checkpoint_fingerprint_sha256": "checkpoint",
                   "trainable_state_sha256": "state", "evaluation_source_sha256": code}
        group = {"request_id": "sg-001", "group_index": 0, "plan_index": 0,
                 "original_seed": 7, "plan_item": item, "seed_manifest": RUN._seed_manifest(item),
                 "run_identity_sha256": "identity", "checkpoint_fingerprint_sha256": "checkpoint",
                 "trainable_state_sha256": "state", "evaluation_source_sha256": code,
                 "status": "running", "arms": {
                     arm: {"attempts": [attempt.copy()], "attempts_used": 1, "success": None}
                     for arm in RUN.ARMS}}
        RUN._validate_group(group, item, "identity", "checkpoint", "state", code)
        changed = {**group, "evaluation_source_sha256": {"run.py": "other"}}
        with self.assertRaisesRegex(ValueError, "another checkpoint or code"):
            RUN._validate_group(changed, item, "identity", "checkpoint", "state", code)
        changed_plan = {**group, "plan_item": {**item, "seed": 9}}
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            RUN._validate_group(changed_plan, item, "identity", "checkpoint", "state", code)

    def test_feedback_reports_recorded_space_group_and_unevaluated_ehull(self):
        text = RUN._checker_feedback({
            "audit": {"valid": False, "reasons": ["formula_header_mismatch"],
                      "facts": {"formula": "Mg2O2", "sg": 194, "atoms": 4, "orbits": 2}},
            "actual_space_group_number": 143,
        }, 194)
        self.assertIn("space group 194", text)
        self.assertIn("decoded space group is 143", text)
        self.assertIn("formula_header_mismatch", text)
        self.assertIn("ehull target has not been evaluated", text)
        self.assertNotIn("<CRYSTAL", text)

    def test_multisegment_denoising_times_and_visibility(self):
        layout = RUN._denoise_layout(
            [5, 90, 11, 91, 7, 90], [99, 99], start_id=90, end_id=91)
        self.assertEqual(layout["time_ids"], [-1, -1, 0, 0, -1, -1, -1, 0, 0])
        self.assertEqual(layout["position_ids"], list(range(6)) + [5, 6, 7])
        allowed = layout["allowed"]
        self.assertEqual(allowed[4], [True, True, True, True, True, False, False, False, False])
        self.assertEqual(allowed[6], [True] * 9)
        self.assertEqual(allowed[7], [True] * 9)

    @unittest.skipUnless(importlib.util.find_spec("torch"),
                         "PyTorch is unavailable in this interpreter")
    def test_single_crystal_denoising_logits_match_frozen_objective(self):
        import torch
        from types import SimpleNamespace
        frozen_denoise_logits = RUN.E.OBJECTIVE.denoise_logits

        class Core(torch.nn.Module):
            def forward(self, inputs_embeds, **_kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds + 0.1)

        class Base(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(128, 8)
                self.model = Core()
                self.lm_head = torch.nn.Linear(8, 128, bias=False)

            def get_input_embeddings(self):
                return self.embedding

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = Base()
                self.time = torch.nn.Embedding(1, 8)
                self.start_id, self.end_id = 90, 91

        model = ToyModel().eval()
        prefix, current = [4, 5, 90], [99, 99, 99, 99]
        expected = frozen_denoise_logits(model, prefix, current, len(prefix))
        actual = RUN._denoise_logits(model, prefix, current)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
