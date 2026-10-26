import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCORE_PATH = Path(__file__).with_name("score.py")
SPEC = importlib.util.spec_from_file_location("repair_feedback_score", SCORE_PATH)
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def attempt(round_index, *, structural_valid=False, requested_sg_match=False,
            success=False, ehull=None, policy="shared_initial"):
    return {
        "round": round_index,
        "seed": 100 + round_index,
        "policy": policy,
        "structural_valid": structural_valid,
        "requested_sg_match": requested_sg_match,
        "success": success,
        "ehull_proxy_corrected_eV_per_atom": ehull,
        "request_id": "sg-001",
        "plan_index": 0,
        "failure_categories": [] if success else ["strict_invalid"],
        "timing_seconds": {"total": 1.0},
        "cif": None,
    }


class RepairFeedbackScoreTests(unittest.TestCase):
    def test_selects_first_strict_success(self):
        attempts = [
            attempt(0, structural_valid=True, requested_sg_match=False, ehull=-1.0),
            attempt(1, structural_valid=True, requested_sg_match=True, success=True, ehull=9.0,
                    policy="feedback"),
        ]

        self.assertIs(score.select_attempt(attempts), attempts[1])

    def test_failed_group_selects_terminal_attempt_without_property_ranking(self):
        attempts = [
            attempt(0, structural_valid=True, requested_sg_match=False, ehull=-1.0),
            attempt(1, ehull=0.0, policy="independent"),
            attempt(2, ehull=20.0, policy="independent"),
        ]

        self.assertIs(score.select_attempt(attempts), attempts[-1])

    def test_post_relax_recovery_does_not_rescue_a_generated_wrong_sg_group(self):
        class ExactMatcher:
            @staticmethod
            def fit(left, right):
                return left == right

        first = attempt(0, structural_valid=True, requested_sg_match=False)
        second = attempt(0, structural_valid=True, requested_sg_match=True, success=True)
        plan = [{"request_id": "sg-001", "attempt_id": 0},
                {"request_id": "sg-015", "attempt_id": 0}]
        groups = {
            ("sg-001", 0): {"group": {"arms": {"independent": {"attempts": [first]}}}},
            ("sg-015", 0): {"group": {"arms": {"independent": {"attempts": [second]}}}},
        }
        scored = {"score_status": "scored", "converged": True,
                  "structural_valid": True,
                  "post_relax_valid": True, "post_relax_sg_match": True,
                  "ehull_proxy_corrected_eV_per_atom": 0.01,
                  "reduced_formula": "LiO", "_structure": "same",
                  "novel_vs_own_train": True}
        rows = {("sg-001", 0): dict(scored, request_id="sg-001", attempt_id=0),
                ("sg-015", 0): dict(scored, request_id="sg-015", attempt_id=0)}

        result = score._arm_gated_property_counts(plan, groups, rows, "independent", ExactMatcher())

        self.assertEqual(result["generated_strict_successes"], 1)
        self.assertEqual(result["final_post_relax_successes_leq_0_08"], 1)
        self.assertEqual(result["unique_novel_final_successes_leq_0_08"], 1)
        self.assertEqual(result["raw_group_denominator"], 2)
        self.assertEqual(result["per_request"]["sg-001"]["final_post_relax_successes_leq_0_08"], 0)
        self.assertEqual(result["per_request"]["sg-015"]["final_post_relax_successes_leq_0_08"], 1)
        self.assertEqual(result["per_request"]["sg-015"]["raw_group_denominator"], 1)

    def test_unique_representatives_exclude_wrong_generated_space_group_rows(self):
        class ExactMatcher:
            @staticmethod
            def fit(left, right):
                return left == right

        rows = [
            {"request_id": "sg-001", "attempt_id": 0, "reduced_formula": "LiO",
             "_structure": "same", "score_status": "scored", "structural_valid": True,
             "post_relax_valid": True, "converged": True, "post_relax_sg_match": True,
             "novel_vs_own_train": True, "ehull_proxy_corrected_eV_per_atom": 0.01},
            {"request_id": "sg-001", "attempt_id": 1, "reduced_formula": "LiO",
             "_structure": "same", "score_status": "scored", "structural_valid": True,
             "post_relax_valid": True, "converged": True, "post_relax_sg_match": True,
             "novel_vs_own_train": True, "ehull_proxy_corrected_eV_per_atom": 0.01},
        ]

        result = score.gated_unique_novel_counts(rows, {("sg-001", 1)}, ExactMatcher())

        self.assertEqual(result["0.08"]["within_request"], 1)
        self.assertEqual(result["0.08"]["global"], 1)

    def test_export_keeps_all_700_groups_per_arm_and_resumes_by_identity(self):
        with tempfile.TemporaryDirectory(dir=score.EXPERIMENT) as temporary:
            run_dir = Path(temporary)
            checkpoint = "checkpoint-sha256"
            run_identity = "run-identity-sha256"
            plan = []
            for request_id, space_group_number in zip(
                    ("sg-001", "sg-015", "sg-038", "sg-119", "sg-143", "sg-194", "sg-216"),
                    (1, 15, 38, 119, 143, 194, 216)):
                for group_index in range(100):
                    item = {"request_id": request_id, "attempt_id": group_index,
                            "space_group_number": space_group_number,
                            "seed": space_group_number * 1000 + group_index,
                            "plan_index": len(plan)}
                    plan.append(item)
                    shared = attempt(0)
                    arms = {}
                    for arm in ("independent", "feedback"):
                        attempts = [shared, attempt(1, policy=arm), attempt(2, policy=arm)]
                        for row in attempts:
                            row["checkpoint_fingerprint_sha256"] = checkpoint
                            row["request_id"] = request_id
                            row["plan_index"] = item["plan_index"]
                        arms[arm] = {"attempts": attempts, "attempts_used": 3, "success": False}
                    group = {
                        "status": "complete", "request_id": request_id,
                        "group_index": group_index, "plan_index": item["plan_index"],
                        "original_seed": item["seed"], "plan_item": item,
                        "run_identity_sha256": run_identity,
                        "checkpoint_fingerprint_sha256": checkpoint,
                        "arms": arms,
                    }
                    path = (run_dir / "shard-00-of-01" / "groups" / request_id /
                            f"group-{group_index:03d}.json")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(group), encoding="utf-8")

            manifest = {"group_plan": plan, "checkpoint_fingerprint_sha256": checkpoint,
                        "run_identity_sha256": run_identity,
                        "group_plan_sha256": score._json_sha256(plan)}
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = score.export_run(run_dir)
            self.assertEqual(result["raw_attempt_denominator"], 700)
            for arm in ("independent", "feedback"):
                export_dir = run_dir / "selected" / arm
                exported = list((export_dir / "shard-00-of-01" / "attempts").glob("sg-*/attempt-*.json"))
                self.assertEqual(len(exported), 700)
                export_manifest = json.loads((export_dir / "run_manifest.json").read_text())
                self.assertEqual(export_manifest["raw_attempt_denominator"], 700)
                receipt = json.loads(exported[0].read_text())
                self.assertEqual(receipt["loop_selection"]["selected_round"], 2)
                self.assertEqual(receipt["loop_selection"]["checkpoint_fingerprint_sha256"], checkpoint)
                self.assertEqual(receipt["structural_valid"], False)
                if arm == "independent":
                    selected_path = exported[0].resolve()
                    generated_receipt = {"generation_receipt": {
                        "path": str(selected_path), "sha256": score._sha256_file(selected_path)}}
                    score._validate_generation_receipt(generated_receipt, selected_path)
                    generated_receipt["generation_receipt"]["sha256"] = "stale"
                    with self.assertRaisesRegex(ValueError, "different selected receipt"):
                        score._validate_generation_receipt(generated_receipt, selected_path)

            manifest_path = run_dir / "selected" / "feedback" / "run_manifest.json"
            before = manifest_path.read_bytes()
            score.export_run(run_dir)
            self.assertEqual(manifest_path.read_bytes(), before)

            first_group = next((run_dir / "shard-00-of-01" / "groups").glob("sg-*/group-000.json"))
            group = json.loads(first_group.read_text())
            group["tampered_after_export"] = True
            first_group.write_text(json.dumps(group), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                score.export_run(run_dir)


if __name__ == "__main__":
    unittest.main()
