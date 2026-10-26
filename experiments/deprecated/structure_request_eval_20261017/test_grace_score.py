import importlib.util
from pathlib import Path
import unittest


SCORER_PATH = Path(__file__).with_name("grace_score.py")
SPEC = importlib.util.spec_from_file_location("grace_score", SCORER_PATH)
grace_score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(grace_score)


class GraceScoreTests(unittest.TestCase):
    def test_proxy_ehull_is_per_atom_and_rejects_zero_sites(self):
        self.assertAlmostEqual(grace_score.proxy_ehull(2.0, 4, 0.48), 0.02)
        with self.assertRaisesRegex(ValueError, "positive"):
            grace_score.proxy_ehull(1.0, 0, 0.0)

    def test_calibration_selection_matches_id_and_composition_and_excludes_eval_ids(self):
        records = [
            {"material_id": "mp-a", "formula": "LiO"},
            {"material_id": "mp-b", "formula": "NaO"},
            {"material_id": "mp-c", "formula": "LiO"},
            {"material_id": "mp-d", "formula": "LiO"},
        ]
        entries = {
            "mp-a": {"formula": "LiO"},
            "mp-b": {"formula": "LiO"},  # same ID is insufficient when composition differs
            "mp-c": {"formula": "LiO"},
            "mp-d": {"formula": "LiO"},
        }
        selected = grace_score.select_calibration_records(
            records, entries, excluded_ids={"mp-a"}, count=2, seed=42)
        self.assertEqual({row["material_id"] for row in selected}, {"mp-c", "mp-d"})
        self.assertEqual(selected, grace_score.select_calibration_records(
            records, entries, excluded_ids={"mp-a"}, count=2, seed=42))

    def test_summary_keeps_all_attempts_in_denominator_and_uses_inclusive_thresholds(self):
        plan = [
            {"request_id": "sg-001", "attempt_id": 0},
            {"request_id": "sg-001", "attempt_id": 1},
            {"request_id": "sg-015", "attempt_id": 0},
        ]
        results = [
            {"request_id": "sg-001", "attempt_id": 0,
             "structural_valid": False, "score_status": "not_scored"},
            {"request_id": "sg-001", "attempt_id": 1,
             "structural_valid": True, "post_relax_valid": True,
             "post_relax_sg_match": True, "score_status": "scored", "converged": True,
             "ehull_proxy_corrected_eV_per_atom": 0.08,
             "unique_novel_thresholds_eV_per_atom": [0.08, 0.1]},
            {"request_id": "sg-015", "attempt_id": 0,
             "structural_valid": True, "post_relax_valid": True,
             "post_relax_sg_match": False, "score_status": "scored", "converged": False,
             "ehull_proxy_corrected_eV_per_atom": -0.01,
             "unique_novel_thresholds_eV_per_atom": []},
        ]
        summary = grace_score.summarize_scores(plan, results)
        self.assertEqual(summary["raw_attempt_denominator"], 3)
        self.assertEqual(summary["structural_valid_attempts"], 2)
        self.assertEqual(summary["threshold_counts"]["0.08"]["endpoint_proxy_leq_diagnostic"], 2)
        self.assertEqual(summary["threshold_counts"]["0.08"]["converged_proxy_leq"], 1)
        self.assertEqual(summary["threshold_counts"]["0.08"]["unique_novel_conditioned_converged_leq"], 1)
        self.assertEqual(summary["threshold_counts"]["0.1"]["unique_novel_conditioned_converged_leq"], 1)

    def test_minimum_periodic_distance_includes_self_image_and_distinct_sites(self):
        from pymatgen.core import Lattice, Structure

        one_site = Structure(Lattice.cubic(0.4), ["H"], [[0, 0, 0]])
        self.assertAlmostEqual(grace_score._minimum_periodic_distance(one_site), 0.4)
        two_sites = Structure(Lattice.cubic(5), ["H", "H"], [[0, 0, 0], [.05, 0, 0]])
        self.assertAlmostEqual(grace_score._minimum_periodic_distance(two_sites), 0.25)

    def test_successful_unique_filters_ignore_unconverged_duplicates(self):
        class ExactMatcher:
            @staticmethod
            def fit(left, right):
                return left == right

        rows = [
            {"request_id": "sg-001", "attempt_id": 0, "_structure": "same",
             "reduced_formula": "LiO", "score_status": "scored", "structural_valid": True,
             "post_relax_valid": True, "converged": False, "post_relax_sg_match": True,
             "novel_vs_own_train": True, "ehull_proxy_corrected_eV_per_atom": 0.01},
            {"request_id": "sg-001", "attempt_id": 1, "_structure": "same",
             "reduced_formula": "LiO", "score_status": "scored", "structural_valid": True,
             "post_relax_valid": True, "converged": True, "post_relax_sg_match": True,
             "novel_vs_own_train": True, "ehull_proxy_corrected_eV_per_atom": 0.01},
            {"request_id": "sg-015", "attempt_id": 0, "_structure": "same",
             "reduced_formula": "LiO", "score_status": "scored", "structural_valid": True,
             "post_relax_valid": True, "converged": True, "post_relax_sg_match": True,
             "novel_vs_own_train": True, "ehull_proxy_corrected_eV_per_atom": 0.01},
        ]
        grace_score._mark_unique_novel(rows, ExactMatcher())
        self.assertEqual(rows[0]["unique_novel_thresholds_eV_per_atom"], [])
        self.assertEqual(rows[1]["unique_novel_thresholds_eV_per_atom"], [0.08, 0.1])
        self.assertEqual(rows[2]["unique_novel_thresholds_eV_per_atom"], [0.08, 0.1])
        self.assertEqual(rows[2]["global_unique_novel_thresholds_eV_per_atom"], [])


if __name__ == "__main__":
    unittest.main()
