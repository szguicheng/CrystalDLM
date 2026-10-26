from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pymatgen.core import Lattice, Structure

from experiments.crystal_tasks_v1.evaluation.design import evaluate_design
from experiments.crystal_tasks_v1.evaluation.legacy import (
    load_grace_module,
    load_legacy_audit,
)


CRYSTAL = """Na
Spacegroup: Pm-3m
abc: 4 4 4
angles: 90 90 90
Sites (1)
Na 0 0 0 1a"""


def structure():
    return Structure(Lattice.cubic(4), ["Na"], [[0, 0, 0]])


def attempt(attempt_id, seed, *, request_sg=221, output=CRYSTAL,
            ehull=0.04, ehull_upper=0.08, post_sg=221, with_property=True):
    return {
        "attempt_id": attempt_id,
        "seed": seed,
        "request": {"space_group": request_sg, "ehull_upper": ehull_upper},
        "output": output,
        "property_receipt": {
            "converged": True,
            "post_relax_valid": True,
            "actual_space_group_number_post_relax": post_sg,
            "ehull_proxy_corrected_eV_per_atom": ehull,
            "relaxed_structure": structure().as_dict(),
        } if with_property else None,
    }


class DesignEvaluationTests(unittest.TestCase):
    def test_missing_property_scores_are_not_reported_as_zero_sun(self):
        result = evaluate_design([attempt('unscored', 42, with_property=False)], [], [structure()])
        self.assertEqual(result['summary']['property_evaluation']['status'], 'not_run')
        self.assertIsNone(result['summary']['rates']['sun']['rate'])
        self.assertIsNone(result['multiseed']['rates']['sun']['mean'])
        self.assertEqual(result['summary']['rates']['strict_valid']['rate'], 1.0)
        for name in ('converged', 'post_relax_valid', 'post_relax_sg_match'):
            self.assertIsNone(result['summary']['rates'][name]['rate'])
            self.assertIsNone(result['multiseed']['rates'][name]['mean'])
        failed = attempt('scorer-failed', 42)
        failed['property_receipt'] = {'score_status': 'setup_error', 'error': 'model unavailable'}
        result = evaluate_design([failed], [], [structure()])
        self.assertEqual(result['summary']['property_evaluation']['status'], 'failed')
        self.assertIsNone(result['summary']['rates']['sun']['rate'])

    def test_self_training_is_not_novel_and_test_self_coverage_is_one(self):
        reference = structure()
        result = evaluate_design(
            [attempt("a0", 42, output=f"A cubic sodium crystal.\n<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>")],
            training_references=[reference],
            test_references=[reference],
        )

        rates = result["summary"]["rates"]
        self.assertEqual(rates["stable"], {"k": 1, "n": 1, "rate": 1.0})
        self.assertEqual(rates["novel_unique_stable"], {"k": 0, "n": 1, "rate": 0.0})
        self.assertEqual(rates["sun"], {"k": 0, "n": 1, "rate": 0.0})
        self.assertEqual(rates["cov_precision"], {"k": 1, "n": 1, "rate": 1.0})
        self.assertEqual(rates["cov_recall"], {"k": 1, "n": 1, "rate": 1.0})
        self.assertTrue(result["receipts"][0]["parseable"])
        self.assertTrue(result["receipts"][0]["strict_valid"])
        self.assertIsNone(result["multiseed"]["rates"]["sun"]["sample_std"])

    def test_raw_attempt_denominators_duplicate_classes_failures_and_seed_rates(self):
        attempts = [
            attempt("s1-a", 1, ehull=0.08),
            attempt("s1-b", 1, ehull=0.02),
            attempt(
                "s1-bad", 1,
                output=f"<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>"
                       f"<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>",
                with_property=False,
            ),
            attempt("wrong-request-sg", 2, request_sg=1),
            attempt("post-relax-sg-fail", 2, post_sg=1),
            attempt("property-fail", 2, ehull=0.2),
            attempt("s2-a", 2, ehull=0.1, ehull_upper=0.1),
        ]
        result = evaluate_design(attempts, training_references=[], test_references=[structure()])
        rates = result["summary"]["rates"]

        self.assertEqual(rates["parseable"], {"k": 6, "n": 7, "rate": 6 / 7})
        self.assertEqual(rates["strict_valid"], {"k": 6, "n": 7, "rate": 6 / 7})
        self.assertEqual(rates["requested_sg_match"], {"k": 5, "n": 7, "rate": 5 / 7})
        self.assertEqual(rates["property_threshold_pass"], {"k": 5, "n": 7, "rate": 5 / 7})
        self.assertEqual(rates["stable"], {"k": 3, "n": 7, "rate": 3 / 7})
        self.assertEqual(rates["unique_stable"], {"k": 1, "n": 3, "rate": 1 / 3})
        self.assertEqual(rates["unique_stable_attempt_rate"], {"k": 1, "n": 7, "rate": 1 / 7})
        self.assertEqual(rates["novel_unique_stable"], {"k": 1, "n": 1, "rate": 1.0})
        self.assertEqual(rates["sun"], {"k": 1, "n": 7, "rate": 1 / 7})
        self.assertEqual(rates["cov_precision"], {"k": 6, "n": 6, "rate": 1.0})
        self.assertEqual(rates["cov_recall"], {"k": 1, "n": 1, "rate": 1.0})

        self.assertFalse(result["receipts"][3]["requested_sg_match"])
        self.assertEqual(result["receipts"][2]["parse_error"], "ValueError: malformed crystal block markers")
        self.assertFalse(result["receipts"][4]["post_relax_sg_match"])
        self.assertFalse(result["receipts"][5]["property_threshold_pass"])
        self.assertEqual(result["per_seed"][1]["rates"]["unique_stable_attempt_rate"], {
            "k": 1, "n": 3, "rate": 1 / 3
        })
        self.assertEqual(result["per_seed"][2]["rates"]["unique_stable_attempt_rate"], {
            "k": 1, "n": 4, "rate": 1 / 4
        })
        self.assertEqual(result["per_seed"][1]["rates"]["unique_stable"]["k"], 1)
        self.assertEqual(result["per_seed"][2]["rates"]["unique_stable"]["k"], 1)
        self.assertAlmostEqual(result["multiseed"]["rates"]["sun"]["mean"], 7 / 24)
        self.assertAlmostEqual(
            result["multiseed"]["rates"]["sun"]["sample_std"], (2**0.5) / 24
        )

    def test_marker_failure_cannot_become_strict_valid_from_raw_audit(self):
        legacy = load_legacy_audit()
        raw_audit_success = SimpleNamespace(
            parse_fn_wyckoff=legacy.parse_fn_wyckoff,
            audit=lambda _: {"valid": True, "detected_sg": 221},
        )
        bad = attempt(
            "duplicate-markers", 1,
            output=f"<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>"
                   f"<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>",
        )
        with patch(
            "experiments.crystal_tasks_v1.evaluation.design.load_legacy_audit",
            return_value=raw_audit_success,
        ):
            result = evaluate_design([bad], training_references=[], test_references=[])

        self.assertFalse(result["receipts"][0]["parseable"])
        self.assertFalse(result["receipts"][0]["strict_valid"])
        self.assertFalse(result["receipts"][0]["stable"])
        self.assertEqual(result["summary"]["rates"]["strict_valid"], {
            "k": 0, "n": 1, "rate": 0.0
        })

    def test_grace_loader_points_to_the_archived_project_assets(self):
        module = load_grace_module()
        root = Path(__file__).resolve().parents[3]
        experiment = root / "experiments/deprecated/structure_request_eval_20261017"

        self.assertEqual(module.ROOT, root)
        self.assertEqual(module.EXPERIMENT, experiment)
        self.assertEqual(module.ASSETS, experiment / "assets")
        self.assertEqual(module.HULL_PATH, experiment / "assets/2023-02-07-ppd-mp.pkl.gz")
        self.assertEqual(module.EVALUATION_IDS, root / (
            "experiments/deprecated/ehull_pure_ar_qwen3_8b_20261015/"
            "evaluation_001/evaluation-ids.json"
        ))
