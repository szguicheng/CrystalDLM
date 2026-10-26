import math
import unittest

import torch

from experiments.crystal_tasks_v1.evaluation.losses import (
    aggregate_causal_nll,
    causal_nll,
    complementary_denoising_ce,
    denoising_metrics,
    fixed_complementary_masks,
)


class LossMetricTests(unittest.TestCase):
    def test_causal_nll_shifts_logits_and_attributes_targets_to_their_segments(self):
        logits = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [100.0, 0.0]]]
        )
        labels = torch.tensor([[0, 0, 1, 1]])
        segment_ids = torch.tensor([[0, 1, 2, 3]])

        result = causal_nll(logits, labels, segment_ids)

        first = math.log1p(math.exp(-2.0))
        last = math.log1p(math.exp(2.0))
        self.assertAlmostEqual(result["nll_sum"], 2 * first + last, places=6)
        self.assertEqual(result["token_count"], 3)
        self.assertAlmostEqual(result["nll_mean"], (2 * first + last) / 3, places=6)
        self.assertEqual(
            {name: values["token_count"] for name, values in result["segments"].items()},
            {"input": 0, "description": 1, "crystal": 1, "answer": 1},
        )
        self.assertIsNone(result["segments"]["input"]["nll_mean"])
        for name, expected in {
            "description": first,
            "crystal": first,
            "answer": last,
        }.items():
            self.assertAlmostEqual(result["segments"][name]["nll_sum"], expected, places=6)
            self.assertAlmostEqual(result["segments"][name]["nll_mean"], expected, places=6)

    def test_causal_nll_ignores_prompt_labels_masked_with_minus_100(self):
        logits = torch.zeros((1, 4, 2))
        labels = torch.tensor([[-100, -100, 0, 1]])
        segment_ids = torch.tensor([[0, 0, 3, 3]])

        result = causal_nll(logits, labels, segment_ids)

        self.assertEqual(result["token_count"], 2)
        self.assertEqual(result["segments"]["input"]["token_count"], 0)
        self.assertEqual(result["segments"]["answer"]["token_count"], 2)

    def test_aggregate_causal_nll_is_token_weighted_across_batches(self):
        logits = torch.zeros((2, 4, 2))
        logits[0, 0] = torch.tensor([math.log(8.0), 0.0])
        logits[1, :3, 1] = math.log(8.0)
        labels = torch.tensor([[0, 1, -100, -100], [0, 1, 1, 1]])
        segment_ids = torch.tensor([[0, 3, 0, 0], [0, 1, 2, 3]])

        result = aggregate_causal_nll(
            causal_nll(logits[i : i + 1], labels[i : i + 1], segment_ids[i : i + 1])
            for i in range(2)
        )

        total = math.log(9.0) + 3 * math.log(9.0 / 8.0)
        self.assertAlmostEqual(result["nll_sum"], total, places=6)
        self.assertEqual(result["token_count"], 4)
        self.assertAlmostEqual(result["nll_mean"], total / 4, places=6)
        self.assertEqual(result["segments"]["answer"]["token_count"], 2)
        self.assertAlmostEqual(
            result["segments"]["answer"]["nll_sum"],
            math.log(9.0) + math.log(9.0 / 8.0),
            places=6,
        )
        self.assertAlmostEqual(
            result["segments"]["answer"]["nll_mean"],
            (math.log(9.0) + math.log(9.0 / 8.0)) / 2,
            places=6,
        )

    def test_causal_nll_reports_undefined_means_for_empty_targets(self):
        logits = torch.zeros((1, 4, 2))
        labels = torch.full((1, 4), -100)
        segment_ids = torch.tensor([[0, 1, 2, 3]])

        result = causal_nll(logits, labels, segment_ids)

        self.assertEqual(result["nll_sum"], 0.0)
        self.assertEqual(result["token_count"], 0)
        self.assertIsNone(result["nll_mean"])
        self.assertTrue(
            all(values["nll_mean"] is None for values in result["segments"].values())
        )

    def test_denoising_metrics_report_both_named_normalizations(self):
        result = denoising_metrics(ce_sum=12.0, crystal_tokens=3, text_tokens=2)

        self.assertEqual(
            result,
            {"L_crystal_denoise": 4.0, "L_denoise_current": 2.4},
        )

    def test_denoising_metrics_use_none_for_zero_denominators(self):
        no_crystal = denoising_metrics(ce_sum=3.0, crystal_tokens=0, text_tokens=2)
        no_tokens = denoising_metrics(ce_sum=0.0, crystal_tokens=0, text_tokens=0)

        self.assertIsNone(no_crystal["L_crystal_denoise"])
        self.assertEqual(no_crystal["L_denoise_current"], 1.5)
        self.assertEqual(
            no_tokens,
            {"L_crystal_denoise": None, "L_denoise_current": None},
        )

    def test_fixed_complementary_masks_repeat_and_partition_each_target(self):
        masks = fixed_complementary_masks(9, seed=42, sample_id="mp-1")
        repeated = fixed_complementary_masks(9, seed=42, sample_id="mp-1")

        self.assertTrue(all(torch.equal(a, b) for a, b in zip(masks, repeated)))
        self.assertTrue(torch.all((masks[0].int() + masks[1].int()) == 1).item())

    def test_complementary_denoising_ce_matches_independent_per_token_ce(self):
        first_mask = torch.tensor([True, False, True])
        second_mask = ~first_mask
        targets = torch.tensor([0, 1, 0])
        logits_pair = (
            torch.tensor([[2.0, 0.0], [50.0, -50.0], [0.0, 2.0]]),
            torch.tensor([[50.0, -50.0], [0.0, 2.0], [-50.0, 50.0]]),
        )

        result = complementary_denoising_ce(
            logits_pair, targets, (first_mask, second_mask), text_tokens=2
        )

        expected = 2 * math.log1p(math.exp(-2.0)) + math.log1p(math.exp(2.0))
        self.assertAlmostEqual(result["ce_sum"], expected, places=6)
        self.assertEqual(result["crystal_tokens"], 3)
        self.assertEqual(result["text_tokens"], 2)
        self.assertEqual(result["total_tokens"], 5)
        self.assertAlmostEqual(result["L_crystal_denoise"], expected / 3, places=6)
        self.assertAlmostEqual(result["L_denoise_current"], expected / 5, places=6)

    def test_complementary_denoising_ce_rejects_overlap_and_missing_coverage(self):
        logits = torch.zeros((3, 2))
        targets = torch.tensor([0, 1, 0])
        invalid_masks = (
            (torch.tensor([True, True, False]), torch.tensor([False, True, True])),
            (torch.tensor([True, False, False]), torch.tensor([False, False, True])),
        )
        for masks in invalid_masks:
            with self.subTest(masks=masks):
                with self.assertRaises(ValueError):
                    complementary_denoising_ce((logits, logits), targets, masks, 0)


if __name__ == "__main__":
    unittest.main()
