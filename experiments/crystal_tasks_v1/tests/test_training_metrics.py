import math
import unittest
from types import SimpleNamespace

import torch

from experiments.crystal_tasks_v1.training_metrics import (
    EpochCheckpointCallback,
    _batch_token_stats,
    _format_token_stats,
    _segment_ids_from_labels,
)


class TrainingMetricTests(unittest.TestCase):
    def test_design_ce_uses_shifted_labels_and_token_segments(self):
        logits = torch.tensor(
            [[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]]
        )
        labels = torch.tensor([[-100, 0, 1, 0, 1, 0]])
        segments = torch.tensor([[0, 1, 1, 2, 2, 1]])

        stats = _batch_token_stats(logits, labels, "ar_design", segments)
        metrics = _format_token_stats(stats.sum(dim=0), "train")

        description_ce = (
            math.log1p(2 * math.exp(-2))
            + math.log1p(2 * math.exp(-3))
            + math.log1p(2 * math.exp(-1))
        ) / 3
        crystal_ce = (math.log(math.e + 2) + math.log(3)) / 2
        self.assertAlmostEqual(metrics["train_description_ce"], description_ce, places=6)
        self.assertAlmostEqual(metrics["train_crystal_ce"], crystal_ce, places=6)
        self.assertNotIn("train_answer_ce", metrics)
        self.assertEqual(metrics["train_description_tokens"], 3)
        self.assertEqual(metrics["train_crystal_tokens"], 2)
        self.assertEqual(metrics["train_answer_tokens"], 0)

    def test_design_segments_come_from_start_end_and_trailing_eos_labels(self):
        design_labels = torch.tensor([[-100, 14, 151669, 27, 151670, 151645]])
        understanding_labels = torch.tensor([[-100, 5, 8]])

        self.assertEqual(
            _segment_ids_from_labels(design_labels, "ar_design").tolist(),
            [[0, 1, 1, 2, 2, 1]],
        )
        self.assertEqual(
            _segment_ids_from_labels(understanding_labels, "ar_understanding").tolist(),
            [[0, 3, 3]],
        )

    def test_token_nll_aggregates_sums_and_counts_across_batches(self):
        first = _batch_token_stats(
            torch.zeros((1, 2, 2)),
            torch.tensor([[-100, 0]]),
            "ar_understanding",
            torch.tensor([[0, 3]]),
        )
        second = _batch_token_stats(
            torch.tensor([[[4.0, 0.0], [4.0, 0.0], [0.0, 0.0]]]),
            torch.tensor([[-100, 1, 1]]),
            "ar_understanding",
            torch.tensor([[0, 3, 3]]),
        )

        metrics = _format_token_stats(first.sum(dim=0) + second.sum(dim=0), "eval", True)
        expected_sum = math.log(2) + 2 * math.log1p(math.exp(4))

        self.assertAlmostEqual(metrics["eval_answer_ce"], expected_sum / 3, places=6)
        self.assertAlmostEqual(metrics["eval_token_nll"], expected_sum / 3, places=6)
        self.assertEqual(metrics["eval_answer_tokens"], 3)
        self.assertEqual(metrics["eval_description_tokens"], 0)
        self.assertEqual(metrics["eval_crystal_tokens"], 0)
        self.assertNotIn("eval_description_ce", metrics)
        self.assertNotIn("eval_crystal_ce", metrics)

    def test_epoch_callback_requests_save_and_evaluate(self):
        control = SimpleNamespace(should_evaluate=False, should_save=False)

        result = EpochCheckpointCallback().on_epoch_end(None, None, control)

        self.assertIs(result, control)
        self.assertTrue(control.should_evaluate)
        self.assertTrue(control.should_save)


if __name__ == "__main__":
    unittest.main()
