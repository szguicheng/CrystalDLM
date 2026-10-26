import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as joint_model
from prepare import START

REFERENCE_PATH = HERE.parent / "interleaved_relational_blocks_20261005" / "model.py"
spec = importlib.util.spec_from_file_location("blocks_model", REFERENCE_PATH)
reference_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_model)
for module in (joint_model, reference_model):
    module.K, module.STEPS, module.MAXLEN = 8, 4, 64


class FakeBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(8, 32)

    def get_input_embeddings(self):
        return self.embedding


class FakeTokenizer:
    all_special_ids = [0, 1, 2, 3]
    eos_token_id = 4

    def encode(self, value, add_special_tokens=False):
        if value == "p":
            return [4]
        if value == "lead":
            return [5]
        if value.endswith(START):
            return self.encode(value[:-len(START)], add_special_tokens=False) + [1]
        if value == "ABCD":
            return [10, 11, 12, 13]
        raise AssertionError(f"unexpected text: {value!r}")


def make_models(weight):
    torch.manual_seed(11)
    base = FakeBase()
    kwargs = dict(start_id=1, end_id=2, mask_id=3, pad_id=0, block_size=4)
    reference = reference_model.Hybrid(copy.deepcopy(base), **kwargs)
    candidate = joint_model.Hybrid(copy.deepcopy(base), **kwargs,
                                   clean_crystal_ar_weight=weight)
    return reference, candidate


def example():
    return {"prefix": "p", "lead": "lead" + START, "crystal": "ABCD",
            "noise_step": 4, "supervise_lead": False}


class JointModelTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_clean_ar_queries_align_labels_without_current_or_noisy_leak(self):
        _, model = make_models(0.25)
        packed = model.pack(example(), self.tok, "hybrid", torch.Generator().manual_seed(7))
        group = next(group for group in packed["groups"] if group["kind"] == "crystal_ar")
        start, length, noisy_start = 3, 5, 3 + joint_model.K

        self.assertEqual(group["positions"], list(range(start - 1, start + length - 1)))
        self.assertEqual(group["labels"], [10, 11, 12, 13, 2])
        self.assertEqual(group["weights"], [0.25] * length)
        self.assertEqual(group["normalizer"], length)
        for i, query in enumerate(group["positions"]):
            target = start + i
            self.assertFalse(packed["allowed"][0, 0, query, target])
            self.assertFalse(packed["allowed"][0, 0, query, noisy_start:].any())

    def test_joint_loss_adds_weighted_clean_mean_in_one_hidden_forward(self):
        reference, model = make_models(0.25)
        packed = model.pack(example(), self.tok, "hybrid",
                            torch.Generator().manual_seed(17))
        queries = torch.tensor([2, 3, 4, 5, 6])
        labels = torch.tensor([10, 11, 12, 13, 2])
        self.assertEqual(packed["ids"][queries].tolist(), [1, 10, 11, 12, 13])
        self.assertEqual(packed["times"][queries].tolist(), [-1, 0, 0, 0, 0])
        embedded = model.base.get_input_embeddings()(packed["ids"][queries])
        times = packed["times"][queries]
        embedded = embedded + model.time(times.clamp_min(0)) * (times >= 0).unsqueeze(-1)
        clean_mean = F.cross_entropy(model.base.lm_head(embedded).float(), labels)
        baseline, _ = reference.loss([example()], self.tok, "hybrid",
                                     torch.Generator().manual_seed(17))
        joint, stats = model.loss([example()], self.tok, "hybrid",
                                  torch.Generator().manual_seed(17))

        expected = 0.25 * clean_mean
        torch.testing.assert_close(torch.tensor(stats["crystal_ar"]), expected,
                                   rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(joint - baseline, torch.tensor(stats["crystal_ar"]),
                                   rtol=1e-5, atol=1e-6)
        self.assertEqual(reference.forward_count, 1)
        self.assertEqual(model.forward_count, 1)
        self.assertEqual(stats["crystal_noise_steps"], [4])
        plain = {"prefix": "p", "target": "ABCD"}
        _, plain_stats = model.loss([plain], self.tok, "hybrid",
                                    torch.Generator().manual_seed(17))
        self.assertEqual(plain_stats["crystal_noise_steps"], [])

    def test_zero_weight_matches_parent_pack_loss_and_sampler(self):
        reference, model = make_models(0.0)
        self.assertEqual(set(reference.state_dict()), set(model.state_dict()))
        for key, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)

        for mode in ("hybrid", "ar"):
            old_pack = reference.pack(example(), self.tok, mode,
                                      torch.Generator().manual_seed(23))
            new_pack = model.pack(example(), self.tok, mode,
                                  torch.Generator().manual_seed(23))
            for key in ("ids", "allowed", "times", "position_ids"):
                self.assertTrue(torch.equal(old_pack[key], new_pack[key]), key)
            self.assertEqual(old_pack["groups"], new_pack["groups"])
            old_loss, _ = reference.loss([example()], self.tok, mode,
                                         torch.Generator().manual_seed(31))
            new_loss, _ = model.loss([example()], self.tok, mode,
                                     torch.Generator().manual_seed(31))
            torch.testing.assert_close(old_loss, new_loss)

        current = torch.tensor([3, 3, 2, 3])
        predicted = torch.tensor([8, 9, 7, 6])
        old_step = reference_model.reverse_step(current, predicted, 2, 3,
                                                torch.Generator().manual_seed(37))
        new_step = joint_model.reverse_step(current, predicted, 2, 3,
                                            torch.Generator().manual_seed(37))
        self.assertTrue(torch.equal(old_step, new_step))

    def test_checkpoint_weight_must_match_and_legacy_requires_zero(self):
        reference, model = make_models(1.0)
        saved = {"model": model.state_dict(), "config": {
            "mode": "hybrid", "prediction_offset": 0, "block_size": 4,
            "clean_crystal_ar_weight": 1.0}}
        model.restore_checkpoint(saved, expected_clean_crystal_ar_weight=1.0)
        with self.assertRaises(AssertionError):
            model.restore_checkpoint(saved, expected_clean_crystal_ar_weight=0.0)

        old_saved = {"model": reference.state_dict(), "config": {
            "mode": "hybrid", "prediction_offset": 0, "block_size": 4}}
        _, zero_weight = make_models(0.0)
        zero_weight.restore_checkpoint(old_saved, expected_clean_crystal_ar_weight=0.0)
        with self.assertRaises(AssertionError):
            model.restore_checkpoint(old_saved)


if __name__ == "__main__":
    unittest.main()
