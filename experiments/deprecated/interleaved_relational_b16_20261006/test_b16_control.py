import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
prepare = types.ModuleType("prepare")
for name, value in {
    "END": "<END>", "K": 16, "MASK": "<MASK>", "MAXLEN": 128,
    "MODEL": "unused", "PAD": "<PAD>", "REVISION": "unused",
    "START": "<START>", "STEPS": 32,
}.items():
    setattr(prepare, name, value)
sys.modules["prepare"] = prepare
import model as candidate_model

spec = importlib.util.spec_from_file_location(
    "baseline_model", HERE.parent / "interleaved_relational_blocks_20261005" / "model.py"
)
baseline_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline_model)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=4)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"p": [4], "lead<START>": [5, 1], "crystal": [10, 11, 12, 13]}[text]


class B16ControlTests(unittest.TestCase):
    def test_b16_pack_has_no_ar_group_and_matches_frozen_baseline_noise(self):
        baseline = baseline_model.Hybrid(FakeBase(), 1, 2, 3, 0, block_size=16)
        candidate = candidate_model.Hybrid(
            FakeBase(), 1, 2, 3, 0, block_size=16, clean_crystal_ar_weight=0.0
        )
        example = {
            "prefix": "p", "lead": "lead<START>", "crystal": "crystal",
            "noise_step": 17, "supervise_lead": False,
        }
        old = baseline.pack(example, FakeTokenizer(), "hybrid", torch.Generator().manual_seed(29))
        new = candidate.pack(example, FakeTokenizer(), "hybrid", torch.Generator().manual_seed(29))

        self.assertEqual(candidate.block_size, 16)
        self.assertFalse(any(group["kind"] == "crystal_ar" for group in new["groups"]))
        for key in ("ids", "allowed", "times", "position_ids"):
            self.assertTrue(torch.equal(old[key], new[key]), key)
        self.assertEqual(old["groups"], new["groups"])


if __name__ == "__main__":
    unittest.main()
