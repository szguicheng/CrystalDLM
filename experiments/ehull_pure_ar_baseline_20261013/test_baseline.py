import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
OLD_EXPERIMENT = ROOT / "experiments/interleaved_ar_diffusion_20261002"
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("HF_HOME", str(OLD_EXPERIMENT / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from baseline import epoch_order, select_eval_ids
from model import ARBaseline
from prepare import END, EOS, MASK, MAXLEN, MODEL, PAD, REVISION, START, tasks


class PureARBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        except OSError:
            cls.tok = AutoTokenizer.from_pretrained(
                OLD_EXPERIMENT / "data_002/tokenizer", local_files_only=True)
        cls.tok.add_special_tokens({"additional_special_tokens": [START, END, MASK, PAD]})
        assert cls.tok.eos_token == EOS

    def row(self):
        return {
            "material_id": "mp-test",
            "split": "train",
            "plaid_text": "NaCl\nSpacegroup: Fm-3m",
            "structure_description": "NaCl is cubic in space group Fm-3m.",
            "property_labels": {"energy_above_hull": 0.01, "unit": "eV/atom"},
            "core_facts": {
                "cell_formula": "Na4 Cl4", "space_group_number": 225,
                "cell_atom_count": 8, "independent_orbit_count": 2,
            },
        }

    def tiny_model(self):
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(self.tok), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=MAXLEN, eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
        ))
        return ARBaseline(base, *[self.tok.convert_tokens_to_ids(s)
                                  for s in (START, END, MASK, PAD)])

    def test_ar_pack_is_unmasked_and_uses_next_token_positions(self):
        tok = self.tok
        model = self.tiny_model()
        generation, reading = tasks(self.row())
        rng = torch.Generator().manual_seed(42)
        packed_generation = model.pack(generation, tok, "ar", rng)
        packed_reading = model.pack(reading, tok, "ar", rng)

        self.assertNotIn(model.mask_id, packed_generation["ids"].tolist())
        lead = tok.encode(generation["lead"], add_special_tokens=False)
        crystal = tok.encode(generation["crystal"], add_special_tokens=False) + [model.end_id]
        start = len(tok.encode(generation["prefix"], add_special_tokens=False)) + len(lead)
        groups = {group["kind"]: group for group in packed_generation["groups"]}
        self.assertEqual(groups["text"]["labels"], lead)
        self.assertEqual(groups["text"]["positions"], list(range(start - len(lead) - 1, start - 1)))
        self.assertEqual(groups["crystal"]["labels"], crystal)
        self.assertEqual(groups["crystal"]["positions"], list(range(start - 1, start + len(crystal) - 1)))

        target = tok.encode(reading["target"], add_special_tokens=False)
        prefix_len = len(tok.encode(reading["prefix"], add_special_tokens=False))
        self.assertEqual(packed_reading["groups"][0]["labels"], target)
        self.assertEqual(packed_reading["groups"][0]["positions"],
                         list(range(prefix_len - 1, prefix_len + len(target) - 1)))

    def test_ar_logits_ignore_changed_future_tokens(self):
        tok = self.tok
        model = self.tiny_model().eval()
        prefix = tok.encode("User: context.\nAssistant:", add_special_tokens=False)
        first = tok.encode(" NaCl", add_special_tokens=False)[0]
        second = tok.encode(" KBr", add_special_tokens=False)[0]
        self.assertNotEqual(first, second)
        suffix = tok.encode(" future tokens do not affect the prefix", add_special_tokens=False)
        ids_a = torch.tensor(prefix + [first] + suffix)
        ids_b = torch.tensor(prefix + [second] + suffix)

        with torch.inference_mode():
            logits_a = model.logits(ids_a, "ar", positions=[len(prefix) - 1])
            logits_b = model.logits(ids_b, "ar", positions=[len(prefix) - 1])
        self.assertTrue(torch.equal(logits_a, logits_b))

    def test_ar_route_loss_and_gradients_are_finite(self):
        model = self.tiny_model()
        loss, stats = model.loss(tasks(self.row()), self.tok, "ar", torch.Generator().manual_seed(42))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(value is not None and torch.isfinite(torch.tensor(value))
                            for value in stats.values()))
        loss.backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))

    def test_restore_accepts_only_ar_offset_zero_checkpoint(self):
        model = self.tiny_model()
        saved = {"model": model.state_dict(),
                 "config": {"mode": "ar", "prediction_offset": 0}}
        self.assertEqual(model.restore_checkpoint(saved)["mode"], "ar")
        saved["config"] = {"mode": "hybrid", "prediction_offset": 0}
        with self.assertRaises(AssertionError):
            model.restore_checkpoint(saved)

    def test_epoch_and_evaluation_sampling_are_deterministic_and_complete(self):
        rows = [{"material_id": f"mp-{i:04d}"} for i in range(79)]
        ordered = epoch_order(rows, seed=42)
        self.assertEqual([row["material_id"] for row in ordered],
                         [row["material_id"] for row in epoch_order(rows, seed=42)])
        self.assertEqual(sorted(row["material_id"] for row in ordered),
                         sorted(row["material_id"] for row in rows))
        self.assertEqual(len({row["material_id"] for row in ordered}), len(rows))
        self.assertEqual(select_eval_ids(rows, seed=42), select_eval_ids(rows, seed=42))
        self.assertEqual(len(select_eval_ids(rows, seed=42)), 32)


if __name__ == "__main__":
    unittest.main()
