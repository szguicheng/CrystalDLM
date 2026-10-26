import os
from pathlib import Path
import unittest
from decimal import Decimal

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "experiments/interleaved_ar_diffusion_20261002/hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from model import Hybrid, attention
from prepare import END, EOS, MAXLEN, MASK, PAD, START, STEPS, tasks
from smoke import load_tokenizer


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = load_tokenizer()

    def row(self):
        return {
            "material_id": "mp-test",
            "split": "train",
            "plaid_text": "NaCl\nSpacegroup: Fm-3m",
            "structure_description": "NaCl is cubic in space group Fm-3m.",
            "structure_description_raw": "RAW CAPTION MUST NOT LEAK",
            "property_labels": {"energy_above_hull": Decimal("0.010474552249999"), "unit": "eV/atom"},
            "core_facts": {
                "cell_formula": "Na4 Cl4",
                "space_group_number": 225,
                "cell_atom_count": 8,
                "independent_orbit_count": 2,
            },
        }

    def test_tasks_keep_the_ehull_label_exact_and_future_answer_out_of_generation(self):
        row = self.row()
        generation, reading = tasks(row)
        history = generation["prefix"] + generation["lead"] + generation["crystal"] + END

        self.assertIn("0.010474552249999 eV/atom", generation["prefix"])
        self.assertEqual(generation["lead"], " " + row["structure_description"] + "\n" + START)
        self.assertEqual(generation["crystal"], row["plaid_text"])
        self.assertEqual(reading["prefix"], history + "\nUser: " + reading["question"] + "\nAssistant:")
        self.assertEqual(reading["target"],
            "The formula is Na4 Cl4. The space group number is 225. "
            "The unit cell contains 8 atoms and 2 independent orbits." + EOS)
        self.assertNotIn("RAW CAPTION MUST NOT LEAK", generation["prefix"] + generation["lead"])
        self.assertNotIn(reading["target"], generation["prefix"] + generation["lead"] + generation["crystal"])

    def test_diffusion_attention_cannot_read_a_future_question_or_answer(self):
        ids = torch.tensor([5, 6, 90, 11, 12, 91, 21, 22, 23])
        allowed, _ = attention(ids, "hybrid", (3, 5), start_id=90, end_id=91)
        self.assertTrue(allowed[0, 0, 3, 2])
        self.assertFalse(allowed[0, 0, 3, 6])
        self.assertFalse(allowed[0, 0, 4, 8])

    def tiny_hybrid(self):
        tok = self.tok
        base = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=MAXLEN, eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        ))
        return Hybrid(base, *[tok.convert_tokens_to_ids(s) for s in (START, END, MASK, PAD)])

    def test_prefix_logits_ignore_changed_future_crystal_tokens(self):
        tok = self.tok
        model = self.tiny_hybrid().eval()
        prefix = tok.encode("User: context.\nAssistant:", add_special_tokens=False)
        lead = tok.encode(" A crystal with a tetragonal cell.\n" + START, add_special_tokens=False)
        first = tok.encode(" NaCl", add_special_tokens=False)[0]
        second = tok.encode(" KBr", add_special_tokens=False)[0]
        self.assertNotEqual(first, second)
        suffix = tok.encode("<CRYSTAL_END>\nUser: read this crystal.\nAssistant: answer",
                            add_special_tokens=False)
        start = len(prefix) + len(lead)
        span = (start, start + 1)
        ids_a = torch.tensor(prefix + lead + [first] + suffix)
        ids_b = torch.tensor(prefix + lead + [second] + suffix)

        with torch.inference_mode():
            logits_a = model.logits(ids_a, "hybrid", span, k=STEPS, positions=[len(prefix) - 1])
            logits_b = model.logits(ids_b, "hybrid", span, k=STEPS, positions=[len(prefix) - 1])
        self.assertTrue(torch.equal(logits_a, logits_b))

    def test_text_and_diffusion_objectives_have_finite_losses_and_gradients(self):
        tok = self.tok
        model = self.tiny_hybrid()
        examples = tasks(self.row())

        for mode in ("hybrid", "ar"):
            model.zero_grad(set_to_none=True)
            loss, stats = model.loss(examples, tok, mode, torch.Generator().manual_seed(42))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(stats["text"] is not None and torch.isfinite(torch.tensor(stats["text"])))
            self.assertTrue(stats["crystal"] is not None and torch.isfinite(torch.tensor(stats["crystal"])))
            loss.backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
