import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from model import Qwen3ARBaseline, inject_lora
from prepare import END, EOS, MASK, PAD, START, tasks
from train import (EXPERIMENT, LEARNING_RATE, MATERIALS_PER_UPDATE, UPDATES,
                   check_new_token_gradients, initialize_run_output, learning_rate,
                   local_material_positions, prepare_output, restore_checkpoint,
                   save_checkpoint, state_sha256, trainable_sha256)


class TinyTokenizer:
    def __init__(self, texts):
        self.special_tokens = [START, END, MASK, PAD, EOS]
        chars = sorted({char for text in texts for char in text})
        self.tokens = self.special_tokens + [char for char in chars if char not in self.special_tokens]
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = list(self.tokens)
        self.eos_token = EOS
        self.eos_token_id = self.convert_tokens_to_ids(EOS)
        self.all_special_ids = [self.convert_tokens_to_ids(token) for token in self.special_tokens]

    def __len__(self):
        return len(self.tokens)

    def convert_tokens_to_ids(self, token):
        return self.token_to_id[token]

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            match = next((token for token in self.special_tokens if text.startswith(token)), None)
            if match:
                result.append(self.convert_tokens_to_ids(match))
                text = text[len(match):]
            else:
                result.append(self.convert_tokens_to_ids(text[0]))
                text = text[1:]
        return result

    def decode(self, token_ids, skip_special_tokens=False):
        special = set(self.special_tokens)
        return "".join(self.id_to_token[int(index)] for index in token_ids
                       if not skip_special_tokens or self.id_to_token[int(index)] not in special)


def example_row():
    return {
        "material_id": "mp-tiny",
        "split": "train",
        "plaid_text": "NaCl\nSpacegroup: Fm-3m",
        "structure_description": "NaCl is cubic in space group Fm-3m.",
        "property_labels": {"energy_above_hull": 0.01, "unit": "eV/atom"},
        "core_facts": {
            "cell_formula": "Na4 Cl4", "space_group_number": 225,
            "cell_atom_count": 8, "independent_orbit_count": 2,
        },
    }


def tiny_model():
    row = example_row()
    example_texts = [value for example in tasks(row)
                     for value in example.values() if isinstance(value, str)]
    tok = TinyTokenizer(example_texts)
    token_ids = [tok.convert_tokens_to_ids(value) for value in (START, END, MASK, PAD)]
    config = Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id, tie_word_embeddings=False,
    )
    base = Qwen3ForCausalLM(config)
    inject_lora(base, token_ids)
    return Qwen3ARBaseline(base, *token_ids), tok, token_ids, row


class Qwen3PureARTests(unittest.TestCase):
    def test_causal_mask_and_control_token_gradients(self):
        torch.manual_seed(11)
        model, tok, token_ids, row = tiny_model()
        model.train()
        packed = model.pack(tasks(row)[0], tok, "ar", torch.Generator().manual_seed(42))
        self.assertNotIn(model.mask_id, packed["ids"].tolist())
        self.assertTrue(torch.equal(packed["allowed"][0, 0],
                                    torch.ones_like(packed["allowed"][0, 0]).tril()))
        loss, _ = model.loss(tasks(row), tok, "ar", torch.Generator().manual_seed(42))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = check_new_token_gradients(model, token_ids)
        self.assertGreater(gradients["input_token_grad_norm"], 0)
        self.assertGreater(gradients["output_token_lm_head_lora_b_grad_norm"], 0)

    def test_future_tokens_do_not_change_prior_causal_logits(self):
        torch.manual_seed(13)
        model, tok, _, row = tiny_model()
        model.eval()
        prefix = tok.encode(tasks(row)[0]["prefix"], add_special_tokens=False)
        left, right = tok.encode("N", add_special_tokens=False)[0], tok.encode("C", add_special_tokens=False)[0]
        with torch.inference_mode():
            logits_a = model.logits(torch.tensor(prefix + [left]), "ar", positions=[len(prefix) - 1])
            logits_b = model.logits(torch.tensor(prefix + [right]), "ar", positions=[len(prefix) - 1])
        self.assertTrue(torch.equal(logits_a, logits_b))

    def test_two_rank_accumulation_preserves_global_eight_material_update(self):
        rank0 = local_material_positions(1, 0, 2)
        rank1 = local_material_positions(1, 1, 2)
        self.assertEqual(rank0, [0, 2, 4, 6])
        self.assertEqual(rank1, [1, 3, 5, 7])
        self.assertEqual(sorted(rank0 + rank1), list(range(MATERIALS_PER_UPDATE)))
        self.assertEqual(UPDATES, 909)
        self.assertAlmostEqual(learning_rate(1), LEARNING_RATE / 46)
        self.assertEqual(learning_rate(46), LEARNING_RATE)
        self.assertAlmostEqual(learning_rate(UPDATES), 0)

    def test_state_fingerprint_accepts_bfloat16_rows(self):
        rows = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        self.assertEqual(state_sha256({"rows": rows}), state_sha256({"rows": rows.clone()}))

    def test_rank_zero_creates_shared_output_once(self):
        with tempfile.TemporaryDirectory(dir=EXPERIMENT) as tmp:
            out = Path(tmp) / "run"
            with patch("train.prepare_output", wraps=prepare_output) as prepare:
                self.assertEqual(initialize_run_output(out, None, 1), out.resolve())
                self.assertFalse(out.exists())
                initialize_run_output(out, None, 0)
                initialize_run_output(out, None, 1)
                self.assertEqual(prepare.call_count, 1)

    def test_checkpoint_reload_restores_adapter_control_rows_optimizer_and_logits(self):
        torch.manual_seed(101)
        model, tok, token_ids, row = tiny_model()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        train_generator = torch.Generator().manual_seed(1042)
        loss, _ = model.loss(tasks(row), tok, "ar", train_generator)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        identity = {"fingerprint_sha256": "tiny", "token_ids": {
            START: token_ids[0], END: token_ids[1], MASK: token_ids[2], PAD: token_ids[3]}}
        prefix = torch.tensor(tok.encode(tasks(row)[0]["prefix"], add_special_tokens=False))
        model.eval()
        with torch.inference_mode():
            expected_logits = model.logits(prefix, "ar", positions=[len(prefix) - 1])
        expected_state_hash = trainable_sha256(model)

        with tempfile.TemporaryDirectory(dir=EXPERIMENT) as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(checkpoint_path, model, optimizer, identity, 1, 1, 0, train_generator)
            torch.manual_seed(202)
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        parameter.add_(0.5)
            torch.manual_seed(101)
            reloaded, _, _, _ = tiny_model()
            reloaded_optimizer = torch.optim.AdamW(
                (p for p in reloaded.parameters() if p.requires_grad), lr=1e-3)
            restored = restore_checkpoint(checkpoint_path, reloaded, reloaded_optimizer,
                                          identity, 0, 1, torch.Generator().manual_seed(9),
                                          torch.device("cpu"))
            self.assertEqual(restored["next_material_index"], MATERIALS_PER_UPDATE)
            self.assertEqual(trainable_sha256(reloaded), expected_state_hash)
            self.assertTrue(reloaded_optimizer.state)
            reloaded.eval()
            with torch.inference_mode():
                actual_logits = reloaded.logits(prefix, "ar", positions=[len(prefix) - 1])
            torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
