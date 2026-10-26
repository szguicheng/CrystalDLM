import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from model import Qwen3HybridBaseline, inject_lora
from prepare import END, EOS, K, MASK, PAD, START, STEPS, tasks


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


def tiny_model(seed=11, model_class=Qwen3HybridBaseline, tok=None):
    row = example_row()
    example_texts = [value for example in tasks(row)
                     for value in example.values() if isinstance(value, str)]
    tok = tok or TinyTokenizer(example_texts)
    token_ids = [tok.convert_tokens_to_ids(value) for value in (START, END, MASK, PAD)]
    config = Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id, tie_word_embeddings=False,
    )
    torch.manual_seed(seed)
    base = Qwen3ForCausalLM(config)
    inject_lora(base, token_ids)
    return model_class(base, *token_ids, prediction_offset=0), tok, token_ids, row


def full_mask_example(row):
    return dict(tasks(row)[0], noise_step=STEPS)


class Qwen3HybridTests(unittest.TestCase):
    def test_initial_trainable_state_matches_the_pure_ar_initializer(self):
        hybrid, tok, token_ids, _ = tiny_model(seed=31)
        pure_ar_class = Qwen3HybridBaseline.__bases__[0]
        pure_ar, _, _, _ = tiny_model(seed=31, model_class=pure_ar_class, tok=tok)
        hybrid_params = {name: param for name, param in hybrid.named_parameters()
                         if param.requires_grad}
        pure_ar_params = {name: param for name, param in pure_ar.named_parameters()
                          if param.requires_grad}
        self.assertEqual(list(hybrid_params), list(pure_ar_params))
        self.assertEqual(len(token_ids), 4)
        for name in hybrid_params:
            self.assertTrue(torch.equal(hybrid_params[name], pure_ar_params[name]), name)

    def test_full_mask_labels_use_each_target_position(self):
        model, tok, _, row = tiny_model()
        packed = model.pack(full_mask_example(row), tok, "hybrid",
                            torch.Generator().manual_seed(42))
        crystal = next(group for group in packed["groups"] if group["kind"] == "crystal")
        example = full_mask_example(row)
        start = len(tok.encode(example["prefix"], add_special_tokens=False))
        start += len(tok.encode(example["lead"], add_special_tokens=False))
        target = tok.encode(example["crystal"], add_special_tokens=False) + [model.end_id]
        target += [model.pad_id] * (K - len(target))

        self.assertEqual(crystal["positions"], list(range(start, start + K)))
        self.assertEqual(crystal["labels"], target)
        self.assertTrue(torch.all(packed["ids"][start:start + K] == model.mask_id))

    def test_current_crystal_attention_is_bidirectional(self):
        model, tok, _, row = tiny_model()
        example = full_mask_example(row)
        packed = model.pack(example, tok, "hybrid", torch.Generator().manual_seed(42))
        start = len(tok.encode(example["prefix"], add_special_tokens=False))
        start += len(tok.encode(example["lead"], add_special_tokens=False))
        allowed = packed["allowed"][0, 0]

        self.assertTrue(allowed[start, start + K - 1])
        self.assertTrue(allowed[start + K - 1, start])

    def test_future_crystal_tokens_do_not_change_prior_prefix_logits(self):
        model, tok, _, row = tiny_model(seed=13)
        model.eval()
        prompt = tok.encode(tasks(row)[0]["prefix"], add_special_tokens=False)
        first = tok.convert_tokens_to_ids("N")
        second = tok.convert_tokens_to_ids("C")
        left = torch.tensor(prompt + [model.start_id, first, model.end_id])
        right = torch.tensor(prompt + [model.start_id, second, model.end_id])
        position = len(prompt) - 1
        with torch.inference_mode():
            logits_left = model.logits(left, "hybrid", positions=[position])
            logits_right = model.logits(right, "hybrid", positions=[position])
        self.assertTrue(torch.equal(logits_left, logits_right))

    def test_hybrid_loss_has_finite_mask_time_and_added_token_gradients(self):
        model, tok, token_ids, row = tiny_model()
        model.train()
        examples = [full_mask_example(row), tasks(row)[1]]
        loss, stats = model.loss(examples, tok, "hybrid",
                                 torch.Generator().manual_seed(42))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.tensor(stats["text"])))
        self.assertTrue(torch.isfinite(torch.tensor(stats["crystal"])))
        with self.assertRaises(AssertionError):
            model.loss(examples, tok, "ar", torch.Generator().manual_seed(42))
        loss.backward()

        self.assertTrue(torch.isfinite(model.time.weight.grad).all())
        self.assertGreater(float(model.time.weight.grad.norm()), 0)
        mask_rows = [param.grad[2] for name, param in model.named_parameters()
                     if "trainable_tokens_delta" in name and param.grad is not None]
        self.assertTrue(mask_rows)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in mask_rows))
        self.assertGreater(sum(float(grad.norm()) for grad in mask_rows), 0)

        from train import load_training_module
        engine = load_training_module()
        gradients = engine.check_new_token_gradients(model, token_ids)
        self.assertTrue(gradients["finite"] and gradients["nonzero"])

    def test_crystal_ignores_future_answer_and_readback_stays_causal(self):
        model, tok, _, row = tiny_model(seed=17)
        model.eval()
        left = torch.tensor([model.start_id] + [model.mask_id] * 4 +
                            [model.end_id, tok.convert_tokens_to_ids("N")])
        right = left.clone()
        right[-1] = tok.convert_tokens_to_ids("C")
        with torch.inference_mode():
            first = model.logits(left, "hybrid", span=(1, 5), k=STEPS,
                                 positions=[1, 2, 3, 4])
            second = model.logits(right, "hybrid", span=(1, 5), k=STEPS,
                                  positions=[1, 2, 3, 4])
        self.assertTrue(torch.equal(first, second))
        packed = model.pack(tasks(row)[1], tok, "hybrid", torch.Generator().manual_seed(42))
        start = int((packed["ids"] == model.start_id).nonzero()[0])
        end = int((packed["ids"] == model.end_id).nonzero()[0])
        self.assertTrue(packed["allowed"][0, 0, start + 1, end])
        self.assertTrue(torch.all(packed["times"][start + 1:end + 1] == 0))
        target_position = packed["groups"][0]["positions"][0]
        self.assertFalse(packed["allowed"][0, 0, target_position, target_position + 1])

    def test_engine_binding_runs_preflight_in_hybrid_mode(self):
        from train import CONFIG, EXPERIMENT, load_training_module

        engine = load_training_module()
        self.assertEqual(CONFIG["mode"], "hybrid")
        self.assertEqual(engine.MODEL_MODE, "hybrid")
        self.assertEqual(Path(engine.__file__).resolve(), (EXPERIMENT / "trainer_engine.py").resolve())
        self.assertEqual(engine.Qwen3ARBaseline.__name__, "Qwen3HybridBaseline")

        model, tok, token_ids, row = tiny_model(model_class=engine.Qwen3ARBaseline)
        report = engine.record_preflight_gradient(model, model, tok, row, token_ids, 1)
        self.assertTrue(report["finite"] and report["nonzero"])
        self.assertTrue(torch.isfinite(torch.tensor(report["loss"])))

    def test_full_mask_logits_survive_checkpoint_reload(self):
        from train import EXPERIMENT, load_training_module

        engine = load_training_module()
        model, tok, token_ids, row = tiny_model()
        task = full_mask_example(row)
        generator = torch.Generator().manual_seed(1042)
        optimizer = torch.optim.AdamW((param for param in model.parameters()
                                       if param.requires_grad), lr=1e-3)
        loss, _ = model.loss([task], tok, "hybrid", generator)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        packed = model.pack(task, tok, "hybrid", torch.Generator().manual_seed(42))
        crystal = next(group for group in packed["groups"] if group["kind"] == "crystal")
        positions = [crystal["positions"][index] for index in (0, K // 2, K - 1)]
        span = (crystal["positions"][0], crystal["positions"][0] + K)
        model.eval()
        with torch.inference_mode():
            expected_logits = model.logits(packed["ids"], "hybrid", span=span,
                                           k=STEPS, positions=positions)
        expected_state = engine.trainable_sha256(model)
        identity = {"fingerprint_sha256": "tiny-hybrid", "token_ids": {
            START: token_ids[0], END: token_ids[1], MASK: token_ids[2], PAD: token_ids[3]}}

        with tempfile.TemporaryDirectory(dir=EXPERIMENT) as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            engine.save_checkpoint(checkpoint, model, optimizer, identity, 1, 1, 0, generator)
            # Adapter checkpoints assume the same frozen base, as in the pinned 8B run.
            reloaded, _, _, _ = tiny_model(tok=tok)
            self.assertNotEqual(engine.trainable_sha256(reloaded), expected_state)
            reloaded_optimizer = torch.optim.AdamW(
                (param for param in reloaded.parameters() if param.requires_grad), lr=1e-3)
            engine.restore_checkpoint(checkpoint, reloaded, reloaded_optimizer, identity,
                                      0, 1, torch.Generator().manual_seed(9), torch.device("cpu"))
            self.assertEqual(engine.trainable_sha256(reloaded), expected_state)
            reloaded.eval()
            with torch.inference_mode():
                actual_logits = reloaded.logits(packed["ids"], "hybrid", span=span,
                                                k=STEPS, positions=positions)
            torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
