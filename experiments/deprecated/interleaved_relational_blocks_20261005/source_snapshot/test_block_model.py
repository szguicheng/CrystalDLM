import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

import model as block_model
from model import Hybrid, unpack_block
from prepare import K, STEPS


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(16, 4)
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(4, 16)

    def get_input_embeddings(self):
        return self.embedding


class TinyBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class ToyTokenizer:
    all_special_ids = [0, 3, 4, 5, 6]
    eos_token_id = 2
    start_id, end_id, mask_id, pad_id = 3, 4, 5, 6

    def encode(self, text, add_special_tokens=False):
        return {"prefix": [1, 2], "lead": [10, 3], "body": [8, 9], "sample": [1, 3]}[text]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


class ScriptedHybrid(Hybrid):
    def __init__(self, block_size, invalid_end=False):
        super().__init__(TinyBase(), 3, 4, 5, 6, block_size=block_size)
        self.block_size = block_size
        self.invalid_end = invalid_end
        self.first_blocks = []

    def logits(self, ids, mode, span=None, k=0, positions=None, times=None, position_ids=None):
        self.assert_mode = mode
        if k == STEPS:
            self.first_blocks.append((len(ids), ids[span[0]:span[1]].tolist()))
        logits = torch.full((len(positions), 16), -100.)
        for row, position in enumerate(positions):
            global_index = span[0] - 2 + position - span[0]
            if self.invalid_end:
                value = 4 if global_index == 5 else 7 if global_index < 10 else 6
            else:
                value = 4 if global_index == 0 else 6
            logits[row, value] = 100.
        return logits


class BlockModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_block_covers_the_fixed_crystal_budget(self):
        model = Hybrid(TinyBase(), 3, 4, 5, 6)
        self.assertEqual(getattr(model, "block_size", None), K)

    def test_dual_attention_blocks_gold_leak_and_matches_clean_prefix_view(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=64, attention_dropout=0.)
        model = Hybrid(Qwen2ForCausalLM(config), 3, 4, 5, 6, block_size=32).eval()
        model.time.weight.data.normal_(0., .05)
        context = [1, 2, ToyTokenizer.start_id]
        clean = [8, ToyTokenizer.end_id, ToyTokenizer.pad_id, ToyTokenizer.pad_id,
                 12, 13, 14, 15, 16, 17, 18, 19]
        noisy = [ToyTokenizer.mask_id] * 4 + [ToyTokenizer.mask_id] * 4 + [20, 21, 22, 23]
        ids = torch.tensor(context + clean + noisy)
        allowed = block_model.block_attention(ids, len(context), 4)
        logical_positions = torch.tensor([list(range(3)) + list(range(3, 15)) * 2])
        times = torch.tensor([[-1, -1, -1] + [0] * 12 + [17] * 12])
        train_hidden = model.hidden(ids[None], allowed, times, logical_positions)
        train_queries = train_hidden[0, 3 + 12 + 4:3 + 12 + 8]
        train_logits = model.base.lm_head(train_queries)

        changed = torch.tensor(context + clean[:4] + [31, 30, 29, 28, 27, 26, 25, 24]
                               + [30, 31, 30, 31] + noisy[4:8] + [24, 25, 26, 27])
        changed_hidden = model.hidden(changed[None], allowed, times, logical_positions)
        torch.testing.assert_close(train_logits, model.base.lm_head(changed_hidden[0, 19:23]), atol=1e-5, rtol=1e-5)

        inference_ids = torch.tensor(context + clean[:4] + noisy[4:8])
        inference_logits = model.logits(inference_ids, "hybrid", (7, 11), 17, list(range(7, 11)))
        torch.testing.assert_close(train_logits, inference_logits, atol=1e-5, rtol=1e-5)

    def test_pack_supervises_fixed_k_and_uses_corrected_whole_target_weights(self):
        model = Hybrid(TinyBase(), 3, 4, 5, 6, block_size=K)
        tok = ToyTokenizer()
        example = {"prefix": "prefix", "lead": "lead", "crystal": "body", "noise_step": 16}
        seed = 21
        packed = model.pack(example, tok, "hybrid", torch.Generator().manual_seed(seed))
        group = next(g for g in packed["groups"] if g["kind"] == "crystal")
        expected_mask = torch.rand(K, generator=torch.Generator().manual_seed(seed)) < .5
        expected_indices = expected_mask.nonzero().flatten().tolist()
        noisy_start = 4 + K
        expected_target = [8, 9, ToyTokenizer.end_id] + [ToyTokenizer.pad_id] * (K - 3)
        expected_weights = [1., 1., 1.] + [.1] * (K - 3)
        self.assertEqual(group["positions"], [noisy_start + i for i in expected_indices])
        self.assertEqual(group["labels"], [expected_target[i] for i in expected_indices])
        for actual, expected in zip(group["weights"], [expected_weights[i] * 2 for i in expected_indices]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(group["normalizer"], 3 + (K - 3) * .1, places=4)
        self.assertEqual(packed["ids"].numel(), 4 + 2 * K)
        self.assertEqual(packed["ids"][noisy_start:].tolist(),
                         [ToyTokenizer.mask_id if expected_mask[i] else expected_target[i] for i in range(K)])
        self.assertEqual(packed["position_ids"][4:4 + K].tolist(), list(range(4, 4 + K)))
        self.assertEqual(packed["position_ids"][4 + K:].tolist(), list(range(4, 4 + K)))

        example["noise_step"] = STEPS
        full = model.pack(example, tok, "hybrid", torch.Generator().manual_seed(seed))
        full_group = next(g for g in full["groups"] if g["kind"] == "crystal")
        self.assertEqual(len(full_group["positions"]), K)
        self.assertEqual(full_group["positions"], list(range(noisy_start, noisy_start + K)))

    def test_loss_matches_full_weighted_target_and_reports_body_end_pad_counts(self):
        tok = ToyTokenizer()
        example = {"prefix": "prefix", "lead": "lead", "crystal": "body",
                   "noise_step": STEPS, "supervise_lead": False}
        labels = torch.tensor([8, 9, tok.end_id] + [tok.pad_id] * (K - 3))
        weights = torch.tensor([1., 1., 1.] + [.1] * (K - 3))
        masked = torch.full((K,), tok.mask_id)
        for block_size in (K, 32):
            with self.subTest(block_size=block_size):
                model = Hybrid(TinyBase(), 3, 4, 5, 6, block_size=block_size)
                logits = model.base.lm_head(model.base.embedding(masked)).float()
                expected = (F.cross_entropy(logits, labels, reduction="none") * weights).sum() / weights.sum()
                loss, stats = model.loss([example], tok, "hybrid", torch.Generator().manual_seed(3))
                self.assertAlmostEqual(float(loss.detach()), float(expected.detach()), places=5)
                self.assertEqual(model.forward_count, 1)
                self.assertEqual(stats["crystal_metrics"]["body"]["count"], 2)
                self.assertEqual(stats["crystal_metrics"]["end"]["count"], 1)
                self.assertEqual(stats["crystal_metrics"]["pad"]["count"], K - 3)
                self.assertAlmostEqual(stats["crystal_body"] + stats["crystal_end"] + stats["crystal_pad"],
                                       stats["crystal"], places=5)

    def test_sampling_fills_every_block_before_one_global_end_pad_check(self):
        tok = ToyTokenizer()
        model = ScriptedHybrid(64).eval()
        body, raw = model.crystal("sample", tok, "hybrid", torch.Generator().manual_seed(5))
        self.assertEqual(body, "")
        self.assertEqual(len(raw), K)
        self.assertEqual(raw, [ToyTokenizer.end_id] + [ToyTokenizer.pad_id] * (K - 1))
        self.assertEqual(len(model.first_blocks), K // 64)
        self.assertEqual([length for length, _ in model.first_blocks], [2 + 64 * i for i in range(1, K // 64 + 1)])
        self.assertTrue(all(ids[-64:] == [ToyTokenizer.mask_id] * 64 for _, ids in model.first_blocks))

        invalid = ScriptedHybrid(K, invalid_end=True).eval()
        with self.assertRaises(block_model.CrystalBlockError) as caught:
            invalid.crystal("sample", tok, "hybrid", torch.Generator().manual_seed(5))
        self.assertEqual(len(caught.exception.raw_ids), K)
        self.assertEqual(caught.exception.raw_ids[5:7], [ToyTokenizer.end_id, 7])

    def test_unpack_rejects_every_end_pad_boundary_error(self):
        for raw in ([7, 6], [4, 4], [6, 4], [4, 7]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                unpack_block(list(raw), 4, 6)


if __name__ == "__main__":
    unittest.main()
