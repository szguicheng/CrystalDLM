import importlib.util
from pathlib import Path
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import model as candidate


class NumericalContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        from types import SimpleNamespace
        config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, attention_dropout=0.)
        config._attn_implementation = 'sdpa'
        self.model = SimpleNamespace(base=Qwen3ForCausalLM(config).eval(), time=torch.nn.Embedding(33, 32))
        torch.nn.init.zeros_(self.model.time.weight)

    def test_packed_clean_stream_matches_causal_and_noisy_matches_inference(self):
        clean = [1, 2, 3, 4, 5, 6, 7]
        plan = candidate.layout(clean, 2, [True] * 5, 31)
        with torch.no_grad():
            h = candidate.forward_hidden(self.model, plan['ids'], plan['allowed'], plan['positions'], 2)
            ar = self.model.base(torch.tensor([clean])).logits[0]
            self.assertTrue(torch.allclose(self.model.base.lm_head(h[:len(clean)]), ar, atol=1e-6))
            for begin in [2, 6]:
                size = min(4, len(clean)-begin)
                logits = candidate.denoise_logits(self.model, clean[:begin], [31]*size, 2)
                queries = [q for q,t in zip(plan['queries'], plan['target_positions']) if begin<=t<begin+size]
                self.assertTrue(torch.allclose(logits,self.model.base.lm_head(h[queries]),atol=1e-6))
            changed = list(clean); changed[4] = 25
            other = candidate.layout(changed, 2, [True]*5,31)
            h2 = candidate.forward_hidden(self.model,other['ids'],other['allowed'],other['positions'],2)
            self.assertTrue(torch.allclose(h[plan['queries'][:4]],h2[other['queries'][:4]],atol=1e-6))


if __name__ == '__main__':
    unittest.main()
