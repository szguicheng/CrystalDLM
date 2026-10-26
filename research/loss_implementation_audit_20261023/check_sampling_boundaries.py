"""Reproduce sampler boundary behaviors without loading any trained model."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

root = Path(sys.argv[1]).resolve()
torch.set_num_threads(1)
assert not torch.cuda.is_available()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load("prepare", "experiments/ehull_interleaved_smoke_20261013/prepare.py")
shared = load("boundary_shared", "experiments/ehull_interleaved_smoke_20261013/model.py")
evaluator = load("boundary_flare_evaluator", "experiments/flare_crystal_full_20261019/evaluate.py")


class Tokens:
    all_special_ids = [0, 1, 2, 3, 4]

    def encode(self, text, add_special_tokens=False):
        return [0]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


model = SimpleNamespace(time=torch.nn.Embedding(33, 32), start_id=0, end_id=1, pad_id=3)


def forced_logits(ids, mode, positions):
    logits = torch.full((1, 5), -torch.inf)
    logits[0, 3 if len(ids) == 1 else 1] = 0
    return logits


model.logits = forced_logits
try:
    shared.Hybrid.crystal(model, "prefix", Tokens(), "ar", torch.Generator().manual_seed(42))
except ValueError as error:
    assert "invalid crystal PAD boundary" in str(error)
    assert error.raw_tokens[:2] == [3, 1]
    pad_case = {"reproduced": True, "raw_prefix": error.raw_tokens[:2],
                "error": "invalid crystal PAD boundary"}
else:
    raise AssertionError("Expected old AR sampler to accept PAD and then reject its own output.")

model.mask_id = 2
states = []


def fake_denoise(model, prefix, current, crystal_start):
    states.append(list(current))
    return torch.zeros((len(current), 20))


def fake_samples(logits, generator, banned_ids):
    positions = [i for i, token in enumerate(states[-1]) if token == model.mask_id]
    tokens, confidence = [5, 1, 6, 7], [.1, 1., .5, .7]
    return torch.tensor([tokens[i] for i in positions]), torch.tensor([confidence[i] for i in positions])


with patch.object(evaluator.OBJECTIVE, "denoise_logits", side_effect=fake_denoise), \
     patch.object(evaluator, "_sample_rows", side_effect=fake_samples):
    block = evaluator._sample_flare_block(
        model, [0], [], 4, None, torch.Generator().manual_seed(42), set())
assert block == [5, 1, 6, 7]
assert states[1] == [2, 1, 2, 2]
assert states[-1] == [2, 1, 6, 7]
truncated, ended = evaluator._truncate_at_end(block, 1)
assert ended and truncated == [5, 1]

torch.manual_seed(42)
config = Qwen3Config(vocab_size=20, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=8, attention_dropout=0.)
config._attn_implementation = "sdpa"
tiny = SimpleNamespace(base=Qwen3ForCausalLM(config).eval(), time=torch.nn.Embedding(33, 32))
torch.nn.init.zeros_(tiny.time.weight)
with torch.inference_mode():
    left = evaluator.OBJECTIVE.denoise_logits(tiny, [8, 0], [2, 1, 5, 6], 2)[0]
    right = evaluator.OBJECTIVE.denoise_logits(tiny, [8, 0], [2, 1, 7, 8], 2)[0]
delta = float((left - right).abs().max())
assert delta > 1e-7

print(json.dumps({"status": "observations_reproduced", "old_ar_pad": pad_case,
                  "flare_end_mid_block": {"states_before_each_forward": states,
                                          "completed_block": block, "retained_tokens": truncated,
                                          "discarded_suffix_can_change_retained_position_logits": True,
                                          "tiny_qwen_max_logit_change": delta,
                                          "scope": "Confirmed boundary mismatch; real-run effect is not quantified."}},
                 indent=2))

