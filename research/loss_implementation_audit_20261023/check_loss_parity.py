"""CPU audit: compare real project loss/gradients with HF labeled causal-LM calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import transformers
from transformers import Qwen3Config, Qwen3ForCausalLM

root = Path(sys.argv[1]).resolve()
torch.set_num_threads(1)
assert not torch.cuda.is_available(), "Run with CUDA_VISIBLE_DEVICES empty."


def load(name, relative):
    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load("prepare", "experiments/ehull_interleaved_smoke_20261013/prepare.py")
shared = load("audit_shared_model", "experiments/ehull_interleaved_smoke_20261013/model.py")
adapters = load("audit_adapter_model", "experiments/ehull_pure_ar_qwen3_8b_20261015/model.py")
full = load("audit_full_objective", "experiments/flare_crystal_full_20261019/model.py")


class Tokens:
    symbols = [prepare.START, prepare.END, prepare.MASK, prepare.PAD, prepare.EOS]
    alphabet = list("abcdefghijuvwx ")
    tokens = symbols + alphabet
    eos_token_id = 4

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            token = next((s for s in self.symbols if text.startswith(s)), text[0])
            result.append(self.tokens.index(token))
            text = text[len(token):]
        return result


tok = Tokens()
generation = {"task": "G", "prefix": "uv", "lead": "a" + prepare.START, "crystal": "bcdef"}
reading = {"task": "Q",
           "prefix": "uv" + generation["lead"] + generation["crystal"] + prepare.END + "wx",
           "target": "ghi" + prepare.EOS}
examples = [generation, reading]


def tiny(use_lora=False):
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=len(tok.tokens), hidden_size=32,
                        intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, attention_dropout=0., tie_word_embeddings=False)
    config._attn_implementation = "sdpa"
    base = Qwen3ForCausalLM(config)
    if use_lora:
        adapters.inject_lora(base, [0, 1, 2, 3])
    model_type = adapters.Qwen3ARBaseline if use_lora else shared.Hybrid
    model = model_type(base, 0, 1, 2, 3).eval()
    # Exercise the auxiliary embedding after it has learned, not only at zero init.
    with torch.no_grad():
        model.time.weight[0].normal_(mean=0., std=.02)
    return model


def hf_segment(model, task, part):
    """Independent label-span oracle: HF performs its own next-token shift."""
    if task["task"] == "G":
        prefix = tok.encode(task["prefix"])
        lead = tok.encode(task["lead"])
        body = tok.encode(task["crystal"]) + [model.end_id]
        ids = prefix + lead + body
        span = (len(prefix), len(prefix) + len(lead)) if part == "lead" else (
            (len(prefix) + len(lead), len(ids)) if part == "crystal" else (len(prefix), len(ids)))
    else:
        prefix = tok.encode(task["prefix"])
        ids = prefix + tok.encode(task["target"])
        span = (len(prefix), len(ids))
    x = torch.tensor([ids])
    labels = torch.full_like(x, -100)
    labels[:, span[0]:span[1]] = x[:, span[0]:span[1]]
    # Construct embeddings independently; keep the same documented crystal offset.
    crystal_positions = torch.zeros_like(x, dtype=torch.bool)
    left = ids.index(model.start_id) + 1
    right = ids.index(model.end_id) + 1
    crystal_positions[:, left:right] = True
    embeddings = model.base.get_input_embeddings()(x)
    embeddings = embeddings + model.time.weight[0] * crystal_positions.unsqueeze(-1)
    return model.base(inputs_embeds=embeddings, labels=labels, use_cache=False).loss


def compare(kind, use_lora=False):
    custom, oracle = tiny(use_lora), tiny(use_lora)
    if kind == "parent_segment_weighted":
        actual, _ = custom.loss(examples, tok, "ar", torch.Generator().manual_seed(7))
        expected = (hf_segment(oracle, generation, "lead")
                    + hf_segment(oracle, generation, "crystal")
                    + hf_segment(oracle, reading, "answer")) / 2
    else:
        actual, _ = full.objective_tasks(custom, tok, examples,
                                         torch.Generator().manual_seed(7), "ar")
        expected = (hf_segment(oracle, generation, "whole")
                    + hf_segment(oracle, reading, "answer")) / 2
    actual.backward()
    expected.backward()
    errors = []
    for (name, param), (other_name, other) in zip(custom.named_parameters(), oracle.named_parameters()):
        assert name == other_name
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is not None:
            errors.append(float((param.grad - other.grad).abs().max()))
    loss_error = abs(float(actual.detach()) - float(expected.detach()))
    gradient_error = max(errors)
    assert loss_error < 2e-6, (kind, loss_error)
    assert gradient_error < 2e-5, (kind, gradient_error)
    return {"custom_loss": float(actual.detach()), "hf_oracle_loss": float(expected.detach()),
            "absolute_loss_error": loss_error, "maximum_absolute_gradient_error": gradient_error}


results = {("peft_" if use_lora else "") + kind: compare(kind, use_lora)
           for use_lora in (False, True)
           for kind in ("parent_segment_weighted", "continued_ar_task_weighted")}
model = tiny()
x = torch.tensor([[5, 6, 7, 8, 9, 10]])
logits = model.base(x).logits
shifted = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, len(tok.tokens)),
                                            x[:, 1:].reshape(-1))
wrong = torch.nn.functional.cross_entropy(logits.reshape(-1, len(tok.tokens)), x.reshape(-1))
assert abs(float(shifted.detach() - wrong.detach())) > 1e-4
results["off_by_one_negative_control"] = {"correct_shift_loss": float(shifted.detach()),
                                        "wrong_same_position_loss": float(wrong.detach()),
                                        "different": True}
paths = ["experiments/ehull_interleaved_smoke_20261013/model.py",
         "experiments/ehull_pure_ar_qwen3_8b_20261015/model.py",
         "experiments/flare_crystal_full_20261019/model.py",
         "experiments/flare_crystal_qwen3_8b_20261019/model.py"]
print(json.dumps({"status": "passed", "device": "cpu", "torch": torch.__version__,
                  "transformers": transformers.__version__, "checks": results,
                  "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                    for p in paths}}, indent=2))

