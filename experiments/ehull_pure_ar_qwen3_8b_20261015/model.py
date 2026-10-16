"""Qwen3-8B pure-AR model with in-place PEFT adapters."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from prepare import END, MASK, PAD, START

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ehull_pure_ar_baseline_20261013/model.py"
_spec = spec_from_file_location("ehull_pure_ar_baseline_model", SOURCE)
_shared = module_from_spec(_spec)
_spec.loader.exec_module(_shared)

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "lm_head",
]


def adapter_config(token_ids):
    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        trainable_token_indices=token_ids,
    )


def inject_lora(base, token_ids):
    layers = base.config.num_hidden_layers
    counts = {target: sum(name.rsplit(".", 1)[-1] == target
                         for name, _ in base.named_modules()) for target in TARGET_MODULES}
    assert counts == {**{target: layers for target in TARGET_MODULES[:-1]}, "lm_head": 1}
    config = adapter_config(token_ids)
    peft_model = get_peft_model(base, config, adapter_name="default")
    assert peft_model.get_base_model() is base
    trainable = [(name, param) for name, param in base.named_parameters() if param.requires_grad]
    assert trainable, "PEFT injected no trainable parameters"
    assert any("trainable_tokens_delta" in name for name, _ in trainable), "input token rows are not trainable"
    assert any("lm_head" in name and "lora_B" in name for name, _ in trainable), "lm_head LoRA is missing"
    assert not any("embed_tokens" in name and name.endswith("weight") for name, _ in trainable)
    assert not any("lm_head" in name and name.endswith("base_layer.weight")
                   for name, _ in trainable)
    for _, param in trainable:
        param.data = param.data.float()
    return base


class Qwen3ARBaseline(_shared.ARBaseline):
    @classmethod
    def pretrained(cls, tok, model_path, gradient_checkpointing=True):
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        base.resize_token_embeddings(len(tok), mean_resizing=False)
        if gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.config.use_cache = False
        token_ids = [tok.convert_tokens_to_ids(s) for s in (START, END, MASK, PAD)]
        assert len(set(token_ids)) == 4
        inject_lora(base, token_ids)
        return cls(base, *token_ids, prediction_offset=0)

    def forward(self, examples, tok, mode, generator):
        return self.loss(examples, tok, mode, generator)


# The prior evaluator imports this name from `model` and uses the same model API.
ARBaseline = Qwen3ARBaseline
