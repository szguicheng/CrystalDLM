"""Hybrid Qwen3-8B model with the matched pure-AR LoRA initializer."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015/model.py"
_spec = spec_from_file_location("structure_conditioned_hybrid8b_reference_model", SOURCE)
_reference = module_from_spec(_spec)
_spec.loader.exec_module(_reference)
SHARED_SOURCE = ROOT / "experiments/ehull_interleaved_smoke_20261013/model.py"
_shared_spec = spec_from_file_location("structure_conditioned_hybrid8b_shared_model", SHARED_SOURCE)
_shared = module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared)


class Qwen3HybridBaseline(_reference.Qwen3ARBaseline):
    def loss(self, examples, tok, mode, generator):
        assert mode == "hybrid"
        return _shared.Hybrid.loss(self, examples, tok, mode, generator)

    def restore_checkpoint(self, saved, expected_mode="hybrid", expected_offset=0):
        assert expected_mode == "hybrid" and expected_offset == 0
        return _shared.Hybrid.restore_checkpoint(
            self, saved, expected_mode="hybrid", expected_offset=0)

    def text(self, prefix, tok, mode, trigger=False, limit=160):
        assert mode == "hybrid"
        return _shared.Hybrid.text(self, prefix, tok, mode, trigger, limit)

    def crystal(self, prefix, tok, mode, generator, temperature=.8):
        assert mode == "hybrid"
        return _shared.Hybrid.crystal(self, prefix, tok, mode, generator, temperature)


# The copied trainer imports this historical symbol; the class itself is hybrid-only.
Qwen3ARBaseline = Qwen3HybridBaseline
ARBaseline = Qwen3HybridBaseline
inject_lora = _reference.inject_lora
