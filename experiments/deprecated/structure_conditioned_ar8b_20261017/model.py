"""Bind the pinned Qwen3-8B pure-AR implementation to this task builder."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015/model.py"
_spec = spec_from_file_location("structure_conditioned_ar8b_reference_model", SOURCE)
_reference = module_from_spec(_spec)
_spec.loader.exec_module(_reference)

Qwen3ARBaseline = _reference.Qwen3ARBaseline
ARBaseline = Qwen3ARBaseline
inject_lora = _reference.inject_lora
