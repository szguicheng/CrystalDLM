"""Bind the frozen G/Q task builder and token contract from the 1.5B run."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ehull_pure_ar_baseline_20261013/prepare.py"
_spec = spec_from_file_location("ehull_pure_ar_baseline_prepare", SOURCE)
_source = module_from_spec(_spec)
_spec.loader.exec_module(_source)

for _name in ("END", "EOS", "K", "MASK", "MAXLEN", "PAD", "READ_QUESTION", "START", "STEPS", "tasks"):
    globals()[_name] = getattr(_source, _name)

MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
