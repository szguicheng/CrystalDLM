"""Reuse the exact frozen G/Q task construction from the shared smoke run."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ehull_interleaved_smoke_20261013/prepare.py"
_spec = spec_from_file_location("ehull_interleaved_smoke_prepare", SOURCE)
_source = module_from_spec(_spec)
_spec.loader.exec_module(_source)

for _name in ("END", "EOS", "K", "MASK", "MAXLEN", "MODEL", "PAD", "PREDICTION_OFFSET",
              "READ_QUESTION", "REVISION", "START", "STEPS", "tasks"):
    globals()[_name] = getattr(_source, _name)
