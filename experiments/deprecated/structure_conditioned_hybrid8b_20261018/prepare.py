"""Reuse the identical structure-request contract from the matched pure-AR run."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "experiments/structure_conditioned_ar8b_20261017/prepare.py"
_spec = spec_from_file_location("hybrid_matched_prepare", PATH)
SOURCE = module_from_spec(_spec)
_spec.loader.exec_module(SOURCE)
for _name in ("END", "EOS", "K", "MASK", "MAXLEN", "MODEL", "PAD", "READ_QUESTION", "REVISION", "START", "STEPS", "EHULL_UPPER_BOUNDS", "ehull_upper_bound", "tasks"):
    globals()[_name] = getattr(SOURCE, _name)
