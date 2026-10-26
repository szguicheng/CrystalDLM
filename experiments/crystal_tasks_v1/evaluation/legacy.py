"""Load the frozen PLaID audit and historical GRACE scoring functions."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
AUDIT_PATH = ROOT / "experiments/deprecated/interleaved_ar_diffusion_20261002/score.py"
GRACE_PATH = ROOT / "experiments/deprecated/structure_request_eval_20261017/grace_score.py"


def extract_crystal(text):
    start_marker, end_marker = "<CRYSTAL_START>", "<CRYSTAL_END>"
    if text.count(start_marker) == 0 and text.count(end_marker) == 0:
        return text.strip()
    if text.count(start_marker) != 1 or text.count(end_marker) != 1:
        raise ValueError("malformed crystal block markers")
    start, end = text.find(start_marker), text.find(end_marker)
    if end < start:
        raise ValueError("malformed crystal block markers")
    return text[start + len(start_marker):end].strip()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_legacy_audit():
    plaid_path = str(ROOT / "work/PLaID")
    if plaid_path not in sys.path:
        sys.path.insert(0, plaid_path)
    archived_wrong_path = str(AUDIT_PATH.resolve().parents[2] / "work/PLaID")
    had_archived_wrong_path = archived_wrong_path in sys.path
    try:
        module = _load_module("crystal_tasks_v1_legacy_audit", AUDIT_PATH)
    finally:
        if not had_archived_wrong_path and archived_wrong_path in sys.path:
            sys.path.remove(archived_wrong_path)
    module.PROJECT = ROOT
    return module


@lru_cache(maxsize=1)
def load_grace_module():
    module = _load_module("crystal_tasks_v1_legacy_grace", GRACE_PATH)
    module.ROOT = ROOT
    module.EXPERIMENT = ROOT / "experiments/deprecated/structure_request_eval_20261017"
    module.ASSETS = module.EXPERIMENT / "assets"
    module.MODEL_MANIFEST = module.ASSETS / "grace-weight-manifest.json"
    module.HULL_PATH = module.ASSETS / "2023-02-07-ppd-mp.pkl.gz"
    module.SOURCE_VAL = module.ASSETS / "source-val.jsonl.gz"
    module.TRAIN_TRIPLES = ROOT / "research/crystal_training_preflight_20261013/data_001/train-triples.jsonl.gz"
    module.EVALUATION_IDS = ROOT / (
        "experiments/deprecated/ehull_pure_ar_qwen3_8b_20261015/"
        "evaluation_001/evaluation-ids.json"
    )
    module.CALIBRATION_OUT = module.EXPERIMENT / "calibration_001"
    return module
