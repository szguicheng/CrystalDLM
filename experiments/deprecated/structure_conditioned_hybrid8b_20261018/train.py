"""Preflight or train the frozen structure-conditioned Qwen3-8B hybrid task."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
REFERENCE_EXPERIMENT = ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015"
MATCHED_AR_EXPERIMENT = ROOT / "experiments/structure_conditioned_ar8b_20261017"
BASELINE_DIR = ROOT / "experiments/ehull_pure_ar_baseline_20261013"
SHARED_DIR = ROOT / "experiments/ehull_interleaved_smoke_20261013"
OLD_MODEL_DIR = ROOT / "experiments/interleaved_ar_diffusion_20261002"
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
assert CONFIG["mode"] == "hybrid"
DEFAULT_DATA = ROOT / CONFIG["data"]["directory"]
DEFAULT_MODEL = ROOT / "models/Qwen3-8B-b968826d9c46"
_TRAINING = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_paths():
    sources = {
        "train.py": EXPERIMENT / "train.py",
        "trainer_engine.py": EXPERIMENT / "trainer_engine.py",
        "model.py": EXPERIMENT / "model.py",
        "prepare.py": EXPERIMENT / "prepare.py",
        "config.json": EXPERIMENT / "config.json",
        "data-contract-audit.json": EXPERIMENT / "data-contract-audit.json",
        "matched-task-audit.json": EXPERIMENT / "matched-task-audit.json",
        "original_engine.py": REFERENCE_EXPERIMENT / "train.py",
        "reference_model.py": REFERENCE_EXPERIMENT / "model.py",
        "reference_prepare.py": REFERENCE_EXPERIMENT / "prepare.py",
        "matched_ar_prepare.py": MATCHED_AR_EXPERIMENT / "prepare.py",
        "baseline.py": BASELINE_DIR / "baseline.py",
        "baseline_model.py": BASELINE_DIR / "model.py",
        "baseline_prepare.py": BASELINE_DIR / "prepare.py",
        "shared_model.py": SHARED_DIR / "model.py",
        "shared_prepare.py": SHARED_DIR / "prepare.py",
        "strict_plaid_parser.py": OLD_MODEL_DIR / "score.py",
        "plaid_utils.py": ROOT / "work/PLaID/utils.py",
    }
    return {name: path.resolve(strict=True) for name, path in sources.items()}


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprints():
    return {name: _sha256_file(path) for name, path in _source_paths().items()}


def _read_frozen_data(data_dir, baseline):
    data_dir = Path(data_dir).resolve(strict=True)
    assert data_dir.is_relative_to(ROOT.resolve())
    expected_directory = CONFIG["data"]["directory"]
    assert data_dir.relative_to(ROOT.resolve()).as_posix() == expected_directory
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = data_dir / "train-triples.jsonl.gz"
    val_path = data_dir / "val-triples.jsonl.gz"
    data_hashes = {
        "manifest_sha256": _sha256_file(manifest_path),
        "train_sha256": _sha256_file(train_path),
        "val_sha256": _sha256_file(val_path),
        "manifest_sources": manifest["sources"],
        "config_sha256": _sha256_file(EXPERIMENT / "config.json"),
        "data_contract_audit_sha256": _sha256_file(EXPERIMENT / "data-contract-audit.json"),
        "matched_task_audit_sha256": _sha256_file(EXPERIMENT / "matched-task-audit.json"),
    }
    expected = CONFIG["data"]
    assert data_hashes["manifest_sha256"] == expected["manifest_sha256"]
    assert data_hashes["train_sha256"] == expected["train_sha256"]
    assert data_hashes["val_sha256"] == expected["validation_sha256"]
    for name, path in (("train-triples.jsonl.gz", train_path),
                       ("val-triples.jsonl.gz", val_path)):
        assert _sha256_file(path) == manifest["artifacts"][name]["sha256"]
    train_rows = baseline.read_rows(train_path, "train")
    val_rows = baseline.read_rows(val_path, "val")
    assert len(train_rows) == manifest["counts"]["train"] == CONFIG["data"]["train_materials"]
    assert len(val_rows) == manifest["counts"]["val"] == CONFIG["data"]["validation_materials"]
    assert manifest["total"] == len(train_rows) + len(val_rows)
    train_ids = [row["material_id"] for row in train_rows]
    val_ids = [row["material_id"] for row in val_rows]
    assert len(set(train_ids)) == len(train_rows) and set(train_ids).isdisjoint(val_ids)
    return manifest, data_hashes, train_rows, val_rows


def load_training_module():
    global _TRAINING
    if _TRAINING is not None:
        return _TRAINING

    prepare = _load_module("structure_conditioned_hybrid8b_prepare", EXPERIMENT / "prepare.py")
    previous_prepare = sys.modules.get("prepare")
    previous_model = sys.modules.get("model")
    sys.modules["prepare"] = prepare
    try:
        model = _load_module("structure_conditioned_hybrid8b_model", EXPERIMENT / "model.py")
        sys.modules["model"] = model
        training = _load_module(
            "structure_conditioned_hybrid8b_engine", EXPERIMENT / "trainer_engine.py")
    finally:
        if previous_prepare is None:
            sys.modules.pop("prepare", None)
        else:
            sys.modules["prepare"] = previous_prepare
        if previous_model is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = previous_model

    training.EXPERIMENT = EXPERIMENT.resolve()
    training.DEFAULT_DATA = DEFAULT_DATA.resolve()
    training.MODEL_MODE = CONFIG["mode"]
    training.EXPECTED_DATA_HASHES = {
        "manifest_sha256": CONFIG["data"]["manifest_sha256"],
        "train_sha256": CONFIG["data"]["train_sha256"],
        "val_sha256": CONFIG["data"]["validation_sha256"],
    }
    training.EXPECTED_ORDER_SHA256 = CONFIG["data"]["sample_order_sha256"]
    training.EXPECTED_EVAL_IDS_SHA256 = CONFIG["data"]["evaluation_ids_sha256"]
    training.tasks = prepare.tasks
    training.BASELINE.tasks = prepare.tasks
    training.BASELINE.validate_lengths.__globals__["tasks"] = prepare.tasks
    training.BASELINE.validate_lengths.__globals__["K"] = training.K
    training.BASELINE.validate_lengths.__globals__["MAXLEN"] = prepare.MAXLEN
    training.source_fingerprints = _source_fingerprints
    training.read_frozen_data = lambda data_dir: _read_frozen_data(data_dir, training.BASELINE)

    original_identity = training.experiment_identity

    def experiment_identity(*args, **kwargs):
        identity = original_identity(*args, **kwargs)
        identity["experiment_path"] = str(EXPERIMENT.resolve())
        identity["source_paths"] = {name: str(path) for name, path in _source_paths().items()}
        identity["model_mode"] = CONFIG["mode"]
        identity["data_contract"] = {
            "id": CONFIG["experiment"],
            "space_group_source": "core_facts.space_group_number",
            "ehull_upper_bounds_eV_per_atom": list(prepare.EHULL_UPPER_BOUNDS),
        }
        identity["fingerprint_sha256"] = training.json_sha256(
            {key: value for key, value in identity.items() if key != "fingerprint_sha256"})
        return identity

    training.experiment_identity = experiment_identity
    original_finalize = training.finalize_identity

    def finalize_identity(identity, data_dir, model_path, tok):
        identity["experiment_path"] = str(EXPERIMENT.resolve())
        identity["source_paths"] = {name: str(path) for name, path in _source_paths().items()}
        identity["model_mode"] = CONFIG["mode"]
        return original_finalize(identity, Path(data_dir).resolve(), Path(model_path).resolve(), tok)

    training.finalize_identity = finalize_identity
    assert training.TRAIN_MATERIALS == CONFIG["data"]["train_materials"]
    assert training.VAL_MATERIALS == CONFIG["data"]["validation_materials"]
    assert training.UPDATES == CONFIG["training"]["updates"]
    assert training.LEARNING_RATE == CONFIG["training"]["learning_rate"]
    _TRAINING = training
    return training


def __getattr__(name):
    return getattr(load_training_module(), name)


def main(argv=None):
    from prepare import REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "train"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=REVISION)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args(argv)
    assert args.checkpoint is None
    assert args.resume is None or args.mode == "train"
    assert args.preflight_receipt is None or args.mode == "train"
    assert args.stop_after_update is None or args.mode == "train"
    assert args.mode != "preflight" or args.resume is None
    load_training_module().run_training(args, args.mode)


if __name__ == "__main__":
    main()
