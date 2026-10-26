"""Run the frozen scaled32 evaluator against the full-history training receipts."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCALED_ROOT = HERE
BASE_EVALUATOR = SCALED_ROOT / "evaluate.py"
BASE_EVALUATOR_SHA256 = "e0ca48792dfd942022e49b98907b2bb0518c5c9e20e7f6b54b96f627d4c63a73"
_actual_base_sha = hashlib.sha256(BASE_EVALUATOR.read_bytes()).hexdigest()
if _actual_base_sha != BASE_EVALUATOR_SHA256:
    raise RuntimeError(f"frozen evaluator changed: {_actual_base_sha}")

_BASE_ALIAS = "_frozen_scaled32_evaluate_20261006"
_SPEC = importlib.util.spec_from_file_location(_BASE_ALIAS, BASE_EVALUATOR)
BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_BASE_ALIAS] = BASE
_SPEC.loader.exec_module(BASE)

TRAIN_DIR = SCALED_ROOT / "training_full_history_001"
ARM_DIR = TRAIN_DIR / "joint_ar_diffusion_b32"
DEFAULT_PIN = SCALED_ROOT / "eval_full_history_input_pin.json"
DEFAULT_OUT = SCALED_ROOT / "eval_full_history_final2048_001"
BASE.TRAIN_DIR = TRAIN_DIR
BASE.ARM_DIR = ARM_DIR
BASE.DEFAULT_PIN = DEFAULT_PIN
BASE.DEFAULT_OUT = DEFAULT_OUT

_READER_RECIPE_KEY = "../interleaved_relational_scaled32_20261006/train_full_history.py"
READER_SOURCE_PINS = {
    _READER_RECIPE_KEY: "e2954cf3eea94639782987596799a772b475d6a3b62eada787ad8a4820633112",
    "../interleaved_relational_scaled32_20261006/train_full_history.slurm": "3156e9199cf4c04e5d1649ac35f32968641dc0d6a6e9b4f5b31f2fd563f7f439",
    "../interleaved_relational_scaled32_20261006/test_train_full_history.py": "2183f127a720a51fdb1c8ab79db3bdba1309ea3867e086cca089f22889121783",
}


def _read_pinned_recipe():
    source = SCALED_ROOT / "train_full_history.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != READER_SOURCE_PINS[_READER_RECIPE_KEY]:
        raise RuntimeError(f"full-history trainer source changed: {digest}")
    tree = ast.parse(source.read_text())
    values = [
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "READER_RECIPE"
                for target in node.targets)
    ]
    if len(values) != 1:
        raise ValueError("pinned full-history trainer must define one READER_RECIPE")
    return ast.literal_eval(values[0])


READER_RECIPE = _read_pinned_recipe()
_BASE_VALIDATE_TRAINING_RECEIPTS = BASE.validate_training_receipts
_BASE_SNAPSHOT = BASE._snapshot


def validate_training_receipts(pin, status, checkpoint_receipt, summary):
    validated = _BASE_VALIDATE_TRAINING_RECEIPTS(pin, status, checkpoint_receipt, summary)
    for name, value in (("status", status), ("summary", summary),
                        ("checkpoint receipt", checkpoint_receipt)):
        BASE.require(value.get("reader_recipe") == READER_RECIPE,
                     f"{name} reader recipe differs from the pinned full-history trainer")
    parent_pins = status["code_hashes"]["parent"]
    for path, digest in READER_SOURCE_PINS.items():
        BASE.require(parent_pins.get(path) == digest,
                     f"training receipt lacks the pinned full-history source: {path}")
    return validated


def snapshot_with_adapter_sources(inputs, runtime):
    snapshot = _BASE_SNAPSHOT(inputs, runtime)
    snapshot["full_history_evaluation_sources"] = {
        name: BASE.sha_file(HERE / name) for name in (
            "evaluate_full_history.py", "test_evaluate_full_history.py",
            "evaluate_full_history.slurm",
        )
    }
    return snapshot


def augment_summary(summary):
    summary["training_variant"] = "full_history_reader_repair"
    summary["reader_recipe"] = READER_RECIPE
    return summary


BASE.validate_training_receipts = validate_training_receipts
BASE._snapshot = snapshot_with_adapter_sources


def run(pin_path=DEFAULT_PIN, out=DEFAULT_OUT):
    result = BASE.run(pin_path, out)
    summary_path = Path(result) / "summary.json"
    summary = BASE.read_json(summary_path)
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(augment_summary(summary), ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(summary_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "complete", "output": str(run(args.pin, args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
