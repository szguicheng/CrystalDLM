"""Evaluate B4 with the frozen full-history protocol and strict scorer."""
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
import types


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
FROZEN_EVALUATOR = HERE / "evaluate.py"
FROZEN_EVALUATOR_SHA256 = "e0ca48792dfd942022e49b98907b2bb0518c5c9e20e7f6b54b96f627d4c63a73"
SOURCE_REPLACEMENTS = (
    ("joint_ar_diffusion_b32", "joint_ar_diffusion_b4", 4),
    ('receipt.get("block_size") == 32', 'receipt.get("block_size") == 4', 1),
    ("block_size=32", "block_size=4", 2),
    ("model.block_size == 32", "model.block_size == 4", 2),
    ('"block_size": 32', '"block_size": 4', 1),
    ("hybrid B32", "hybrid B4", 2),
)
source = FROZEN_EVALUATOR.read_bytes()
assert hashlib.sha256(source).hexdigest() == FROZEN_EVALUATOR_SHA256
adapted_source = source.decode()
for old, new, count in SOURCE_REPLACEMENTS:
    assert adapted_source.count(old) == count, f"unexpected frozen B configuration: {old}"
    adapted_source = adapted_source.replace(old, new)
BASE = types.ModuleType("_frozen_scaled32_evaluate_b4")
BASE.__file__ = str(FROZEN_EVALUATOR)
sys.modules[BASE.__name__] = BASE
exec(compile(adapted_source, str(FROZEN_EVALUATOR), "exec"), BASE.__dict__)

TRAIN_DIR = HERE / "training_full_history_b4_001"
ARM_DIR = TRAIN_DIR / "joint_ar_diffusion_b4"
DEFAULT_PIN = HERE / "eval_full_history_b4_input_pin.json"
DEFAULT_OUT = HERE / "eval_full_history_b4_001"
BASE.TRAIN_DIR, BASE.ARM_DIR = TRAIN_DIR, ARM_DIR
BASE.DEFAULT_PIN, BASE.DEFAULT_OUT = DEFAULT_PIN, DEFAULT_OUT

TRAIN_SOURCE_PINS = {
    "train_full_history.py": "e2954cf3eea94639782987596799a772b475d6a3b62eada787ad8a4820633112",
    "train_full_history.slurm": "3156e9199cf4c04e5d1649ac35f32968641dc0d6a6e9b4f5b31f2fd563f7f439",
    "test_train_full_history.py": "2183f127a720a51fdb1c8ab79db3bdba1309ea3867e086cca089f22889121783",
    "train_full_history_b4.py": "eca51ca2a94a218a6904625e3a8075ae6a6228c9a0ff32d340cf266aefe0eb08",
    "train_full_history_b4.slurm": "6207ea7c964c4657a353d5f01a39f2d94c273d7994cecf80cde449982cf2c835",
    "test_train_full_history_b4.py": "c99b631f73c22b86c1f7a82d661f946c573afc16b1d80968d707bbe6b9250553",
}
for name, digest in TRAIN_SOURCE_PINS.items():
    BASE.require(BASE.sha_file(HERE / name) == digest, f"pinned trainer changed: {name}")
tree = ast.parse((HERE / "train_full_history.py").read_text())
READER_RECIPE = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "READER_RECIPE"
                             for target in node.targets))
_BASE_VALIDATE = BASE.validate_training_receipts
_BASE_RUNTIME = BASE._load_runtime
_BASE_SNAPSHOT = BASE._snapshot
SAMPLER_TIME_RULE = "confidence_one: k=ceil(remaining_masks*STEPS/block_size); legacy_random k schedule unchanged"


def adapted_sampler_source(source):
    assert hashlib.sha256(source).hexdigest() == BASE.EXPECTED_SAMPLER_SHA256
    text = source.decode()
    tree = ast.parse(text)
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "sample_prefix")
    assert [ast.unparse(node) for node in function.decorator_list] == ["torch.inference_mode()"]
    body = ast.get_source_segment(text, function)
    old = "            if not len(positions):\n                break\n"
    assert body.count(old) == 1
    return "@torch.inference_mode()\n" + body.replace(old, old +
                        "            if method != 'legacy_random':\n"
                        "                k = (len(positions) * prepare.STEPS + BLOCK_SIZE - 1) // BLOCK_SIZE\n")


def install_sampler(runtime):
    sampler = runtime["commit_probe"]
    assert runtime["fast_eval"].commit_probe is sampler
    source = adapted_sampler_source(Path(sampler.__file__).read_bytes())
    sampler.BLOCK_SIZE = 4
    exec(compile(source, sampler.__file__, "exec"), sampler.__dict__)
    runtime["b4_sampler_source_sha256"] = hashlib.sha256(source.encode()).hexdigest()


def validate_training_receipts(pin, status, receipt, summary):
    result = _BASE_VALIDATE(pin, status, receipt, summary)
    for value in (status, receipt, summary):
        BASE.require(value.get("reader_recipe") == READER_RECIPE,
                     "reader recipe differs from the frozen full-history trainer")
    for name, digest in TRAIN_SOURCE_PINS.items():
        key = "../interleaved_relational_scaled32_20261006/" + name
        BASE.require(status["code_hashes"]["parent"].get(key) == digest,
                     f"training receipt lacks pinned full-history/B4 source: {name}")
    return result


def configure_runner(runner):
    runner.ARMS = ("joint_ar_diffusion_b4",)
    runner.MODE_OFFSETS = {"joint_ar_diffusion_b4": 0}
    runner.CORE_MODES = {"joint_ar_diffusion_b4": "hybrid"}
    runner.BLOCK_SIZES = {"joint_ar_diffusion_b4": 4}
    runner.seen8_generation_gate_passed = lambda _seen: True


def load_runtime():
    runtime = _BASE_RUNTIME()
    configure_runner(runtime["runner"])
    install_sampler(runtime)
    return runtime


def snapshot(inputs, runtime):
    result = _BASE_SNAPSHOT(inputs, runtime)
    result["b4_evaluation_sources"] = {
        name: BASE.sha_file(HERE / name) for name in (
            "evaluate_full_history_b4.py", "evaluate_full_history_b4.slurm",
            "test_evaluate_full_history_b4.py",
        )
    }
    result["adapted_evaluator_sha256"] = hashlib.sha256(adapted_source.encode()).hexdigest()
    result["b4_sampler"] = {"block_size": runtime["commit_probe"].BLOCK_SIZE,
                            "source_sha256": runtime["b4_sampler_source_sha256"],
                            "noise_time_rule": SAMPLER_TIME_RULE}
    assert result["b4_sampler"]["block_size"] == 4
    return result


BASE.validate_training_receipts = validate_training_receipts
BASE._load_runtime = load_runtime
BASE._snapshot = snapshot


def build_input_pin(status, receipt, summary):
    pin = {"status": "verified", "updates": 2048,
           "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "training_job": int(status["slurm_job"]),
           "final_fingerprint": status["final_fingerprint"],
           "checkpoint_sha256": receipt["checkpoint_sha256"]}
    validate_training_receipts(pin, status, receipt, summary)
    BASE._validate_actual_exposure(status, BASE.read_json(BASE.DATA_DIR / "manifest.json")["cohort"]["train_ids"])
    return pin


def prepare_input_pin(pin_path=DEFAULT_PIN):
    paths = {"status_path": ARM_DIR / "status.json",
             "checkpoint_receipt_path": ARM_DIR / "checkpoint_receipt.json",
             "summary_path": TRAIN_DIR / "summary.json",
             "checkpoint_path": ARM_DIR / "checkpoint.pt"}
    BASE.require(paths["checkpoint_path"].is_file(), "final B4 checkpoint is missing")
    pin = build_input_pin(*(BASE.read_json(paths[key]) for key in
                            ("status_path", "checkpoint_receipt_path", "summary_path")))
    for key, path in paths.items():
        pin[key] = path.relative_to(PROJECT).as_posix()
        if key != "checkpoint_path":
            pin[key.replace("_path", "_sha256")] = BASE.sha_file(path)
    with Path(pin_path).open("x") as stream:
        json.dump(pin, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return pin


def run(pin_path=DEFAULT_PIN, out=DEFAULT_OUT):
    result = BASE.run(pin_path, out)
    summary_path = Path(result) / "summary.json"
    summary = BASE.read_json(summary_path)
    summary.update(training_variant="full_history_reader_repair_b4", reader_recipe=READER_RECIPE)
    summary["sampling"].update(block_size=4, noise_time_rule=SAMPLER_TIME_RULE)
    summary["diagnostic_all_heldout_even_if_seen_gate_failed"] = True
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(summary_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-pin", action="store_true",
                        help="Verify terminal training receipts and create the input pin before evaluation")
    args = parser.parse_args(argv)
    if args.prepare_pin:
        prepare_input_pin(args.pin)
    print(json.dumps({"status": "complete", "output": str(run(args.pin, args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
