"""Strict full-history evaluation of the coverage256 fixed-update B8 run."""
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SCALED = PROJECT / "experiments/interleaved_relational_scaled32_20261006"
FROZEN_PATH = SCALED / "evaluate_full_history_b8.py"
assert hashlib.sha256(FROZEN_PATH.read_bytes()).hexdigest() == "edbccaedd80cc0fb2d064fc7315d2623a5ec7c471c72786f427ff5aa9af9cbf7"
spec = importlib.util.spec_from_file_location("_coverage256_frozen_b8_eval", FROZEN_PATH)
FROZEN_B8 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FROZEN_B8
spec.loader.exec_module(FROZEN_B8)
BASE = FROZEN_B8.BASE
require, sha_file, read_json = BASE.require, BASE.sha_file, BASE.read_json
DATA_DIR = HERE / "data_001"
TRAIN_DIR = HERE / "training_fixed2048_b8_001"
ARM_DIR = TRAIN_DIR / "joint_ar_diffusion_b8"
DEFAULT_PIN = HERE / "eval_fixed2048_b8_input_pin.json"
DEFAULT_OUT = HERE / "eval_fixed2048_b8_001"
RECORDS_SHA = "a84ca310fac45944dea7da64aa960fdf7b14891e6d2c1624a1fbc5df6a8c627c"
MANIFEST_SHA = "483fbc8646b2223ff86cc15b34172cccf25a8c08f9bcac9b08797247ba9b1bbf"
OLD48_SHA = "231f336234d15a216dc9cb800d8dc1c3f30cf90985184ccc0d3e5c26333e95c6"
DATA_HASHES = {"manifest_sha256": MANIFEST_SHA, "records_sha256": RECORDS_SHA}
TRAIN_SOURCE_PINS = {
    "train.py": "944cabeba0094e8f1e0cc520526aca5966b67982ca8d3f1fb562c7a303eff855",
    "train.slurm": "ffe477bda9b27a996d6eeb8529d38141b98d7dc41c118a81213f64e783735e06",
    "test_train.py": "5aefb285554a3b3a6390488404a9e3215da56e0e02fc088590634e74f05ea2c0",
    "prepare_data.py": "fbd9ffaf2e795ca8d33eb77d6b5a85cdd2c8c30be910ac88b5848d31a0dc7c1d",
}
for name, digest in TRAIN_SOURCE_PINS.items():
    require(sha_file(HERE / name) == digest, f"coverage training source changed: {name}")
tree = ast.parse((HERE / "train.py").read_text())
READER_RECIPE = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "READER_RECIPE" for t in node.targets))


def function_source(name):
    node = next(n for n in ast.parse(FROZEN_B8.adapted_source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(FROZEN_B8.adapted_source, node)


# Keep native receipt/reload/parity/metric checks; adapt only this stage's metadata.
validation = function_source("validate_training_receipts")
validation = validation.replace('"scaled32_training_only"', '"coverage256_fixed2048_b8_training_only"')
validation = validation.replace('{"parent", "wrapper", "data"}', '{"parent", "coverage256"}')
validation = validation.replace('hashes["data"]', 'status["data_hashes"]')
validation = validation.replace('"train32": "not_run"', '"train256": "not_run"')
exec(compile(validation, str(SCALED / "evaluate.py"), "exec"), BASE.__dict__)
NATIVE_VALIDATE = BASE.validate_training_receipts
run_source = function_source("run")
assert run_source.count('"trained_case_count": 32') == 1
run_source = run_source.replace('"trained_case_count": 32', '"trained_case_count": 256')
run_source = run_source.replace('"scaled32_final_2048_evaluation"', '"coverage256_fixed2048_b8_evaluation"')
run_source = run_source.replace('cohort["remaining24"]', 'cohort["remaining248"]')
exec(compile(run_source, str(SCALED / "evaluate.py"), "exec"), BASE.__dict__)
model_source = function_source("_load_final_model")
model_source = model_source.replace('config["completed_updates"] == 2048',
    'config["completed_updates"] == 2048 and config["K"] == 2720 and config["MAXLEN"] == 8448')
exec(compile(model_source, str(SCALED / "evaluate.py"), "exec"), BASE.__dict__)
BASE.DATA_DIR, BASE.TRAIN_DIR, BASE.ARM_DIR = DATA_DIR, TRAIN_DIR, ARM_DIR
BASE.EXPECTED_RECORDS_SHA256, BASE.EXPECTED_MANIFEST_SHA256 = RECORDS_SHA, MANIFEST_SHA


def validate_training_exposure(status, train_ids):
    exposure = status["actual_exposure_counts"]
    require(set(exposure["per_material"]) == set(train_ids), "exposure must cover all256 train IDs")
    require(all(exposure["per_material"][i]["tasks"] == {"G": 16, "E": 8, "Q1": 4, "Q2": 4}
                for i in train_ids), "wrong per-case training exposure")
    require(exposure["total"] == {"G": 4096, "E": 2048, "Q1": 1024, "Q2": 1024},
            "wrong aggregate training exposure")


def read_cohort():
    require(sha_file(DATA_DIR / "manifest.json") == MANIFEST_SHA
            and sha_file(DATA_DIR / "records.jsonl") == RECORDS_SHA, "coverage data changed")
    manifest = read_json(DATA_DIR / "manifest.json")
    raw = (DATA_DIR / "records.jsonl").read_bytes()
    old = (SCALED / "data_001/records.jsonl").read_bytes()
    require(sha_file(SCALED / "data_001/records.jsonl") == OLD48_SHA and raw.startswith(old),
            "old48 prefix must remain byte identical")
    require(manifest["status"] == "data_prepared_not_trained"
            and manifest["representation"] == "relational_cnl_v1"
            and manifest["capacity"] == {"K": 2720, "MAXLEN": 8448}, "wrong data stage/capacity")
    rows = [json.loads(line) for line in raw.splitlines()]
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] == "heldout"]
    require(len(rows) == len({r["id"] for r in rows}) == 272 and len(train) == 256 and len(held) == 16,
            "wrong cohort size or duplicate IDs")
    require([r["id"] for r in train] == manifest["cohort"]["train_ids"]
            and [r["id"] for r in held] == manifest["cohort"]["heldout_ids"], "split IDs differ")
    require([r["id"] for r in rows] == [r["id"] for r in train[:8] + held + train[8:]],
            "frozen original cohort order changed")
    require({r["source_hashes"]["source_cif_sha256"] for r in train}.isdisjoint(
        {r["source_hashes"]["source_cif_sha256"] for r in held}), "train/held CIF overlap")
    return {"manifest": manifest, "rows": rows, "seen8": train[:8], "heldout16": held,
            "remaining248": train[8:], "train_ids": [r["id"] for r in train]}


def validate_training_receipts(pin, status, receipt, summary):
    result = NATIVE_VALIDATE(pin, status, receipt, summary)
    require(status["code_hashes"]["coverage256"] == TRAIN_SOURCE_PINS, "wrong coverage source pins")
    for value in (status, receipt, summary):
        require(value["reader_recipe"] == READER_RECIPE, "reader recipe changed")
    for value in (status, summary):
        require(value["K"] == 2720 and value["MAXLEN"] == 8448, "wrong training capacity")
        require(value["initial_fingerprint"] == "e97aed646b7289c06f10be376b37d67cd7aa6a42ad6214b61da008cd85b8404b",
                "wrong initial model fingerprint")
    require(summary["epochs_per_train_case"] == 8, "wrong per-case training repetition")
    validate_training_exposure(status, read_cohort()["train_ids"])
    return result


def configure_capacity(runtime):
    for name in ("prepare", "model_module", "runner"):
        runtime[name].K, runtime[name].MAXLEN = 2720, 8448


def continue_after_seen(runner, seen, criterion=None):
    criterion = criterion or runner.seen8_generation_gate_passed
    runner.coverage256_seen8_generation_gate_passed = bool(criterion(seen))
    return True


def load_runtime():
    runtime = FROZEN_B8.load_runtime()
    configure_capacity(runtime)
    runner = runtime["runner"]
    criterion = runner.seen8_generation_gate_passed
    runner.seen8_generation_gate_passed = lambda seen: continue_after_seen(runner, seen, criterion)
    return runtime


def read_runtime_cohort(inputs, runtime):
    cohort = read_cohort()
    validate_training_exposure(inputs["status"], cohort["train_ids"])
    return cohort


def snapshot(inputs, runtime):
    require(sha_file(SCALED / "evaluate.py") == FROZEN_B8.FROZEN_EVALUATOR_SHA256,
            "frozen base evaluator changed")
    hashes = inputs["code_hashes"]
    parent = {name: sha_file(BASE.JOINT_ROOT / name) for name in hashes["parent"]}
    coverage = {name: sha_file(HERE / name) for name in hashes["coverage256"]}
    require(parent == hashes["parent"] and coverage == TRAIN_SOURCE_PINS == hashes["coverage256"],
            "training source changed after checkpoint creation")
    helpers = {"commit_probe.py": runtime["commit_probe"], "full_commit_fast_eval.py": runtime["fast_eval"],
               "fast_inference_metadata.py": runtime["fast_metadata"],
               "test_fast_inference_metadata.py": runtime["fast_eval"].metadata_check,
               "joint_confidence_eval.py": runtime["joint_eval"]}
    expected = (BASE.EXPECTED_SAMPLER_SHA256, BASE.EXPECTED_FAST_EVAL_SHA256,
                BASE.EXPECTED_FAST_METADATA_SHA256, BASE.EXPECTED_FAST_TEST_SHA256,
                BASE.EXPECTED_JOINT_PARITY_SHA256)
    helper_hashes = {name: sha_file(module.__file__) for name, module in helpers.items()}
    require(list(helper_hashes.values()) == list(expected), "frozen inference helpers changed")
    cohort = read_cohort()
    codec = BASE._fresh_codec_snapshot(inputs, runtime)
    tokens = BASE._fresh_tokenizer_hashes(inputs, runtime)
    require(tokens == cohort["manifest"]["provenance"]["tokenizer_files"], "coverage tokenizer differs")
    return {
        "pin_sha256": sha_file(inputs["pin_path"]),
        "training_status_sha256": sha_file(inputs["paths"]["status_path"]),
        "checkpoint_receipt_sha256": sha_file(inputs["paths"]["checkpoint_receipt_path"]),
        "training_summary_sha256": sha_file(inputs["paths"]["summary_path"]),
        "checkpoint_sha256": sha_file(inputs["paths"]["checkpoint_path"]),
        "final_fingerprint": inputs["fingerprint"],
        "code_hashes": {"parent": parent, "coverage256": coverage, "evaluation_helpers": helper_hashes},
        "data_hashes": DATA_HASHES, "tokenizer_hashes": tokens, "codec_pin": codec,
        "evaluation_sources": {n: sha_file(HERE / n) for n in ("evaluate.py", "evaluate.slurm", "test_evaluate.py")},
        "frozen_B8_evaluator_sha256": sha_file(FROZEN_PATH),
        "frozen_base_evaluator_sha256": sha_file(SCALED / "evaluate.py"),
        "b8_sampler": {"block_size": runtime["commit_probe"].BLOCK_SIZE,
                       "source_sha256": runtime["b8_sampler_source_sha256"],
                       "noise_time_rule": FROZEN_B8.SAMPLER_TIME_RULE},
        "capacity": {name: {"K": runtime[name].K, "MAXLEN": runtime[name].MAXLEN}
                     for name in ("prepare", "model_module", "runner")},
    }


BASE.validate_training_receipts = validate_training_receipts
BASE._load_runtime, BASE._read_cohort, BASE._snapshot = load_runtime, read_runtime_cohort, snapshot


def prepare_input_pin(pin_path=DEFAULT_PIN):
    status = read_json(ARM_DIR / "status.json")
    receipt = read_json(ARM_DIR / "checkpoint_receipt.json")
    summary = read_json(TRAIN_DIR / "summary.json")
    pin = {"status": "verified", "updates": 2048, "training_job": int(status["slurm_job"]),
           "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "final_fingerprint": status["final_fingerprint"], "checkpoint_sha256": receipt["checkpoint_sha256"]}
    validate_training_receipts(pin, status, receipt, summary)
    for key, path in (("status", ARM_DIR / "status.json"), ("checkpoint_receipt", ARM_DIR / "checkpoint_receipt.json"),
                      ("summary", TRAIN_DIR / "summary.json"), ("checkpoint", ARM_DIR / "checkpoint.pt")):
        require(path.is_file(), f"missing terminal artifact: {path}")
        pin[key + "_path"] = path.relative_to(PROJECT).as_posix()
        if key != "checkpoint":
            pin[key + "_sha256"] = sha_file(path)
    with Path(pin_path).open("x") as stream:
        json.dump(pin, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def run(pin_path=DEFAULT_PIN, out=DEFAULT_OUT):
    result = BASE.run(pin_path, out)
    summary_path = Path(result) / "summary.json"
    summary = read_json(summary_path)
    summary.update(training_variant="coverage256_fixed2048_b8", reader_recipe=READER_RECIPE,
                   training_epochs_per_case=8, diagnostic_all_heldout_even_if_seen_gate_failed=True)
    summary["sampling"].update(block_size=8, noise_time_rule=FROZEN_B8.SAMPLER_TIME_RULE)
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(summary_path)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare-pin", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_pin:
        prepare_input_pin(args.pin)
    print(json.dumps({"status": "complete", "output": str(run(args.pin, args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
