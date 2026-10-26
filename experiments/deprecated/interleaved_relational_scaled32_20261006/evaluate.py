"""Run the frozen final-2048 scaled32 evaluation on an allocated A100."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import socket
import sys


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
JOINT_ROOT = PROJECT / "experiments/interleaved_relational_joint_20261006"
BLOCK_ROOT = PROJECT / "experiments/interleaved_relational_blocks_20261005"
DIAGNOSTIC_ROOT = BLOCK_ROOT / "diagnostics/root_cause_20261006"
DATA_DIR = HERE / "data_001"
TRAIN_DIR = HERE / "training_grounded_001"
ARM_DIR = TRAIN_DIR / "joint_ar_diffusion_b32"
DEFAULT_PIN = HERE / "eval_input_pin.json"
DEFAULT_OUT = HERE / "eval_grounded_final2048_001"

EXPECTED_RECORDS_SHA256 = "231f336234d15a216dc9cb800d8dc1c3f30cf90985184ccc0d3e5c26333e95c6"
EXPECTED_MANIFEST_SHA256 = "25f150360f173970895dafd31e390f94b4b01ca174e6ad09b9e288c94b1e48cc"
EXPECTED_BASE_RECORDS_SHA256 = "ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f"
EXPECTED_BASE_MANIFEST_SHA256 = "b8c290dd64f210dd469b83cbf41ad321ae4d7c5c40303c70e8cda0e4335cd0c3"
EXPECTED_TRAIN_SHA256 = "1bb619fb809d71c68813eb07aa162b2b2bfa1b8f2de7cef1a67b60054d4ae816"
EXPECTED_PREPARE_DATA_SHA256 = "3323b74d93ac7895c9c1233c018d1c58341743eb1bb23d1df1835eaf5573c101"
EXPECTED_SAMPLER_SHA256 = "081ad3a97090651bba4b146e77fcd47807e0942d3ab4f7da450274a1e594b8d3"
EXPECTED_FAST_EVAL_SHA256 = "444e6d1fca8bf7f1de00dca951d64961cec5b3b64e3461034174140742b4bb36"
EXPECTED_FAST_METADATA_SHA256 = "0163435947c8d8efa4c8f1455fba8ff22080b4a2969e2b98891f9dc9cfba356d"
EXPECTED_FAST_TEST_SHA256 = "c41ea06b2b941ca289f4033bf431958783f494cd0f6d8d88a6aa51991b76b5d2"
EXPECTED_JOINT_PARITY_SHA256 = "1c739713672ed7530f0b1921cf58d85192c36647faa3b3225718d7d808b974f3"
OUTPUT_KEYS = ("G", "Q_reference", "E_reference", "closed_loop", "flow_complete",
               "Q_swap", "Q_no_crystal")
FACT_FIELDS = ("sg", "formula", "atoms", "orbits", "a", "b", "c")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _pin_path(pin, key, expected):
    value = Path(pin[key])
    path = (PROJECT / value).resolve() if not value.is_absolute() else value.resolve()
    require(path == expected.resolve(), f"pin {key} must name {expected}")
    return path


def validate_training_receipts(pin, status, checkpoint_receipt, summary):
    """Reject every nonterminal or differently configured training artifact."""
    require(pin.get("status") == "verified", "evaluation input pin is not verified")
    require(pin.get("updates") == 2048, "evaluation input pin is not for update 2048")
    require(status.get("status") == "trained" and status.get("completed_updates") == 2048
            and status.get("updates") == 2048, "training did not finish exactly 2048 updates")
    require(checkpoint_receipt.get("status") == "verified"
            and checkpoint_receipt.get("checkpoint_update") == 2048,
            "final checkpoint receipt is not a verified 2048-update reload")
    require(summary.get("status") == "training_only_checkpoint_verified"
            and summary.get("stage") == "scaled32_training_only"
            and summary.get("updates") == 2048,
            "training summary is not the terminal training-only receipt")
    require(status.get("reader_mode") == checkpoint_receipt.get("reader_mode") == "grounded"
            and summary.get("reader_mode") == "grounded", "expected grounded reader training")
    require(status.get("seed") == 17 and summary.get("seed") == 17,
            "training seed differs from the frozen seed-17 recipe")
    require(summary.get("model") == "Qwen/Qwen2.5-1.5B-Instruct"
            and summary.get("revision") == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
            and summary.get("clean_crystal_ar_weight") == 1.0,
            "training summary model configuration differs from the frozen recipe")
    for receipt in (status, checkpoint_receipt):
        require(receipt.get("mode") == "hybrid"
                and receipt.get("arm") == "joint_ar_diffusion_b32"
                and receipt.get("prediction_offset") == 0
                and receipt.get("block_size") == 32
                and receipt.get("clean_crystal_ar_weight") == 1.0,
                "training configuration differs from frozen hybrid B32 seed-17 recipe")
    require(status.get("reader_schedule") == "alternating_q1_q2",
            "training reader schedule differs from the frozen recipe")
    fingerprint = pin.get("final_fingerprint")
    require(bool(fingerprint), "pin lacks the final model fingerprint")
    require(status.get("final_fingerprint") == checkpoint_receipt.get("final_fingerprint")
            == checkpoint_receipt.get("reload_fingerprint") == summary.get("final_fingerprint")
            == fingerprint, "final model fingerprint does not match the pin and receipts")
    require(checkpoint_receipt.get("reload_verified") is True
            and checkpoint_receipt.get("strict_state_load") is True
            and status.get("checkpoint_reload_verified") is True
            and status.get("checkpoint_reload_fingerprint") == fingerprint
            and summary.get("checkpoint_reload_verified") is True,
            "strict final checkpoint reload was not verified")
    checkpoint_sha = pin.get("checkpoint_sha256")
    require(bool(checkpoint_sha)
            and status.get("checkpoint_sha256") == checkpoint_receipt.get("checkpoint_sha256")
            == checkpoint_sha, "checkpoint SHA-256 does not match the terminal receipts")
    require(status.get("code_hashes") == checkpoint_receipt.get("source_hashes"),
            "status and checkpoint receipt have different source pins")
    hashes = status.get("code_hashes", {})
    require(set(hashes) == {"parent", "wrapper", "data"},
            "training receipt must pin parent, wrapper, and data sources")
    require(hashes["data"] == {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "records_sha256": EXPECTED_RECORDS_SHA256,
    }, "training data pins differ from the frozen scaled32 cohort")
    require(status.get("dataset_manifest_sha256") == EXPECTED_MANIFEST_SHA256
            and status.get("dataset_records_sha256") == EXPECTED_RECORDS_SHA256,
            "training receipt data hashes differ from the frozen scaled32 cohort")
    require(summary.get("evaluation") == {
        "train32": "not_run", "heldout16": "not_run", "AR_crystal_inference": "not_run",
    }, "training summary already contains evaluation results")
    return fingerprint, checkpoint_sha, hashes


def select_cohorts(manifest, rows):
    cohort = manifest["cohort"]
    train_ids, heldout_ids = cohort["train_ids"], cohort["heldout_ids"]
    require(len(rows) == 48 and len(train_ids) == 32 and len(heldout_ids) == 16,
            "scaled32 cohort must contain 32 train and 16 heldout records")
    expected_order = train_ids[:8] + heldout_ids + train_ids[8:]
    require([row["id"] for row in rows] == expected_order,
            "records do not preserve the frozen original8/heldout16/appended24 order")
    require([row["split"] for row in rows] == ["train"] * 8 + ["heldout"] * 16 + ["train"] * 24,
            "records do not preserve the frozen split order")
    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    require([row["id"] for row in train_rows] == train_ids
            and [row["id"] for row in heldout_rows] == heldout_ids,
            "record IDs differ from the frozen split manifest")
    train_cifs = {row["source_hashes"]["source_cif_sha256"] for row in train_rows}
    heldout_cifs = {row["source_hashes"]["source_cif_sha256"] for row in heldout_rows}
    require(train_cifs.isdisjoint(heldout_cifs), "train and heldout source CIFs overlap")
    return train_rows[:8], heldout_rows, train_rows[8:]


def _validate_actual_exposure(status, train_ids):
    exposure = status["actual_exposure_counts"]
    expected = {"G": 128, "E": 64, "Q1": 32, "Q2": 32}
    require(set(exposure["per_material"]) == set(train_ids),
            "actual training exposure does not cover exactly the 32 train cases")
    for case_id in train_ids:
        require(exposure["per_material"][case_id]["tasks"] == expected,
                f"actual task counts differ for {case_id}")
    require(exposure["total"] == {"G": 4096, "E": 2048, "Q1": 1024, "Q2": 1024},
            "actual aggregate task counts differ from the frozen 2048-update recipe")


def _load_inputs(pin_path):
    pin = read_json(pin_path)
    paths = {
        "status_path": ARM_DIR / "status.json",
        "checkpoint_receipt_path": ARM_DIR / "checkpoint_receipt.json",
        "summary_path": TRAIN_DIR / "summary.json",
        "checkpoint_path": ARM_DIR / "checkpoint.pt",
    }
    resolved = {key: _pin_path(pin, key, path) for key, path in paths.items()}
    for key in ("status_path", "checkpoint_receipt_path", "summary_path"):
        expected_hash = pin.get(key.replace("_path", "_sha256"))
        require(bool(expected_hash) and sha_file(resolved[key]) == expected_hash,
                f"pinned {key} content changed")
    require(resolved["checkpoint_path"].is_file(), "pinned final checkpoint is missing")
    status = read_json(resolved["status_path"])
    receipt = read_json(resolved["checkpoint_receipt_path"])
    summary = read_json(resolved["summary_path"])
    fingerprint, checkpoint_sha, hashes = validate_training_receipts(pin, status, receipt, summary)
    return {
        "pin": pin, "pin_path": Path(pin_path).resolve(), "paths": resolved,
        "status": status, "receipt": receipt, "training_summary": summary,
        "fingerprint": fingerprint, "checkpoint_sha256": checkpoint_sha,
        "code_hashes": hashes,
    }


def _load_runtime():
    for path in (DIAGNOSTIC_ROOT, BLOCK_ROOT, JOINT_ROOT, HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import model as model_module
    import prepare
    import run as runner
    import prepare_data as scaled_data
    import joint_confidence_eval as joint_eval

    joint_eval._load_runtime()
    return {
        "torch": joint_eval.torch, "runner": runner, "prepare": prepare,
        "scaled_data": scaled_data,
        "model_module": model_module, "joint_eval": joint_eval,
        "fast_eval": joint_eval.fast_eval, "commit_probe": joint_eval.commit_probe,
        "fast_metadata": joint_eval.fast_metadata,
    }


def _verify_sources(inputs, runtime):
    code_hashes = inputs["code_hashes"]
    actual_parent = {name: sha_file(JOINT_ROOT / name) for name in code_hashes["parent"]}
    actual_wrapper = {name: sha_file(HERE / name) for name in code_hashes["wrapper"]}
    data_hashes = {
        "manifest_sha256": sha_file(DATA_DIR / "manifest.json"),
        "records_sha256": sha_file(DATA_DIR / "records.jsonl"),
    }
    require(actual_parent == code_hashes["parent"], "parent runner source changed after training")
    require(actual_wrapper == code_hashes["wrapper"], "training wrapper or data helper changed after training")
    require(data_hashes == code_hashes["data"], "scaled32 training data changed after training")
    helper = {
        "commit_probe.py": sha_file(runtime["commit_probe"].__file__),
        "full_commit_fast_eval.py": sha_file(runtime["fast_eval"].__file__),
        "fast_inference_metadata.py": sha_file(runtime["fast_metadata"].__file__),
        "test_fast_inference_metadata.py": sha_file(runtime["fast_eval"].metadata_check.__file__),
        "joint_confidence_eval.py": sha_file(runtime["joint_eval"].__file__),
    }
    require(helper["commit_probe.py"] == EXPECTED_SAMPLER_SHA256
            and helper["full_commit_fast_eval.py"] == EXPECTED_FAST_EVAL_SHA256
            and helper["fast_inference_metadata.py"] == EXPECTED_FAST_METADATA_SHA256
            and helper["test_fast_inference_metadata.py"] == EXPECTED_FAST_TEST_SHA256
            and helper["joint_confidence_eval.py"] == EXPECTED_JOINT_PARITY_SHA256,
            "reused sampler or optimization helper differs from its reviewed source")
    require(actual_wrapper.get("train.py") == EXPECTED_TRAIN_SHA256
            and actual_wrapper.get("prepare_data.py") == EXPECTED_PREPARE_DATA_SHA256,
            "scaled32 trainer or data helper differs from its frozen source pin")
    return {"parent": actual_parent, "wrapper": actual_wrapper, "data": data_hashes,
            "evaluation_helpers": helper}


def _read_cohort(inputs, runtime):
    manifest = read_json(DATA_DIR / "manifest.json")
    records_path = DATA_DIR / "records.jsonl"
    require(sha_file(records_path) == EXPECTED_RECORDS_SHA256
            and sha_file(DATA_DIR / "manifest.json") == EXPECTED_MANIFEST_SHA256,
            "scaled32 data differ from the frozen record and manifest pins")
    require(manifest["status"] == "data_prepared_not_trained"
            and manifest["representation"] == "relational_cnl_v1"
            and manifest["records"]["count"] == 48,
            "data manifest is not the reviewed relational CNL scaled32 cohort")
    require(manifest["source_pins"]["original24_manifest_sha256"] == EXPECTED_BASE_MANIFEST_SHA256
            and manifest["prefix_preservation"]["source_records_sha256"] == EXPECTED_BASE_RECORDS_SHA256,
            "scaled32 manifest lost the verified original24 source boundary")
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    seen8, heldout16, remaining24 = select_cohorts(manifest, rows)
    _validate_actual_exposure(inputs["status"], manifest["cohort"]["train_ids"])
    return {
        "manifest": manifest, "rows": rows, "seen8": seen8,
        "heldout16": heldout16, "remaining24": remaining24,
    }


def _fresh_codec_snapshot(inputs, runtime):
    manifest_path = runtime["scaled_data"].BASE_MANIFEST_PATH
    manifest = read_json(manifest_path)
    manifest_hash = sha_file(manifest_path)
    require(manifest_hash == EXPECTED_BASE_MANIFEST_SHA256
            and manifest["records_sha256"] == EXPECTED_BASE_RECORDS_SHA256,
            "original 24-case codec manifest differs from its frozen pin")
    require(sha_file(runtime["scaled_data"].BASE_RECORDS_PATH) == EXPECTED_BASE_RECORDS_SHA256,
            "original 24-case records differ from their frozen pin")
    return {
        "base_manifest_sha256": manifest_hash,
        "codec_pin": runtime["runner"].verify_codec_pin(manifest),
    }


def _fresh_tokenizer_hashes(inputs, runtime):
    tokenizer_path = runtime["prepare"].TOKENIZER_PATH
    hashes = {
        path.name: sha_file(path) for path in sorted(Path(tokenizer_path).iterdir()) if path.is_file()
    }
    base_manifest = read_json(runtime["scaled_data"].BASE_MANIFEST_PATH)
    require(hashes == base_manifest["provenance"]["tokenizer_files"]
            and hashes == inputs["status"]["source_pin"]["tokenizer_files"],
            "tokenizer files differ from the frozen data and training receipts")
    return hashes


def _snapshot(inputs, runtime):
    source_hashes = _verify_sources(inputs, runtime)
    data_check = runtime["scaled_data"].check_dataset()
    require(data_check == {"status": "check_passed_data_only", "rows": 48, "train": 32,
                           "heldout": 16, "training_run": "not_run"},
            "scaled32 frozen source and split check failed")
    codec = _fresh_codec_snapshot(inputs, runtime)
    tokenizer_hashes = _fresh_tokenizer_hashes(inputs, runtime)
    return {
        "pin_sha256": sha_file(inputs["pin_path"]),
        "training_status_sha256": sha_file(inputs["paths"]["status_path"]),
        "checkpoint_receipt_sha256": sha_file(inputs["paths"]["checkpoint_receipt_path"]),
        "training_summary_sha256": sha_file(inputs["paths"]["summary_path"]),
        "checkpoint_sha256": sha_file(inputs["paths"]["checkpoint_path"]),
        "final_fingerprint": inputs["fingerprint"],
        "code_hashes": source_hashes,
        "evaluation_sources": {
            name: sha_file(HERE / name) for name in ("evaluate.py", "test_evaluate.py", "evaluate.slurm")
        },
        "data_hashes": {
            "manifest_sha256": sha_file(DATA_DIR / "manifest.json"),
            "records_sha256": sha_file(DATA_DIR / "records.jsonl"),
        },
        "tokenizer_hashes": tokenizer_hashes,
        "codec_pin": codec,
    }


def assert_snapshot_unchanged(before, after):
    require(after == before,
            "training, source, tokenizer, data, codec, or checkpoint changed during evaluation")


def _load_final_model(inputs, runtime, tok):
    torch, runner = runtime["torch"], runtime["runner"]
    model = runner.Hybrid.pretrained(
        tok, prediction_offset=0, block_size=32, clean_crystal_ar_weight=1.0
    ).cuda().eval()
    saved = torch.load(inputs["paths"]["checkpoint_path"], map_location="cpu", weights_only=True)
    require(saved["update"] == 2048, "checkpoint payload is not the final update")
    config = model.restore_checkpoint(
        saved, expected_mode="hybrid", expected_offset=0,
        expected_block_size=32, expected_clean_crystal_ar_weight=1.0,
    )
    require(config["completed_updates"] == 2048
            and config["final_fingerprint"] == inputs["fingerprint"],
            "checkpoint payload is not the pinned final training state")
    del saved
    fingerprint = runner.fingerprint(model)
    require(fingerprint == inputs["fingerprint"], "strictly reloaded model fingerprint differs from pin")
    require(model.block_size == 32 and model.prediction_offset == 0
            and model.clean_crystal_ar_weight == 1.0,
            "loaded model differs from the pinned hybrid B32 configuration")
    return model, fingerprint


def _task_metric_rates(seen, heldout):
    fields = {}
    for key in OUTPUT_KEYS:
        fields[key] = {
            "seen8": {"count": seen["counts"][key], "denominator": seen["n"],
                      "rate": seen["rates"][key]},
            "heldout16": ({"count": heldout["counts"][key], "denominator": heldout["n"],
                           "rate": heldout["rates"][key]} if heldout else {
                "status": "NOT_EVALUATED", "count": None, "denominator": 0, "rate": None,
            }),
        }
    return fields


def _attempt_fields(records, attempt_key, grade_key, field_names):
    attempts = [record for record in records if attempt_key in record]
    denominator = len(attempts)
    values = {}
    for field in field_names:
        count = sum(bool(record.get(grade_key, {}).get("fields", {}).get(field, False))
                    for record in attempts)
        values[field] = {
            "count": count, "denominator": denominator,
            "rate": count / denominator if denominator else None,
        }
    return {
        "attempted_cases": denominator,
        "cohort_cases": len(records),
        "successful_responses": sum(record[attempt_key].get("ok") is True for record in attempts),
        "field_metrics": values,
    }


def _read_field_metrics(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    return {
        "Q_reference": _attempt_fields(records, "reference_read", "reference_read_grade", FACT_FIELDS),
        "Q_swap": _attempt_fields(records, "swap_read", "swap_read_grade", FACT_FIELDS),
        "Q1_actual": _attempt_fields(records, "actual_read", "actual_read_grade", FACT_FIELDS),
        "Q2_actual": _attempt_fields(records, "actual_second_read", "actual_second_read_grade", FACT_FIELDS),
    }


def _edit_field_metrics(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    edit_fields = ("lengths_exact", "other_plan_fields_exact", "whole_geometry_scale")
    return {
        "E_reference": _attempt_fields(records, "reference_edit", "reference_edit_grade", edit_fields),
        "E_actual": _attempt_fields(records, "actual_edit", "actual_edit_grade", edit_fields),
    }


def _no_crystal_diagnostic(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    attempts = [record for record in records if "no_crystal_read" in record]
    exact = sum(record.get("no_crystal_fact_match", {}).get("success") is True
                for record in attempts)
    return {
        "semantics": "diagnostic exact-fact match without a structure; not a grounding or success score",
        "attempted_cases": len(attempts), "cohort_cases": len(records),
        "exact_fact_matches": exact,
        "rate_per_attempted_case": exact / len(attempts) if attempts else None,
    }


def run(pin_path=DEFAULT_PIN, out=DEFAULT_OUT):
    pin_path, out = Path(pin_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite final evaluation: {out}")
    inputs = _load_inputs(pin_path)
    runtime = _load_runtime()
    device = runtime["joint_eval"].header_probe.require_gpu_allocation()
    actual_checkpoint_sha = sha_file(inputs["paths"]["checkpoint_path"])
    require(actual_checkpoint_sha == inputs["checkpoint_sha256"],
            "checkpoint hash differs from the explicit final evaluation pin")
    cohort = _read_cohort(inputs, runtime)
    before = _snapshot(inputs, runtime)
    tok = runtime["prepare"].tokenizer()
    model, fingerprint_before = _load_final_model(inputs, runtime, tok)

    module = runtime["model_module"]
    original_functions = (module.attention, module.crystal_prefix_times)
    original_sampler_alias = runtime["commit_probe"].crystal_prefix_times
    metadata = runtime["fast_eval"].install_and_check_fast_metadata(model, "cuda")
    runtime["commit_probe"].crystal_prefix_times = runtime["fast_metadata"].fast_crystal_prefix_times
    parity = runtime["joint_eval"]._parity_gate(
        model, tok, cohort["seen8"], module, original_functions, original_sampler_alias
    )
    require(parity["exact_cases"] == parity["expected_cases"] == 8,
            "old/fast 128-token seen8 parity did not cover all eight cases")
    require(parity["prefix_tokens"] == 128 and parity["method"] == "legacy_random_greedy_tokens"
            and all(case["exact_tokens_text_time_release_header"] for case in parity["cases"]),
            "old/fast seen8 parity did not preserve every 128-token result exactly")
    require(metadata["checked_tensor_comparisons"] == 750,
            "CUDA attention and prefix-time equivalence did not check all 750 tensors")
    require(runtime["runner"].fingerprint(model) == fingerprint_before
            and model.block_size == 32 and model.prediction_offset == 0
            and model.clean_crystal_ar_weight == 1.0,
            "optimization gate changed the final model before evaluation")

    out.mkdir(parents=True, exist_ok=False)
    runtime["runner"].write_json(out / "optimization_gate.json", {
        "status": "passed", "checkpoint_sha256": actual_checkpoint_sha,
        "fingerprint": fingerprint_before, "cuda_metadata_equivalence": metadata,
        "paired_seen8_128_token_parity": parity,
        "source_hashes_before": before,
    })
    runtime["fast_eval"].bind_policy(model, "confidence_one")
    seen = runtime["runner"].evaluate(
        model, tok, "joint_ar_diffusion_b32", cohort["seen8"], out / "seen8"
    )
    require(runtime["runner"].fingerprint(model) == fingerprint_before,
            "model weights changed during seen8 evaluation")
    heldout = None
    if runtime["runner"].seen8_generation_gate_passed(seen):
        heldout = runtime["runner"].evaluate(
            model, tok, "joint_ar_diffusion_b32", cohort["heldout16"], out / "heldout16"
        )
        require(runtime["runner"].fingerprint(model) == fingerprint_before,
                "model weights changed during heldout evaluation")
    seen_read_fields = _read_field_metrics(out / "seen8/evaluation.jsonl")
    seen_edit_fields = _edit_field_metrics(out / "seen8/evaluation.jsonl")
    seen_no_crystal = _no_crystal_diagnostic(out / "seen8/evaluation.jsonl")
    heldout_read_fields = (_read_field_metrics(out / "heldout16/evaluation.jsonl") if heldout else None)
    heldout_edit_fields = (_edit_field_metrics(out / "heldout16/evaluation.jsonl") if heldout else None)
    heldout_no_crystal = (_no_crystal_diagnostic(out / "heldout16/evaluation.jsonl") if heldout else None)
    after = _snapshot(inputs, runtime)
    assert_snapshot_unchanged(before, after)
    require(runtime["runner"].fingerprint(model) == fingerprint_before == inputs["fingerprint"],
            "model fingerprint changed during final evaluation")

    output = {
        "status": "complete", "stage": "scaled32_final_2048_evaluation",
        "training_reader_mode": "grounded", "training_updates": 2048,
        "training_checkpoint_sha256": actual_checkpoint_sha,
        "fingerprint_before": fingerprint_before, "fingerprint_after": runtime["runner"].fingerprint(model),
        "model": {"name": runtime["runner"].MODEL, "revision": runtime["runner"].REVISION,
                  "mode": "hybrid", "block_size": 32, "prediction_offset": 0,
                  "clean_crystal_ar_weight": 1.0},
        "sampling": {"policy": "confidence_one", "token_selection": "greedy_argmax",
                     "raw_tokens_per_crystal": runtime["prepare"].K,
                     "ar_fallback": False, "grammar_repair": False, "physical_repair": False,
                     "early_eos": False, "oracle_numbers_at_inference": False},
        "seen8": seen,
        "heldout16": (heldout if heldout else {
            "status": "NOT_EVALUATED", "reason": "seen8 G count was zero",
            "n": 0, "expected_n": 16,
        }),
        "task_metric_rates": _task_metric_rates(seen, heldout),
        "read_fact_fields": {
            "seen8": seen_read_fields,
            "heldout16": heldout_read_fields if heldout else {"status": "NOT_EVALUATED"},
        },
        "edit_fields": {
            "seen8": seen_edit_fields,
            "heldout16": heldout_edit_fields if heldout else {"status": "NOT_EVALUATED"},
        },
        "no_crystal_diagnostic": {
            "seen8": seen_no_crystal,
            "heldout16": heldout_no_crystal if heldout else {"status": "NOT_EVALUATED"},
        },
        "cohort": {
            "seen8_case_ids": [row["id"] for row in cohort["seen8"]],
            "heldout16_case_ids": [row["id"] for row in cohort["heldout16"]],
            "trained_case_count": 32, "free_evaluated_train_case_count": 8,
            "trained_but_not_free_evaluated_train_case_count": len(cohort["remaining24"]),
            "trained_but_not_free_evaluated_train_case_ids": [row["id"] for row in cohort["remaining24"]],
            "heldout16_status": "evaluated" if heldout else "NOT_EVALUATED",
        },
        "metric_semantics": {
            "G": "free generation strict audit against the requested source case",
            "Q_reference": "source-crystal read control; not an actual-history grounding score",
            "E_reference": "source-crystal edit control; separate from generated-history editing",
            "closed_loop": "actual generated G, Q1, edit, and Q2 sequence all pass their strict checks",
            "flow_complete": "actual Q2 response was produced after the generated sequence",
            "Q_swap": "read control on the original runner's within-split replacement crystal",
            "Q_no_crystal": "exact-fact match without a crystal; diagnostic only, not grounding or success",
        },
        "inference_boundary": "G uses the original condition prompt; actual Q1, E, and Q2 append model outputs in sequence. Q_reference and E_reference are separate source-crystal controls.",
        "optimizer_updates": 0,
        "optimization_verification": metadata,
        "paired_seen8_128_token_parity": parity,
        "evaluation_rng_seeds": {
            row["id"]: int(hashlib.sha256((row["id"] + "20261004").encode()).hexdigest()[:8], 16)
            for row in cohort["seen8"] + (cohort["heldout16"] if heldout else [])
        },
        "runtime": {
            "slurm_job": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
            "python_version": platform.python_version(), "device": device,
            "packages": {name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "numpy", "pyxtal", "pymatgen", "spglib"
            )},
        },
        "hashes_before": before, "hashes_after": after,
    }
    runtime["runner"].write_json(out / "summary.json", output)
    del model
    runtime["torch"].cuda.empty_cache()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "complete", "output": str(run(args.pin, args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
