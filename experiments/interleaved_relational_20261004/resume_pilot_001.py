"""Fixed continuation: replay AR heldout and train only the H-shift arm."""
import argparse
import ast
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

from model import Hybrid
import prepare
import run as runner
from prepare import DATA_DIR, MODEL, REVISION, ROOT

PROJECT = ROOT.parents[1]
PILOT1 = ROOT / "pilot_001"
PILOT2 = ROOT / "pilot_002"
EXCEPTION_PATH = ROOT / "execution_exception_pilot_001.json"
AR_INITIAL = "6bffa202d8485a6ee8bce3553044c9657948e25be702a45a9efb860ced9d1429"
AR_FINAL = "e466cb281d88e6356f3ed73dc2915a3bf947024abecaa534f25d22a6a0dca5e8"
AR_CHECKPOINT_SHA = "9d73aeebc4966efafb6fd2bb03bb448c400d0e20878c1258ec8c8c1321f123e5"
AR_RECORDS_SHA = "ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f"
AR_OLD_MANIFEST_SHA = "e4b67ae8a4e8b831b8c529992b9ea2a83b44603689cdc0b460e1d9d7f9ff5513"
OUTPUT_FIELDS = (
    "id", "split", "direction", "generation", "reference_read", "reference_edit",
    "actual_read", "actual_edit", "actual_second_read", "swap_id", "swap_read",
    "no_crystal_read",
)


def write_json(path, value):
    runner.write_json(path, value)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def assert_counts(records, summary):
    counts = {
        key: sum(bool(record.get(key, False)) for record in records)
        for key in runner.OUTPUT_KEYS
    }
    assert len(records) == summary["n"]
    assert counts == summary["counts"]


def scientific_output(record):
    return {key: record.get(key) for key in OUTPUT_FIELDS}


def function_ast_sha(source, name):
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )
    return runner.sha(ast.dump(function, include_attributes=False).encode())


def assert_current_device():
    assert torch.cuda.is_available(), "Pilot continuation requires its Slurm CUDA allocation"
    assert os.environ.get("SLURM_JOB_ID"), "Pilot continuation must run inside Slurm"
    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    assert "A100" in name and memory_gib >= 79, f"requires A100 80GB, got {name} ({memory_gib:.2f} GiB)"
    return {"device": name, "total_memory_GiB": memory_gib, "slurm_job": os.environ["SLURM_JOB_ID"]}


def verify_predecessor(require_checkpoint):
    assert not PILOT2.exists(), f"Refusing to overwrite fixed continuation output: {PILOT2}"
    manifest, rows, train_rows, heldout = runner.read_data(DATA_DIR)
    assert len(rows) == 24 and len(train_rows) == 8 and len(heldout) == 16
    data_hashes = runner.data_source_hashes(DATA_DIR)
    assert data_hashes["records_sha256"] == AR_RECORDS_SHA
    source_pin = runner.verify_codec_pin(manifest)
    exposure = runner.exposure_receipt(train_rows)

    status_path = PILOT1 / "ar/status.json"
    checkpoint_receipt_path = PILOT1 / "ar/checkpoint_receipt.json"
    original_run_path = ROOT / "q2_coverage_preparation/source_snapshot/run.py"
    status = read_json(status_path)
    checkpoint_receipt = read_json(checkpoint_receipt_path)
    failure = read_json(EXCEPTION_PATH)
    old_gpu_preflight = read_json(PILOT1 / "gpu_preflight.json")

    assert status["mode"] == "ar" and status["arm"] == "ar"
    assert status["status"] in ("trained", "evaluated")
    assert status["completed_updates"] == 512 and status["seed"] == 17
    assert status["prediction_offset"] == 0
    assert status["model"] == MODEL and status["model_revision"] == REVISION
    assert status["initial_fingerprint"] == AR_INITIAL
    assert status["final_fingerprint"] == AR_FINAL
    assert status["checkpoint_sha256"] == AR_CHECKPOINT_SHA
    assert status["checkpoint_reload_verified"] is True
    assert status["checkpoint_reload_mode"] == "ar" and status["checkpoint_reload_offset"] == 0
    assert status["dataset_records_sha256"] == AR_RECORDS_SHA
    assert status["dataset_manifest_sha256"] == AR_OLD_MANIFEST_SHA
    assert status["actual_exposure_sha256"] == runner.sha(
        json.dumps(exposure["events"], sort_keys=True, separators=(",", ":")).encode()
    )
    assert status["actual_exposure_counts"]["per_material"] == exposure["per_material"]
    assert status["actual_exposure_counts"]["total"] == exposure["total"]
    assert status["code_hashes"]["model.py"] == runner.sha((ROOT / "model.py").read_bytes())
    assert status["code_hashes"]["prepare.py"] == runner.sha((ROOT / "prepare.py").read_bytes())
    assert status["code_hashes"]["pilot_001.slurm"] == runner.sha((ROOT / "pilot_001.slurm").read_bytes())
    original_run_source = original_run_path.read_text()
    current_run_source = (ROOT / "run.py").read_text()
    original_run_sha = runner.sha(original_run_source.encode())
    current_run_sha = runner.sha(current_run_source.encode())
    assert original_run_sha == status["code_hashes"]["run.py"]
    assert runner.DEFAULT_READER_SCHEDULE == "q1_only"
    evaluate_ast_unchanged = (
        function_ast_sha(original_run_source, "evaluate")
        == function_ast_sha(current_run_source, "evaluate")
    )
    assert evaluate_ast_unchanged, "pilot_001 evaluator changed during the Q2 schedule extension"

    assert checkpoint_receipt["status"] == "verified"
    assert checkpoint_receipt["checkpoint_update"] == 512
    assert checkpoint_receipt["checkpoint_sha256"] == AR_CHECKPOINT_SHA
    assert checkpoint_receipt["mode"] == "ar" and checkpoint_receipt["prediction_offset"] == 0
    assert checkpoint_receipt["final_fingerprint"] == AR_FINAL
    assert checkpoint_receipt["reload_verified"] is True
    assert checkpoint_receipt["reload_fingerprint"] == AR_FINAL
    assert checkpoint_receipt["strict_state_load"] is True

    assert failure["terminal_state"] == "FAILED"
    assert failure["slurm_job_id"] == "604554"
    assert failure["failure"]["phase"] == "ar/heldout16"
    assert failure["failure"]["record_id"] == "mp-1079976"
    assert failure["failure"]["failed_row_written_to_raw"] is False
    assert failure["partial_evaluation"]["hybrid_shift_started"] is False
    assert failure["partial_evaluation"]["complete_pilot_result"] is False
    assert failure["frozen_data"]["records_sha256"] == AR_RECORDS_SHA
    assert failure["frozen_data"]["manifest_sha256"] == AR_OLD_MANIFEST_SHA
    refresh = manifest.get("post_generation_score_refresh", {})
    assert refresh.get("records_sha256_unchanged") == AR_RECORDS_SHA
    current_score_sha = runner.sha((ROOT / "score.py").read_bytes())
    assert manifest["provenance"]["files"]["adapter_score"] == current_score_sha
    old_manifest_path = ROOT / "recovery_predecessor_001/source_snapshot/manifest.json"
    old_score_path = ROOT / "recovery_predecessor_001/source_snapshot/score.py"
    old_manifest = read_json(old_manifest_path)
    old_manifest_sha = runner.sha_file(old_manifest_path)
    old_score_sha = old_manifest["post_generation_score_refresh"]["current_score_sha256"]
    assert old_manifest_sha == AR_OLD_MANIFEST_SHA
    assert runner.sha_file(old_score_path) == old_score_sha
    assert status["code_hashes"]["score.py"] == old_score_sha
    assert refresh.get("previous_score_sha256") == old_score_sha
    assert refresh.get("current_score_sha256") == current_score_sha
    migrated_manifest = deepcopy(old_manifest)
    for key in ("previous_score_sha256", "current_score_sha256", "reason"):
        migrated_manifest["post_generation_score_refresh"][key] = refresh[key]
    migrated_manifest["provenance"]["files"]["adapter_score"] = current_score_sha
    assert migrated_manifest == manifest, "manifest migration changed fields outside the adapter score pin"

    old_code_hashes = status["code_hashes"]
    old_source_map = runner.runner_source_hashes()
    for name, digest in old_code_hashes.items():
        if name in ("score.py", "test_checks.py"):
            continue
        if name == "run.py":
            assert original_run_sha == digest
            continue
        assert old_source_map.get(name) == digest, f"unexpected source migration in {name}"

    seen_raw_path = PILOT1 / "ar/seen8/evaluation.jsonl"
    seen_summary_path = PILOT1 / "ar/seen8/evaluation_summary.json"
    held_raw_path = PILOT1 / "ar/heldout16/evaluation.jsonl"
    seen_records = read_jsonl(seen_raw_path)
    seen_summary = read_json(seen_summary_path)
    old_heldout_records = read_jsonl(held_raw_path)
    assert [record["id"] for record in seen_records] == [row["id"] for row in train_rows]
    assert_counts(seen_records, seen_summary)
    assert len(old_heldout_records) == 2
    assert [record["id"] for record in old_heldout_records] == [row["id"] for row in heldout[:2]]
    assert heldout[2]["id"] == failure["failure"]["record_id"]
    assert all(record["id"] != heldout[2]["id"] for record in old_heldout_records)
    assert runner.sha(seen_raw_path.read_bytes()) == failure["local_raw_sha256"]["ar/seen8/evaluation.jsonl"]
    assert runner.sha(seen_summary_path.read_bytes()) == failure["local_raw_sha256"]["ar/seen8/evaluation_summary.json"]
    assert runner.sha(held_raw_path.read_bytes()) == failure["local_raw_sha256"]["ar/heldout16/evaluation.jsonl"]
    assert seen_summary["counts"] == failure["partial_evaluation"]["seen8"]["counts"]

    assert old_gpu_preflight["status"] == "passed"
    assert "A100" in old_gpu_preflight["device"] and old_gpu_preflight["device_total_GiB"] >= 79
    assert old_gpu_preflight["shared_initial_fingerprint"] == AR_INITIAL
    assert old_gpu_preflight["arms"]["ar"]["peak_reserved_GiB"] <= 70
    assert old_gpu_preflight["arms"]["hybrid_shift"]["peak_reserved_GiB"] <= 70
    assert old_gpu_preflight["arms"]["hybrid_shift"]["prediction_offset"] == 1

    checkpoint_path = PILOT1 / "ar/checkpoint.pt"
    checkpoint_present = checkpoint_path.is_file()
    checkpoint_actual_sha = None
    if require_checkpoint:
        assert checkpoint_present, f"missing fixed AR checkpoint: {checkpoint_path}"
        checkpoint_actual_sha = runner.sha_file(checkpoint_path)
        assert checkpoint_actual_sha == AR_CHECKPOINT_SHA
    return {
        "manifest": manifest,
        "rows": rows,
        "train_rows": train_rows,
        "heldout": heldout,
        "data_hashes": data_hashes,
        "source_pin": source_pin,
        "exposure": exposure,
        "old_status": status,
        "old_checkpoint_receipt": checkpoint_receipt,
        "old_failure": failure,
        "old_preflight": old_gpu_preflight,
        "old_seen_records": seen_records,
        "old_seen_summary": seen_summary,
        "old_heldout_records": old_heldout_records,
        "old_seen_raw_sha256": runner.sha(seen_raw_path.read_bytes()),
        "old_seen_summary_sha256": runner.sha(seen_summary_path.read_bytes()),
        "old_heldout_raw_sha256": runner.sha(held_raw_path.read_bytes()),
        "old_score_sha256": old_code_hashes["score.py"],
        "current_score_sha256": current_score_sha,
        "run_py_schedule_migration": {
            "predecessor_sha256": original_run_sha,
            "current_sha256": current_run_sha,
            "default_reader_schedule": runner.DEFAULT_READER_SCHEDULE,
            "q1_exposure_hash_matches_predecessor": status["actual_exposure_sha256"],
            "evaluate_ast_unchanged": evaluate_ast_unchanged,
        },
        "checkpoint_path": checkpoint_path,
        "checkpoint_present_locally": checkpoint_present,
        "checkpoint_sha256_verified": checkpoint_actual_sha,
    }


def _source_guard(source_hashes):
    runner.assert_runner_sources(source_hashes["runner"])
    assert runner.sha_file(ROOT / "pilot_002.slurm") == source_hashes["pilot_002.slurm"]


def run_fixed():
    source_hashes = {
        "runner": runner.runner_source_hashes(),
        "pilot_002.slurm": runner.sha_file(ROOT / "pilot_002.slurm"),
    }
    device = assert_current_device()
    fixed = verify_predecessor(require_checkpoint=True)
    _source_guard(source_hashes)

    tok = prepare.tokenizer()
    ar_model = Hybrid.pretrained(tok)
    checkpoint = torch.load(fixed["checkpoint_path"], map_location="cpu", weights_only=True)
    assert checkpoint["update"] == 512
    checkpoint_config = ar_model.restore_checkpoint(
        checkpoint, expected_mode="ar", expected_offset=0
    )
    assert checkpoint_config["initial_fingerprint"] == AR_INITIAL
    assert checkpoint_config["final_fingerprint"] == AR_FINAL
    assert runner.fingerprint(ar_model) == AR_FINAL
    del checkpoint
    gc.collect()
    ar_model = ar_model.cuda().eval()
    assert runner.fingerprint(ar_model) == AR_FINAL

    PILOT2.mkdir(parents=True, exist_ok=False)
    ar_seen_dir = PILOT2 / "ar/seen8"
    ar_seen_dir.mkdir(parents=True)
    shutil.copyfile(PILOT1 / "ar/seen8/evaluation.jsonl", ar_seen_dir / "evaluation.jsonl")
    shutil.copyfile(
        PILOT1 / "ar/seen8/evaluation_summary.json",
        ar_seen_dir / "evaluation_summary.json",
    )
    shutil.copyfile(PILOT1 / "gpu_preflight.json", PILOT2 / "inherited_gpu_preflight.json")
    shutil.copyfile(EXCEPTION_PATH, PILOT2 / "pilot_001_execution_exception.json")
    verification = {
        "status": "verified",
        "predecessor": "pilot_001/ar",
        "predecessor_job_id": "604554",
        "predecessor_checkpoint_update": 512,
        "predecessor_initial_fingerprint": AR_INITIAL,
        "predecessor_final_fingerprint": AR_FINAL,
        "checkpoint_sha256": fixed["checkpoint_sha256_verified"],
        "strict_mode": "ar",
        "strict_prediction_offset": 0,
        "checkpoint_reload_verified": True,
        "data_records_sha256": fixed["data_hashes"]["records_sha256"],
        "old_manifest_sha256": AR_OLD_MANIFEST_SHA,
        "current_manifest_sha256": fixed["data_hashes"]["manifest_sha256"],
        "old_score_sha256": fixed["old_score_sha256"],
        "current_score_sha256": fixed["current_score_sha256"],
        "run_py_schedule_migration": fixed["run_py_schedule_migration"],
        "manifest_changed_fields": [
            "post_generation_score_refresh.previous_score_sha256",
            "post_generation_score_refresh.current_score_sha256",
            "post_generation_score_refresh.reason",
            "provenance.files.adapter_score",
        ],
        "score_pin_migrated": fixed["old_score_sha256"] != fixed["current_score_sha256"],
        "score_migration_note": "Only the adapter score pin may migrate; data records and selected split remain byte-identical.",
        "seen8_copied_verbatim": True,
        "seen8_raw_sha256": fixed["old_seen_raw_sha256"],
        "seen8_summary_sha256": fixed["old_seen_summary_sha256"],
        "old_heldout_raw_sha256": fixed["old_heldout_raw_sha256"],
        "old_heldout_raw_rows": 2,
        "failed_row_raw_recovered": False,
        "failed_row_replay_is_new_observation": fixed["heldout"][2]["id"],
        "checkpoint_present_at_start": fixed["checkpoint_present_locally"],
        "new_compute_device": device,
        "inherited_gpu_preflight": fixed["old_preflight"],
        "test256_used": False,
    }
    write_json(PILOT2 / "predecessor_verification.json", verification)
    runner.write_json(PILOT2 / "training_exposure.json", {
        "updates": 512, "effective_batch": 4,
        "task_counts_per_update": {"G": 2, "E": 1, "Q1": 1},
        **fixed["exposure"],
    })
    write_json(PILOT2 / "ar/status.json", {
        "status": "checkpoint_restored",
        "mode": "ar",
        "arm": "ar",
        "seed": 17,
        "updates": 512,
        "prediction_offset": 0,
        "initial_fingerprint": AR_INITIAL,
        "final_fingerprint": runner.fingerprint(ar_model),
        "checkpoint_sha256": fixed["checkpoint_sha256_verified"],
        "checkpoint_reload_verified": True,
        "score_sha256": fixed["current_score_sha256"],
    })

    ar_held_summary = runner.evaluate(
        ar_model, tok, "ar", fixed["heldout"], PILOT2 / "ar/heldout16"
    )
    new_held_records = read_jsonl(PILOT2 / "ar/heldout16/evaluation.jsonl")
    old_by_id = {record["id"]: record for record in fixed["old_heldout_records"]}
    comparisons = []
    for record in new_held_records[:2]:
        predecessor = old_by_id[record["id"]]
        previous_output = scientific_output(predecessor)
        current_output = scientific_output(record)
        comparisons.append({
            "id": record["id"],
            "old_raw_sha256": runner.sha(json.dumps(previous_output, sort_keys=True, separators=(",", ":")).encode()),
            "replay_sha256": runner.sha(json.dumps(current_output, sort_keys=True, separators=(",", ":")).encode()),
            "scientific_model_outputs_equal": previous_output == current_output,
        })
    assert len(new_held_records) == 16
    assert [record["id"] for record in new_held_records] == [row["id"] for row in fixed["heldout"]]
    assert all(item["scientific_model_outputs_equal"] for item in comparisons), (
        "AR replay differs from the two preserved old heldout model outputs"
    )
    assert new_held_records[2]["id"] == fixed["heldout"][2]["id"]
    assert_counts(new_held_records, ar_held_summary)
    write_json(PILOT2 / "ar/heldout16/replay_comparison.json", {
        "old_raw_recovered": False,
        "old_failure_row": fixed["heldout"][2]["id"],
        "row3_and_later_are_new_observations": True,
        "first_two_scientific_model_outputs_match": True,
        "comparisons": comparisons,
        "timing_and_forward_calls_compared": False,
    })
    ar_status = read_json(PILOT2 / "ar/status.json")
    ar_status.update(status="evaluated", seen8=fixed["old_seen_summary"], heldout16=ar_held_summary)
    write_json(PILOT2 / "ar/status.json", ar_status)
    runner.write_json(PILOT2 / "summary.json", {
        "status": "partial",
        "source": "pilot_001 fixed AR checkpoint; AR not retrained",
        "model": MODEL, "revision": REVISION, "seed": 17,
        "records_sha256": fixed["data_hashes"]["records_sha256"],
        "manifest_sha256": fixed["data_hashes"]["manifest_sha256"],
        "checkpoint_sha256": fixed["checkpoint_sha256_verified"],
        "ar": {"seen8": fixed["old_seen_summary"], "heldout16": ar_held_summary},
        "ar_first_two_replay_comparison": comparisons,
        "failed_row_raw_recovered": False,
    })

    _source_guard(source_hashes)
    del ar_model
    gc.collect()
    torch.cuda.empty_cache()

    h_model, tok, h_receipt = runner.train_one(
        "hybrid_shift", fixed["train_rows"], tok, DATA_DIR,
        fixed["manifest"], fixed["data_hashes"], fixed["source_pin"],
        fixed["exposure"], source_hashes["runner"], AR_INITIAL, PILOT2,
    )
    h_seen = runner.evaluate(
        h_model, tok, "hybrid_shift", fixed["train_rows"], PILOT2 / "hybrid_shift/seen8"
    )
    h_held = runner.evaluate(
        h_model, tok, "hybrid_shift", fixed["heldout"], PILOT2 / "hybrid_shift/heldout16"
    )
    _source_guard(source_hashes)
    h_receipt.update(status="evaluated", seen8=h_seen, heldout16=h_held)
    runner.write_json(PILOT2 / "hybrid_shift/status.json", h_receipt)
    results = {
        "ar": {
            "training": "inherited, no retraining",
            "initial_fingerprint": AR_INITIAL,
            "final_fingerprint": AR_FINAL,
            "checkpoint_sha256": fixed["checkpoint_sha256_verified"],
            "seen8": fixed["old_seen_summary"],
            "heldout16": ar_held_summary,
        },
        "hybrid_shift": {
            "initial_fingerprint": h_receipt["initial_fingerprint"],
            "final_fingerprint": h_receipt["final_fingerprint"],
            "checkpoint_sha256": h_receipt["checkpoint_sha256"],
            "seen8": h_seen,
            "heldout16": h_held,
        },
    }
    assert h_receipt["initial_fingerprint"] == AR_INITIAL
    runner.write_json(PILOT2 / "summary.json", {
        "status": "complete",
        "model": MODEL, "revision": REVISION, "seed": 17,
        "updates_per_new_arm": 512,
        "ar_training": "not repeated",
        "initial_fingerprints_equal": True,
        "checkpoint_provenance": verification,
        "results": results,
        "heldout_score_note": "The first two AR raw model outputs match pilot_001; the failed third-row raw was absent, so this full replay is a new observation for that row and the later rows.",
    })
    del h_model
    gc.collect()
    torch.cuda.empty_cache()
    return PILOT2 / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-check", action="store_true")
    args = parser.parse_args(argv)
    if args.offline_check:
        fixed = verify_predecessor(require_checkpoint=False)
        print(json.dumps({
            "status": "predecessor_offline_check_passed",
            "checkpoint_present_locally": fixed["checkpoint_present_locally"],
            "data_records_sha256": fixed["data_hashes"]["records_sha256"],
            "seen8_raw_sha256": fixed["old_seen_raw_sha256"],
            "old_heldout_raw_rows": 2,
            "pilot_002_exists": False,
        }, ensure_ascii=False))
        return 0
    summary = run_fixed()
    print(json.dumps({"status": "complete", "summary": str(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
