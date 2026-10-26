"""Small CPU checks for the terminal receipt gate and frozen evaluation split."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import evaluate


def reader_metadata_from_source(parent_sources):
    source_path = Path(__file__).with_name("train.py")
    tree = ast.parse(source_path.read_text())
    metadata_names = {"READER_RECIPE", "READER_DIRECTION_NOTE"}
    constants = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in metadata_names
                for target in node.targets)
    ]
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_reader_metadata")
    namespace = {"sys": sys}
    module = ast.Module(body=constants + [function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["_reader_metadata"]("grounded", {}, parent_sources)


def check_wrong_update_stops_before_inference():
    hashes = {
        "parent": {"run.py": "parent"},
        "wrapper": {"train.py": evaluate.EXPECTED_TRAIN_SHA256,
                    "train.slurm": "slurm",
                    "prepare_data.py": evaluate.EXPECTED_PREPARE_DATA_SHA256},
        "data": {"manifest_sha256": evaluate.EXPECTED_MANIFEST_SHA256,
                 "records_sha256": evaluate.EXPECTED_RECORDS_SHA256},
    }
    pin = {"status": "verified", "updates": 2048, "final_fingerprint": "fp",
           "checkpoint_sha256": "checkpoint"}
    status = {
        "status": "trained", "updates": 2048, "completed_updates": 2048,
        "reader_mode": "grounded", "seed": 17, "mode": "hybrid",
        "arm": "joint_ar_diffusion_b32", "prediction_offset": 0,
        "block_size": 32, "clean_crystal_ar_weight": 1.0,
        "reader_schedule": "alternating_q1_q2", "final_fingerprint": "fp",
        "checkpoint_sha256": "checkpoint", "checkpoint_reload_verified": True,
        "checkpoint_reload_fingerprint": "fp", "code_hashes": hashes,
        "dataset_manifest_sha256": evaluate.EXPECTED_MANIFEST_SHA256,
        "dataset_records_sha256": evaluate.EXPECTED_RECORDS_SHA256,
    }
    receipt = {
        "status": "verified", "checkpoint_update": 2048,
        "reader_mode": "grounded", "mode": "hybrid",
        "arm": "joint_ar_diffusion_b32", "prediction_offset": 0,
        "block_size": 32, "clean_crystal_ar_weight": 1.0,
        "final_fingerprint": "fp", "reload_fingerprint": "fp",
        "reload_verified": True, "strict_state_load": True,
        "checkpoint_sha256": "checkpoint",
    }
    receipt.update(reader_metadata_from_source(hashes))
    summary = {
        "status": "training_only_checkpoint_verified", "stage": "scaled32_training_only",
        "updates": 2048, "reader_mode": "grounded", "seed": 17,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "clean_crystal_ar_weight": 1.0, "final_fingerprint": "fp",
        "checkpoint_reload_verified": True,
        "evaluation": {"train32": "not_run", "heldout16": "not_run",
                       "AR_crystal_inference": "not_run"},
    }
    assert evaluate.validate_training_receipts(pin, status, receipt, summary)[:2] == ("fp", "checkpoint")
    cases = (
        ({**pin, "updates": 2047}, status, receipt),
        (pin, status, {**receipt, "checkpoint_update": 2047}),
    )
    for bad_pin, good_status, bad_receipt in cases:
        inferred = []
        try:
            evaluate.validate_training_receipts(bad_pin, good_status, bad_receipt, summary)
        except ValueError as error:
            assert "2048" in str(error)
        else:
            inferred.append(True)
        assert inferred == []


def check_seen8_heldout16_and_excluded_train24():
    train_ids = [f"t{i:02}" for i in range(32)]
    heldout_ids = [f"h{i:02}" for i in range(16)]
    manifest = {"cohort": {"train_ids": train_ids, "heldout_ids": heldout_ids}}
    rows = ([{"id": case, "split": "train", "source_hashes": {"source_cif_sha256": case}}
             for case in train_ids[:8]]
            + [{"id": case, "split": "heldout", "source_hashes": {"source_cif_sha256": case}}
               for case in heldout_ids]
            + [{"id": case, "split": "train", "source_hashes": {"source_cif_sha256": case}}
               for case in train_ids[8:]])
    seen8, heldout16, remaining24 = evaluate.select_cohorts(manifest, rows)
    assert [row["id"] for row in seen8] == train_ids[:8]
    assert [row["id"] for row in heldout16] == heldout_ids
    assert [row["id"] for row in remaining24] == train_ids[8:]


def check_failed_read_attempt_stays_in_field_denominator():
    result = evaluate._attempt_fields(
        [
            {"read": {"ok": True}, "grade": {"fields": {"sg": True}}},
            {"read": {"ok": False, "error": "parse failure"}},
        ], "read", "grade", ("sg",),
    )
    assert result["attempted_cases"] == result["cohort_cases"] == 2
    assert result["successful_responses"] == 1
    assert result["field_metrics"]["sg"] == {"count": 1, "denominator": 2, "rate": 0.5}


def check_each_snapshot_rechecks_helper_tokenizer_and_codec_inputs():
    state = {"helper": "helper-v1", "tokenizer": "tokenizer-v1", "codec": "codec-v1"}
    calls = []

    def source_snapshot(*_):
        calls.append("sources")
        return {"parent": {"run.py": state["helper"]},
                "wrapper": {"train.py": evaluate.EXPECTED_TRAIN_SHA256},
                "data": {"records_sha256": evaluate.EXPECTED_RECORDS_SHA256},
                "evaluation_helpers": {"sampler.py": state["helper"]}}

    def tokenizer_snapshot(*_):
        calls.append("tokenizer")
        return {"tokenizer.json": state["tokenizer"]}

    def codec_snapshot(*_):
        calls.append("codec")
        return {"base_manifest_sha256": evaluate.EXPECTED_BASE_MANIFEST_SHA256,
                "codec_pin": {"codec.py": state["codec"]}}

    inputs = {
        "pin_path": Path("pin.json"), "paths": {
            "status_path": Path("status.json"),
            "checkpoint_receipt_path": Path("checkpoint_receipt.json"),
            "summary_path": Path("summary.json"), "checkpoint_path": Path("checkpoint.pt"),
        }, "fingerprint": "fingerprint",
    }
    runtime = {"scaled_data": SimpleNamespace(check_dataset=lambda: {
        "status": "check_passed_data_only", "rows": 48, "train": 32,
        "heldout": 16, "training_run": "not_run",
    })}
    with patch.object(evaluate, "sha_file", return_value="fixed-file-hash"), \
         patch.object(evaluate, "_verify_sources", side_effect=source_snapshot), \
         patch.object(evaluate, "_fresh_tokenizer_hashes", side_effect=tokenizer_snapshot), \
         patch.object(evaluate, "_fresh_codec_snapshot", side_effect=codec_snapshot):
        before = evaluate._snapshot(inputs, runtime)
        for key in ("helper", "tokenizer", "codec"):
            state[key] += "-changed"
            after = evaluate._snapshot(inputs, runtime)
            try:
                evaluate.assert_snapshot_unchanged(before, after)
            except ValueError:
                pass
            else:
                raise AssertionError(f"changed {key} input passed the frozen snapshot comparison")
            state[key] = state[key].removesuffix("-changed")
    assert calls == ["sources", "codec", "tokenizer"] * 4


if __name__ == "__main__":
    check_wrong_update_stops_before_inference()
    check_seen8_heldout16_and_excluded_train24()
    check_failed_read_attempt_stays_in_field_denominator()
    check_each_snapshot_rechecks_helper_tokenizer_and_codec_inputs()
    print("Scaled32 evaluation CPU checks passed")
