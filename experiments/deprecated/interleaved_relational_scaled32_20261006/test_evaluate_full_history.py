"""CPU-only reader-receipt and frozen-evaluator adapter checks."""
import copy
import json
import sys

import evaluate_full_history as evaluation


def _old_artifacts():
    root = evaluation.HERE
    train = root / "training_grounded_001"
    return (
        json.loads((root / "eval_input_pin.json").read_text()),
        json.loads((train / "joint_ar_diffusion_b32/status.json").read_text()),
        json.loads((train / "joint_ar_diffusion_b32/checkpoint_receipt.json").read_text()),
        json.loads((train / "summary.json").read_text()),
    )


def _new_receipt_fixture():
    pin, status, receipt, summary = map(copy.deepcopy, _old_artifacts())
    for artifact in (status, receipt, summary):
        artifact["reader_recipe"] = evaluation.READER_RECIPE
    for sources in (status["code_hashes"]["parent"], receipt["source_hashes"]["parent"]):
        sources.update(evaluation.READER_SOURCE_PINS)
    paths = {
        "status_path": evaluation.ARM_DIR / "status.json",
        "checkpoint_receipt_path": evaluation.ARM_DIR / "checkpoint_receipt.json",
        "summary_path": evaluation.TRAIN_DIR / "summary.json",
        "checkpoint_path": evaluation.ARM_DIR / "checkpoint.pt",
    }
    for key, path in paths.items():
        pin[key] = path.relative_to(evaluation.PROJECT).as_posix()
    return pin, status, receipt, summary


def demo():
    assert "torch" not in sys.modules
    assert evaluation.SCALED_ROOT == evaluation.HERE
    assert evaluation.BASE.TRAIN_DIR == evaluation.HERE / "training_full_history_001"
    assert evaluation.BASE.ARM_DIR == evaluation.BASE.TRAIN_DIR / "joint_ar_diffusion_b32"
    assert evaluation.BASE.DEFAULT_PIN == evaluation.HERE / "eval_full_history_input_pin.json"
    assert evaluation.BASE.DEFAULT_OUT == evaluation.HERE / "eval_full_history_final2048_001"
    metrics = {"seen8": {"G": {"count": 3}}, "read_fact_fields": {"Q1_actual": 2}}
    augmented = evaluation.augment_summary(dict(metrics))
    assert {key: augmented[key] for key in metrics} == metrics
    assert augmented["training_variant"] == "full_history_reader_repair"
    assert augmented["reader_recipe"] == evaluation.READER_RECIPE

    old_pin, old_status, old_receipt, old_summary = _old_artifacts()
    try:
        evaluation.BASE._pin_path(old_pin, "status_path", evaluation.ARM_DIR / "status.json")
    except ValueError:
        pass
    else:
        raise AssertionError("grounded-training pin was accepted for full-history evaluation")
    try:
        evaluation.validate_training_receipts(old_pin, old_status, old_receipt, old_summary)
    except ValueError as error:
        assert "reader recipe" in str(error)
    else:
        raise AssertionError("old reader recipe passed the full-history receipt gate")

    pin, status, receipt, summary = _new_receipt_fixture()
    assert evaluation.validate_training_receipts(pin, status, receipt, summary)[:2] == (
        pin["final_fingerprint"], pin["checkpoint_sha256"],
    )
    for key, path in (
        ("status_path", evaluation.ARM_DIR / "status.json"),
        ("checkpoint_receipt_path", evaluation.ARM_DIR / "checkpoint_receipt.json"),
        ("summary_path", evaluation.TRAIN_DIR / "summary.json"),
        ("checkpoint_path", evaluation.ARM_DIR / "checkpoint.pt"),
    ):
        assert evaluation.BASE._pin_path(pin, key, path) == path.resolve()

    calls = []
    base_snapshot = evaluation._BASE_SNAPSHOT
    try:
        evaluation._BASE_SNAPSHOT = lambda *_: calls.append(True) or {"base": "snapshot"}
        snapshot = evaluation.snapshot_with_adapter_sources({}, {})
    finally:
        evaluation._BASE_SNAPSHOT = base_snapshot
    assert calls == [True]
    assert snapshot["full_history_evaluation_sources"] == {
        name: evaluation.BASE.sha_file(evaluation.HERE / name) for name in (
            "evaluate_full_history.py", "test_evaluate_full_history.py",
            "evaluate_full_history.slurm",
        )
    }
    assert "torch" not in sys.modules
    print("Full-history evaluation CPU receipt and source-snapshot checks passed")


if __name__ == "__main__":
    demo()
