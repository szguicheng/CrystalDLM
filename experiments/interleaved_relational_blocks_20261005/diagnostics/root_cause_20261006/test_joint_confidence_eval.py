"""CPU-only receipt gate check; it does not instantiate a model."""
import copy

from joint_confidence_eval import (
    JOINT_CHECKPOINT_SHA256,
    JOINT_FINGERPRINT,
    JOINT_MODEL_SHA256,
    MANIFEST_SHA256,
    PREPARE_SHA256,
    RECORDS_SHA256,
    SCORE_SHA256,
    validate_joint_receipt,
)


def check_receipt_gate():
    status = {
        "arm": "joint_ar_diffusion_b32", "mode": "hybrid", "updates": 512,
        "prediction_offset": 0, "block_size": 32, "clean_crystal_ar_weight": 1.0,
        "final_fingerprint": JOINT_FINGERPRINT,
        "dataset_records_sha256": RECORDS_SHA256,
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "code_hashes": {
            "model.py": JOINT_MODEL_SHA256, "prepare.py": PREPARE_SHA256,
            "score.py": SCORE_SHA256,
        },
    }
    receipt = {
        "arm": "joint_ar_diffusion_b32", "mode": "hybrid", "checkpoint_update": 512,
        "prediction_offset": 0, "block_size": 32, "clean_crystal_ar_weight": 1.0,
        "final_fingerprint": JOINT_FINGERPRINT,
        "checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
        "reload_fingerprint": JOINT_FINGERPRINT,
        "reload_verified": True, "strict_state_load": True,
    }
    validate_joint_receipt(status, receipt)

    wrong_arm = copy.deepcopy(status)
    wrong_arm["arm"] = "diffusion_b32"
    try:
        validate_joint_receipt(wrong_arm, receipt)
    except AssertionError:
        pass
    else:
        raise AssertionError("receipt gate accepted the wrong training arm")


if __name__ == "__main__":
    check_receipt_gate()
    print("joint checkpoint receipt CPU check passed")
