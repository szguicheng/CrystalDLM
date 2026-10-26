"""Offline contract check for the Q2 coverage schedule."""
import json
from pathlib import Path

import prepare
import run as runner

ROOT = Path(__file__).resolve().parent
LEGACY_EVENT_SHA256 = "79e62e1a7e4f3c7e92ef5d4354fbd907d4c3ba90076924300861c43bb04f07f4"
Q2_EVENT_SHA256 = "d9c1f0576a30c62d81a9e00bdcf84cd13412057012b85da34d8c154f8424f560"


def test_q2_schedule():
    _, rows, train_rows, _ = runner.read_data(prepare.DATA_DIR)
    old_status = json.loads((ROOT / "pilot_002/hybrid_shift/status.json").read_text())
    assert old_status["actual_exposure_sha256"] == LEGACY_EVENT_SHA256
    assert runner.DEFAULT_READER_SCHEDULE == "q1_only"

    legacy = runner.exposure_receipt(train_rows, "q1_only")
    legacy_hash = runner.sha(
        json.dumps(legacy["events"], sort_keys=True, separators=(",", ":")).encode()
    )
    assert legacy_hash == LEGACY_EVENT_SHA256
    assert legacy["events_sha256"] == LEGACY_EVENT_SHA256
    assert runner.exposure_receipt(train_rows)["events"] == legacy["events"]

    exposure = runner.exposure_receipt(train_rows, "alternating_q1_q2")
    q2_hash = runner.sha(
        json.dumps(exposure["events"], sort_keys=True, separators=(",", ":")).encode()
    )
    assert q2_hash == Q2_EVENT_SHA256 == exposure["events_sha256"]
    assert exposure["total"] == {"E": 512, "G": 1024, "Q1": 256, "Q2": 256}
    actual = runner.summarize_exposure(exposure["events"])
    for material_id, counts in exposure["per_material"].items():
        assert counts["tasks"] == {"E": 64, "G": 128, "Q1": 32, "Q2": 32}
        assert counts["E_factors"] == {"minus": 32, "plus": 32}
        assert counts["Q2_factors"] == {"minus": 16, "plus": 16}
        assert actual["per_material"][material_id] == counts

    for update in range(1, runner.UPDATES + 1):
        epoch, j, _, _, examples = runner.update_examples(
            train_rows, update, "alternating_q1_q2"
        )
        if epoch % 2:
            q2 = examples[-1]
            expected_direction = "minus" if (epoch // 2 + j) % 2 == 0 else "plus"
            assert q2["task"] == "Q2"
            assert q2["reader_direction"] == expected_direction

    row = train_rows[0]
    for direction in ("minus", "plus"):
        q2 = prepare.tasks(row, "train", direction, 0)[3]
        assert row["c1"] in q2["prefix"]
        assert row[direction] in q2["prefix"]
        assert q2["target"] == (
            " " + prepare.score_answer(row["plans"][direction]) + prepare.EOS
        )

    lengths = runner.pack_audit(rows, prepare.tokenizer())
    assert prepare.K == 2560 and prepare.MAXLEN == 8192
    assert lengths["train"]["Q2"]["max"] <= prepare.MAXLEN
    assert lengths["heldout"]["Q2"]["max"] <= prepare.MAXLEN
    print(f"legacy_event_sha256={legacy_hash} q2_event_sha256={q2_hash}")


if __name__ == "__main__":
    test_q2_schedule()
    print("Q2 schedule checks passed")
