"""CPU-only contract check; loads data/tokenizer but allocates no model or GPU."""
from collections import Counter, defaultdict
import hashlib
import os

import train_full_history as train


def demo():
    _, rows, heldout_ids = train.BASE.load_data()
    assert len(rows) == 32 and len(heldout_ids) == 16
    assert {row["id"] for row in rows}.isdisjoint(heldout_ids)

    base, runner = train.BASE, train.runner
    base._reader_prefix = train._ORIGINAL_READER_PREFIX
    old_exposure = base.exposure_receipt(rows, "grounded")
    old_examples = [base.update_examples(rows, update, "grounded") for update in range(1, train.UPDATES + 1)]
    base._reader_prefix = train._reader_prefix
    new_exposure = base.exposure_receipt(rows, "grounded")
    new_examples = [base.update_examples(rows, update, "grounded") for update in range(1, train.UPDATES + 1)]
    assert old_exposure["events"] == new_exposure["events"]
    assert old_exposure["events_sha256"] == new_exposure["events_sha256"]
    assert all(before[:4] == after[:4] for before, after in zip(old_examples, new_examples))
    assert all(before[4][:3] == after[4][:3] for before, after in zip(old_examples, new_examples))

    counts = defaultdict(Counter)
    for event in new_exposure["events"]:
        if event["task"] in ("Q1", "Q2"):
            counts[event["id"], event["task"], event.get("reader_direction", "source")][event["context_mode"]] += 1
    for row in rows:
        case = row["id"]
        assert counts[case, "Q1", "source"] == Counter(neutral=16, conflict=16)
        for direction in ("minus", "plus"):
            assert counts[case, "Q2", direction] == Counter(neutral=8, conflict=8)

    for old, new in zip(old_examples, new_examples):
        epoch, j, direction, _, old_items = old
        _, _, _, _, items = new
        reader = items[3]
        row, partner = rows[j], rows[(j + 1) % len(rows)]
        assert reader["target"] == old_items[3]["target"]
        if reader["context_mode"] == "neutral":
            assert reader["prefix"] == old_items[3]["prefix"]
        elif reader["task"] == "Q2":
            parent_q2 = runner.prepare.tasks(row, "train", reader["reader_direction"], 0)[3]
            assert reader["prefix"] == parent_q2["prefix"]
        else:
            parent = runner.prepare.tasks(row, "train", direction, 0)[1]
            partner_g = runner.prepare.tasks(partner, "train", direction, 0)[0]
            suffix = runner._read_question_suffix(parent["prefix"], row["c1"])
            question = (
                "Read the current replacement crystal, ignoring the earlier design conditions. "
                + runner._read_text_question(suffix)
            )
            expected = (
                partner_g["prefix"] + " I generated a crystal.\n" + runner.START + row["c1"]
                + runner.END + "\nUser: " + question + "\nAssistant:"
            )
            assert reader["prefix"] == expected
            assert reader["target"] not in reader["prefix"]

    source_hashes = runner.runner_source_hashes()
    for name in ("train_full_history.py", "train_full_history.slurm", "test_train_full_history.py"):
        path = train.ROOT / name
        relative = os.path.relpath(path, train.JOINT_ROOT)
        assert source_hashes[relative] == hashlib.sha256(path.read_bytes()).hexdigest()

    print("full-history CPU contract passed: contexts 16/16, unchanged G/E and exposure schedule")


if __name__ == "__main__":
    demo()
