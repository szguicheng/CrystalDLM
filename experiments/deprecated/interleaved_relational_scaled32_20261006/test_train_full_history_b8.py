"""CPU-only B8 training contract; no pretrained model or GPU is loaded."""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import train_full_history_b8 as train


class TinyTokenizer:
    def __init__(self):
        self.special = {
            train.runner.START: 1, train.runner.END: 2,
            train.runner.MASK: 3, train.runner.PAD: 0,
        }
        self.pattern = re.compile("(" + "|".join(map(re.escape, self.special)) + ")")

    def encode(self, text, add_special_tokens=False):
        ids = []
        for part in self.pattern.split(text):
            if not part:
                continue
            if part in self.special:
                ids.append(self.special[part])
            else:
                ids.extend(100 + len(token) for token in part.split())
        return ids


def demo():
    runner = train.runner
    base = train.BASE
    assert hashlib.sha256(train.FULL_HISTORY_PATH.read_bytes()).hexdigest() == train.FROZEN_FULL_HISTORY_SHA256
    assert train.FULL_HISTORY.FROZEN_TRAIN_SHA256 == "1bb619fb809d71c68813eb07aa162b2b2bfa1b8f2de7cef1a67b60054d4ae816"
    assert runner.ARMS == (train.B8_ARM,)
    assert runner.BLOCK_SIZES == {train.B8_ARM: 8}
    assert runner.MODE_OFFSETS == {train.B8_ARM: 0} and runner.CORE_MODES == {train.B8_ARM: "hybrid"}
    assert (base.UPDATES, base.SEED, runner.K, runner.STEPS, runner.MAXLEN) == (2048, 17, 2560, 32, 8192)
    assert runner.ACCUMULATION == 4 and runner.CLEAN_CRYSTAL_AR_WEIGHT == 1.0

    _, rows, heldout_ids = base.load_data()
    assert len(rows) == 32 and len(heldout_ids) == 16
    exposure = base.exposure_receipt(rows, "grounded")
    previous = json.loads((train.ROOT / "training_full_history_001/training_exposure.json").read_text())
    assert len(exposure["events"]) == 2048 * 4
    assert exposure["events_sha256"] == previous["events_sha256"]
    assert exposure["events"] == previous["events"]
    assert (exposure["reader_recipe"], exposure["reader_direction_note"]) == (
        previous["reader_recipe"], previous["reader_direction_note"]
    )
    counts = defaultdict(Counter)
    for event in exposure["events"]:
        if event["task"] in ("Q1", "Q2"):
            counts[event["id"], event["task"], event.get("reader_direction", "source")][event["context_mode"]] += 1
    for row in rows:
        case = row["id"]
        assert counts[case, "Q1", "source"] == Counter(neutral=16, conflict=16)
        for direction in ("minus", "plus"):
            assert counts[case, "Q2", direction] == Counter(neutral=8, conflict=8)

    q1 = base.update_examples(rows, 2, "grounded")[4][3]
    q2 = base.update_examples(rows, 34, "grounded")[4][3]
    assert q1["task"] == "Q1" and q1["context_mode"] == "conflict"
    parent_q1 = runner.prepare.tasks(rows[1], "train", "plus", 0)[1]
    partner_generation = runner.prepare.tasks(rows[2], "train", "plus", 0)[0]
    suffix = runner._read_question_suffix(parent_q1["prefix"], rows[1]["c1"])
    question = "Read the current replacement crystal, ignoring the earlier design conditions. " + runner._read_text_question(suffix)
    expected_q1_prefix = (
        partner_generation["prefix"] + " I generated a crystal.\n" + runner.START + rows[1]["c1"]
        + runner.END + "\nUser: " + question + "\nAssistant:"
    )
    assert q1["prefix"] == expected_q1_prefix and q1["target"] == parent_q1["target"]
    parent_q2 = runner.prepare.tasks(rows[1], "train", q2["reader_direction"], 0)[3]
    assert q2["task"] == "Q2" and q2["context_mode"] == "conflict"
    assert q2["prefix"] == parent_q2["prefix"]

    Hybrid = runner.Hybrid
    tok = TinyTokenizer()
    example = {"prefix": "small context", "lead": runner.START, "crystal": "CNL sample"}
    special_ids = [tok.special[token] for token in (runner.START, runner.END, runner.MASK, runner.PAD)]
    packs = {}
    states = {}
    for block_size in (32, 8):
        model = Hybrid(SimpleNamespace(config=SimpleNamespace(hidden_size=8)), *special_ids,
                       block_size=block_size,
                       clean_crystal_ar_weight=runner.CLEAN_CRYSTAL_AR_WEIGHT)
        generator = runner.torch.Generator().manual_seed(base.SEED + 1000)
        packs[block_size] = model.pack(example, tok, "hybrid", generator)
        states[block_size] = generator.get_state()
        assert model.block_size == block_size
        assert model.clean_crystal_ar_weight == runner.CLEAN_CRYSTAL_AR_WEIGHT == 1.0
    pack32, pack8 = packs[32], packs[8]
    assert len(pack8["ids"]) == len(pack32["ids"]) == len(tok.encode(example["prefix"])) + 1 + 2 * runner.K
    assert pack8["groups"][0]["k"] == pack32["groups"][0]["k"]
    assert runner.torch.equal(states[8], states[32])
    assert pack8["groups"][0]["kind"] == pack32["groups"][0]["kind"] == "crystal"
    assert [group["kind"] for group in pack8["groups"]] == ["crystal", "text", "crystal_ar"]
    assert set(pack8["groups"][2]["weights"]) == {runner.CLEAN_CRYSTAL_AR_WEIGHT}
    start = len(tok.encode(example["prefix"])) + len(tok.encode(example["lead"]))
    for block_size, packed in ((8, pack8), (32, pack32)):
        allowed = packed["allowed"][0, 0]
        noisy_start = start + runner.K
        for offset in range(0, runner.K, block_size):
            query = noisy_start + offset
            assert allowed[query, :start].all()
            assert allowed[query, start:start + offset].all()
            assert not allowed[query, start + offset:start + runner.K].any()
            assert not allowed[query, noisy_start:noisy_start + offset].any()
            assert allowed[query, noisy_start + offset:noisy_start + offset + block_size].all()
            assert not allowed[query, noisy_start + offset + block_size:].any()
    assert not runner.torch.equal(pack8["allowed"], pack32["allowed"])

    hashes = runner.runner_source_hashes()
    for name in ("train_full_history_b8.py", "train_full_history_b8.slurm", "test_train_full_history_b8.py"):
        path = train.ROOT / name
        relative = Path(os.path.relpath(path, train.JOINT_ROOT)).as_posix()
        assert hashes[relative] == hashlib.sha256(path.read_bytes()).hexdigest()
    runner.assert_runner_sources(hashes)
    print("B8 CPU contract passed: B=8 pack/routes, same k draw and RNG advance; 2048-update reader exposure unchanged")


if __name__ == "__main__":
    demo()
