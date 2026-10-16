"""Small CPU contract for the fixed 2048-update B8 training adapter."""
from collections import Counter, defaultdict
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import train


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
    assert train.UPDATES == runner.UPDATES == 2048 and train.SEED == runner.SEED == 17
    assert runner.MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert runner.REVISION == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert runner.EXPECTED_INITIAL_FINGERPRINT == "e97aed646b7289c06f10be376b37d67cd7aa6a42ad6214b61da008cd85b8404b"
    assert (runner.ARMS, runner.BLOCK_SIZES, runner.MODE_OFFSETS) == (
        (train.B8_ARM,), {train.B8_ARM: 8}, {train.B8_ARM: 0},
    )
    model_module = train.MODEL_MODULE
    for module in (runner, runner.prepare, model_module):
        assert (module.K, module.MAXLEN) == (2720, 8448)

    manifest, rows, heldout_ids = train.load_data()
    assert len(rows) == 256 and len(heldout_ids) == 16
    assert manifest["records"]["count"] == 272
    assert train.DATA_HASHES == {
        "manifest_sha256": "483fbc8646b2223ff86cc15b34172cccf25a8c08f9bcac9b08797247ba9b1bbf",
        "records_sha256": "a84ca310fac45944dea7da64aa960fdf7b14891e6d2c1624a1fbc5df6a8c627c",
    }
    exposure = train.exposure_receipt(rows)
    assert len(exposure["events"]) == 2048 * 4
    assert exposure["total"] == {"E": 2048, "G": 4096, "Q1": 1024, "Q2": 1024}
    contexts = defaultdict(Counter)
    directions = defaultdict(Counter)
    for event in exposure["events"]:
        if event["task"] in ("Q1", "Q2"):
            key = (event["id"], event["task"], event.get("reader_direction", "source"))
            contexts[key][event["context_mode"]] += 1
            directions[event["id"], event["task"]][event.get("reader_direction", "source")] += 1
    for row in rows:
        assert exposure["per_material"][row["id"]]["tasks"] == {
            "E": 8, "G": 16, "Q1": 4, "Q2": 4,
        }
        assert contexts[row["id"], "Q1", "source"] == Counter(neutral=2, conflict=2)
        assert directions[row["id"], "Q2"] == Counter(minus=2, plus=2)
        for direction in ("minus", "plus"):
            assert contexts[row["id"], "Q2", direction] == Counter(neutral=1, conflict=1)

    q1_conflict = q2_conflict = None
    for update in range(1, train.UPDATES + 1):
        epoch, j, direction, _, examples = train.update_examples(rows, update)
        reader = examples[3]
        if reader["task"] == "Q1" and reader["context_mode"] == "conflict":
            q1_conflict = (reader, rows[j], rows[(j + 1) % len(rows)], direction)
        elif reader["task"] == "Q2" and reader["context_mode"] == "conflict":
            q2_conflict = (reader, rows[j], reader["reader_direction"])
        if q1_conflict and q2_conflict:
            break
    q1, row, partner, direction = q1_conflict
    parent_q1 = runner.tasks(row, "train", direction, 0)[1]
    partner_g = runner.tasks(partner, "train", direction, 0)[0]
    suffix = runner._read_question_suffix(parent_q1["prefix"], row["c1"])
    expected_q1 = (
        partner_g["prefix"] + " I generated a crystal.\n" + runner.START + row["c1"]
        + runner.END + "\nUser: Read the current replacement crystal, ignoring the earlier design conditions. "
        + runner._read_text_question(suffix) + "\nAssistant:"
    )
    assert q1["prefix"] == expected_q1 and q1["target"] == parent_q1["target"]
    assert q1["partner_id"] == partner["id"]
    q2, row, direction = q2_conflict
    parent_q2 = runner.tasks(row, "train", direction, 0)[3]
    assert q2["prefix"] == parent_q2["prefix"]
    assert q2["partner_id"] is None
    assert all("partner_id" not in event for event in exposure["events"] if event["task"] == "Q2")

    tok = TinyTokenizer()
    seen_updates = []
    old_update_examples, old_task_length = train.update_examples, runner.task_length
    def scan_update(rows_arg, update, reader_mode=train.READER_MODE):
        seen_updates.append(update)
        tasks = ["G", "G", "E", "Q1" if (update - 1) // 256 % 2 == 0 else "Q2"]
        examples = [{"task": task, "update": update, "prefix": str(update),
                     **({"crystal": "x", "lead": runner.START} if task in ("G", "E")
                        else {"target": "x"})} for task in tasks]
        return (update - 1) // 256, (update - 1) % 256, "minus", 0, examples
    try:
        train.update_examples = scan_update
        runner.task_length = lambda example, tok_arg, arm: (None, example["update"])
        _, worst = train.worst_joint_batch(rows, tok)
    finally:
        train.update_examples = old_update_examples
        runner.task_length = old_task_length
    assert seen_updates == list(range(1, 2049))
    assert all(item["candidate_updates_checked"] == 2048 for item in worst)

    ids = [tok.special[token] for token in (runner.START, runner.END, runner.MASK, runner.PAD)]
    tiny = runner.Hybrid(SimpleNamespace(config=SimpleNamespace(hidden_size=8)), *ids,
                         block_size=8, clean_crystal_ar_weight=1.0)
    example = {"prefix": "small context", "lead": "Generate " + runner.START,
               "crystal": "CNL sample", "noise_step": runner.STEPS}
    packed = tiny.pack(example, tok, "hybrid", runner.torch.Generator().manual_seed(17))
    start = len(tok.encode(example["prefix"])) + len(tok.encode(example["lead"]))
    assert tiny.block_size == 8
    assert len(packed["ids"]) == start + 2 * 2720
    assert train.runner.task_length(example, tok, train.B8_ARM)[1] == start + 2 * 2720
    assert [group["kind"] for group in packed["groups"]] == ["crystal", "text", "crystal_ar"]
    allowed = packed["allowed"][0, 0]
    noisy_start = start + 2720
    for offset in (0, 8, 16, 2712):
        query = noisy_start + offset
        clean_end = start + offset
        assert allowed[query, :start].all()
        assert allowed[query, start:clean_end].all()
        assert not allowed[query, clean_end:start + 2720].any()
        assert not allowed[query, noisy_start:query].any()
        assert allowed[query, query:query + 8].all()
        assert not allowed[query, query + 8:].any()

    source_hashes = runner.runner_source_hashes()
    assert set(source_hashes) == {"parent", "coverage256"}
    runner.assert_runner_sources(source_hashes)

    print("coverage256 B8 CPU contract passed: 256/16 split, 8192 exposure events, full-history contexts, K=2720/MAXLEN=8448")


if __name__ == "__main__":
    demo()
