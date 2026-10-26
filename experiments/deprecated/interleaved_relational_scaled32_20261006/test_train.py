"""Small CPU-only contract check for the frozen scaled32 training recipe."""
from collections import Counter, defaultdict
from types import SimpleNamespace

import train


def demo():
    manifest, train_rows, heldout_ids = train.load_data()
    train_ids = {row["id"] for row in train_rows}
    heldout_id_set = set(heldout_ids)
    assert len(train_rows) == 32 and len(heldout_ids) == 16
    assert train_ids.isdisjoint(heldout_id_set)
    assert manifest["cohort"]["train_ids"] == [row["id"] for row in train_rows]
    assert manifest["cohort"]["heldout_ids"] == heldout_ids

    receipt = train.exposure_receipt(train_rows, "grounded")
    assert len(receipt["events"]) == 2048 * 4
    by_id_task = defaultdict(Counter)
    by_id_task_context = defaultdict(Counter)
    by_id_task_direction = defaultdict(Counter)
    by_id_task_direction_context = defaultdict(Counter)
    partners = {}
    for event in receipt["events"]:
        assert event["id"] in train_ids
        by_id_task[event["id"]][event["task"]] += 1
        if event["task"] in ("Q1", "Q2"):
            by_id_task_context[event["id"], event["task"]][event["context_mode"]] += 1
            reader_direction = event.get("reader_direction", "source")
            by_id_task_direction[event["id"], event["task"]][reader_direction] += 1
            by_id_task_direction_context[event["id"], event["task"], reader_direction][event["context_mode"]] += 1
            if event["context_mode"] == "conflict":
                assert event["partner_id"] in train_ids and event["partner_id"] != event["id"]
                assert partners.setdefault(event["id"], event["partner_id"]) == event["partner_id"]
    for row in train_rows:
        material_id = row["id"]
        assert by_id_task[material_id] == Counter(G=128, E=64, Q1=32, Q2=32)
        assert by_id_task_context[material_id, "Q1"] == Counter(neutral=16, conflict=16)
        assert by_id_task_direction[material_id, "Q1"] == Counter(source=32)
        assert by_id_task_direction[material_id, "Q2"] == Counter(minus=16, plus=16)
        for direction in ("minus", "plus"):
            assert by_id_task_direction_context[material_id, "Q2", direction] == Counter(neutral=8, conflict=8)
    assert partners == {
        row["id"]: train_rows[(index + 1) % len(train_rows)]["id"]
        for index, row in enumerate(train_rows)
    }
    assert set(partners.values()) == train_ids

    prep = train.runner.prepare
    start, end = prep.START, prep.END
    for update in range(1, train.UPDATES + 1):
        epoch, j, direction, variant, examples = train.update_examples(train_rows, update, "grounded")
        row, partner = train_rows[j], train_rows[(j + 1) % len(train_rows)]
        stock = prep.tasks(row, "train", direction, variant)
        other_g = prep.tasks(train_rows[(j + len(train_rows) // 2) % len(train_rows)], "train", direction, variant)[0]
        assert examples[:3] == [stock[0], other_g, stock[2]]
        reader = examples[3]
        assert reader["id"] == row["id"] and reader["task"] == ("Q1" if epoch % 2 == 0 else "Q2")
        reader_direction = None if reader["task"] == "Q1" else reader["reader_direction"]
        stock_reader = stock[1] if reader_direction is None else prep.tasks(row, "train", reader_direction, variant)[3]
        assert reader["target"] == stock_reader["target"]
        latest = row["c1"] if reader_direction is None else row[reader_direction]
        prefix = reader["prefix"]
        assert prefix.rpartition(start)[2].partition(end)[0] == latest
        if reader["context_mode"] == "neutral":
            assert prefix.count(start) == 1
            if reader["task"] == "Q2":
                assert prep.tasks(row, "train", reader_direction, variant)[1]["target"] not in prefix
        else:
            previous = partner["c1"] if reader_direction is None else partner[reader_direction]
            partner_task = prep.tasks(partner, "train", reader_direction or direction, variant)[1 if reader_direction is None else 3]
            assert reader["partner_id"] == partner["id"]
            assert prefix.count(start) == 2
            assert prefix.partition(start)[2].partition(end)[0] == previous
            assert partner_task["target"] in prefix

    _, _, _, _, plain_q1 = train.update_examples(train_rows, 1, "plain")
    _, _, _, _, grounded_q1 = train.update_examples(train_rows, 1, "grounded")
    assert plain_q1[3]["prefix"] == prep.tasks(train_rows[0], "train", "minus", 0)[1]["prefix"]
    assert plain_q1[3]["target"] == grounded_q1[3]["target"]
    plain_q2 = train.update_examples(train_rows, 33, "plain")[4][3]
    grounded_q2 = train.update_examples(train_rows, 33, "grounded")[4][3]
    q2_direction = grounded_q2["reader_direction"]
    assert plain_q2["prefix"] == prep.tasks(train_rows[0], "train", q2_direction, 0)[3]["prefix"]
    assert plain_q2["target"] == grounded_q2["target"]

    tok = train.runner.tokenizer()
    tiny = train.runner.Hybrid(
        SimpleNamespace(config=SimpleNamespace(hidden_size=8)),
        *[tok.convert_tokens_to_ids(token) for token in (prep.START, prep.END, prep.MASK, prep.PAD)],
        block_size=32,
    )
    torch = train.runner.torch
    stock_examples = prep.tasks(train_rows[0], "train", "minus", 0)
    g2_example = prep.tasks(train_rows[len(train_rows) // 2], "train", "minus", 0)[0]
    for stock_example, grounded_example in zip(
        (stock_examples[0], g2_example, stock_examples[2]), grounded_q1[:3]
    ):
        g1, g2 = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
        p1 = tiny.pack(stock_example, tok, "hybrid", g1)
        p2 = tiny.pack(grounded_example, tok, "hybrid", g2)
        assert p1["groups"] == p2["groups"]
        assert all(torch.equal(p1[key], p2[key]) for key in ("ids", "allowed", "times", "position_ids"))
        assert torch.equal(g1.get_state(), g2.get_state())
    plain_q2_conflict = train.update_examples(train_rows, 161, "plain")[4][3]
    grounded_q2_conflict = train.update_examples(train_rows, 161, "grounded")[4][3]
    assert plain_q2_conflict["target"] == grounded_q2_conflict["target"]
    for plain_reader, grounded_reader in (
        (plain_q1[3], grounded_q1[3]), (plain_q2, grounded_q2),
        (plain_q2_conflict, grounded_q2_conflict),
    ):
        g1, g2 = torch.Generator().manual_seed(17), torch.Generator().manual_seed(17)
        p1 = tiny.pack(plain_reader, tok, "hybrid", g1)
        p2 = tiny.pack(grounded_reader, tok, "hybrid", g2)
        assert p1["groups"][0]["labels"] == p2["groups"][0]["labels"]
        assert torch.equal(g1.get_state(), g2.get_state())
    print("scaled32 CPU contract passed; no model or GPU loaded")


if __name__ == "__main__":
    demo()
