"""CPU-only checks for the frozen reader-context prompt construction."""
from types import SimpleNamespace

import reader_context_probe as probe


def _suffix(prefix, marker):
    before, found, after = prefix.rpartition(marker)
    assert found
    return after


def check_actual_q2_matches_parent_concatenation_and_controls_share_c2():
    end, start = "<END>", "<START>"
    row = {"id": "case", "split": "train", "c1": "source-c1", "minus": "target-c2"}
    g = {"prefix": "G-PREFIX", "lead": "G-LEAD", "crystal": row["c1"]}
    q1_suffix = "\nUser: read c1\nAssistant:"
    q1 = {"prefix": g["prefix"] + "G-LEAD" + row["c1"] + end + q1_suffix, "target": "gold-q1"}
    e_suffix = "\nUser: edit\nAssistant:"
    edit = {"prefix": q1["prefix"] + q1["target"] + e_suffix}
    q2_suffix = "\nUser: read q2\nAssistant:"
    q2 = {"prefix": edit["prefix"] + "E-LEAD" + row["minus"] + end + q2_suffix}
    tasks = (g, q1, edit, q2)

    runner = SimpleNamespace(
        END=end, START=start, tasks=lambda *_: tasks,
        _read_question_suffix=lambda prefix, crystal: _suffix(prefix, crystal + end),
        _edit_instruction_suffix=lambda prefix, answer: _suffix(prefix, answer),
        _q2_suffix=lambda prefix, crystal: _suffix(prefix, crystal + end),
    )
    prepare = SimpleNamespace(END=end)
    actual_read, c2 = "actual-q1", "CURRENT-C2"
    generated = {"lead": "G-LEAD", "crystal": "generated-c1"}
    actual_edit = {"lead": "E-LEAD", "crystal": c2}
    history, suffix, q2_task = probe._actual_q2_parts(
        runner, prepare, row, "minus", generated, actual_read, actual_edit
    )
    expected_history = (
        "G-PREFIX" + "G-LEAD" + "generated-c1" + end + q1_suffix
        + actual_read + e_suffix + "E-LEAD" + c2 + end
    )
    assert history == expected_history and suffix == q2_suffix and q2_task is q2

    def neutral_reader(base, current, _partner, direction, mode, context_mode):
        assert direction == "minus" and mode == "grounded" and context_mode == "neutral"
        current_c2 = current[direction]
        return "neutral:" + start + current_c2 + end + _suffix(base["prefix"], current_c2 + end), current_c2

    neutral = probe._neutral_prompt(
        SimpleNamespace(_reader_prefix=neutral_reader), runner, row, "minus", q2_task, c2
    )
    instructed = probe._cue_prompt(history, suffix, probe.INSTRUCTION)
    prompts = (history + suffix, neutral, instructed)
    assert all(c2 in prompt for prompt in prompts)
    assert instructed == history + "\nUser: " + probe.INSTRUCTION + " read q2\nAssistant:"
    assert probe.INSTRUCTION == (
        "Read the final crystal block. Earlier answers describe the previous crystal "
        "and must not be copied."
    )


def check_swap_prefix_keeps_exact_partner_block_and_query():
    end, start = "<END>", "<START>"
    row = {"id": "seen", "split": "train", "c1": "source-block", "minus": "edit"}
    partner = {"id": "partner", "split": "train", "c1": "PARTNER-BLOCK", "minus": "partner-edit"}
    question = "What is the structure?"
    q1 = lambda value: {"task": "Q1", "prefix": "earlier" + value + end + "\nUser: " + question + "\nAssistant:"}
    tasks = {
        "seen": ({"prefix": "G-PREFIX"}, q1(row["c1"]), {}, {}),
        "partner": ({"prefix": "P-G"}, q1(partner["c1"]), {}, {}),
    }
    runner = SimpleNamespace(
        END=end, START=start, tasks=lambda value, *_: tasks[value["id"]],
        _read_question_suffix=lambda prefix, crystal: _suffix(prefix, crystal + end),
        _read_text_question=lambda suffix: suffix[len("\nUser: "):-len("\nAssistant:")],
    )
    history, suffix, partner_q1 = probe._swap_parts(runner, row, partner, "minus")
    question_suffix = (
        "\nUser: Read the current replacement crystal, ignoring the earlier design conditions. "
        + question + "\nAssistant:"
    )
    assert history == "G-PREFIX I generated a crystal.\n" + start + partner["c1"] + end
    assert suffix == question_suffix and partner_q1 is tasks["partner"][1]

    def neutral_reader(base, current, _partner, direction, mode, context_mode):
        assert base is partner_q1 and current is partner and direction == "minus"
        assert mode == "grounded" and context_mode == "neutral"
        return "neutral:" + start + current["c1"] + end + "\nUser: " + question + "\nAssistant:", current["c1"]

    neutral = probe._neutral_swap_prompt(
        SimpleNamespace(_reader_prefix=neutral_reader), runner, partner, "minus", partner_q1
    )
    instructed = probe._cue_prompt(history, suffix, probe.SWAP_INSTRUCTION)
    assert partner["c1"] in history and partner["c1"] in neutral and partner["c1"] in instructed
    assert instructed.startswith(history + "\nUser: " + probe.SWAP_INSTRUCTION + " ")
    assert probe.SWAP_INSTRUCTION == "Read the latest crystal block. Ignore earlier answers and design conditions."
    assert not any(char.isdigit() for char in probe.SWAP_INSTRUCTION)


def check_replay_receipt_uses_success_and_failure_attempts():
    receipt = probe._replay_summary([
        {"row": {"id": "raw"}, "stored": {"ok": True},
         "attempt_match": True, "exact_raw_match": True},
        {"row": {"id": "failed"}, "stored": {"ok": False},
         "attempt_match": True, "exact_raw_match": False},
    ])
    assert receipt == {
        "attempts_match": True, "raw_output_denominator": 1,
        "exact_raw_cases": ["raw"], "matched_failed_cases": ["failed"],
    }


if __name__ == "__main__":
    check_actual_q2_matches_parent_concatenation_and_controls_share_c2()
    check_swap_prefix_keeps_exact_partner_block_and_query()
    check_replay_receipt_uses_success_and_failure_attempts()
    print("Reader-context CPU prompt check passed")
