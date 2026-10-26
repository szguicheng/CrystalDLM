"""Compare frozen seen8 Q2 and replacement-read contexts."""
import argparse
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
BLOCK_ROOT = HERE.parents[1]
PROJECT = BLOCK_ROOT.parents[1]
SCALED_ROOT = PROJECT / "experiments/interleaved_relational_scaled32_20261006"
EVAL_OUT = SCALED_ROOT / "eval_grounded_final2048_001"
DEFAULT_OUT = HERE / "reader_context_probe_001"
INSTRUCTION = (
    "Read the final crystal block. Earlier answers describe the previous crystal "
    "and must not be copied."
)
SWAP_INSTRUCTION = "Read the latest crystal block. Ignore earlier answers and design conditions."


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _actual_q2_parts(runner, prepare, row, direction, generation, actual_read, actual_edit):
    g_task, q_task, e_task, q2_task = runner.tasks(row, row["split"], direction, 0)
    q_suffix = runner._read_question_suffix(q_task["prefix"], row["c1"])
    actual_q_prefix = g_task["prefix"] + generation["lead"] + generation["crystal"] + prepare.END + q_suffix
    edit_suffix = runner._edit_instruction_suffix(e_task["prefix"], q_task["target"])
    actual_edit_prefix = actual_q_prefix + actual_read + edit_suffix
    history_prefix = (
        actual_edit_prefix + actual_edit["lead"] + actual_edit["crystal"] + prepare.END
    )
    q2_suffix = runner._q2_suffix(q2_task["prefix"], row[direction])
    return history_prefix, q2_suffix, q2_task


def _neutral_prompt(train, runner, row, direction, q2_task, c2):
    updated_row = dict(row, **{direction: c2})
    old_block = row[direction] + runner.END
    assert q2_task["prefix"].count(old_block) == 1
    base = dict(q2_task, prefix=q2_task["prefix"].replace(old_block, c2 + runner.END, 1))
    prompt, latest = train._reader_prefix(
        base, updated_row, updated_row, direction, "grounded", "neutral"
    )
    assert latest == c2
    return prompt


def _swap_parts(runner, row, partner, direction):
    g_task, q_task, _, _ = runner.tasks(row, row["split"], direction, 0)
    question = runner._read_text_question(
        runner._read_question_suffix(q_task["prefix"], row["c1"])
    )
    suffix = (
        "\nUser: Read the current replacement crystal, ignoring the earlier design conditions. "
        + question + "\nAssistant:"
    )
    history_prefix = (
        g_task["prefix"] + " I generated a crystal.\n" + runner.START
        + partner["c1"] + runner.END
    )
    _, partner_q1, _, _ = runner.tasks(partner, "train", direction, 0)
    return history_prefix, suffix, partner_q1


def _neutral_swap_prompt(train, runner, partner, direction, partner_q1):
    prompt, latest = train._reader_prefix(
        partner_q1, partner, partner, direction, "grounded", "neutral"
    )
    assert latest == partner["c1"]
    return prompt


def _cue_prompt(history_prefix, suffix, instruction):
    assert suffix.startswith("\nUser: ")
    return history_prefix + suffix.replace("\nUser: ", "\nUser: " + instruction + " ", 1)


def _same_attempt(stored, replay):
    if stored.get("ok") is not replay.get("ok"):
        return False
    if stored.get("ok") is True:
        return isinstance(stored.get("value"), str) and isinstance(replay.get("value"), str) \
            and stored["value"].encode("utf-8") == replay["value"].encode("utf-8")
    return stored.get("error") == replay.get("error")


def _raw_exact(stored, replay):
    return stored.get("ok") is True and replay.get("ok") is True \
        and isinstance(stored.get("value"), str) and isinstance(replay.get("value"), str) \
        and stored["value"].encode("utf-8") == replay["value"].encode("utf-8")


def _replay_summary(replayed):
    return {
        "attempts_match": all(item["attempt_match"] for item in replayed),
        "raw_output_denominator": sum(item["stored"].get("ok") is True for item in replayed),
        "exact_raw_cases": [item["row"]["id"] for item in replayed if item["exact_raw_match"]],
        "matched_failed_cases": [item["row"]["id"] for item in replayed
                                 if item["stored"].get("ok") is False and item["attempt_match"]],
    }


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _require_terminal_summary(evaluator):
    path = EVAL_OUT / "summary.json"
    if not path.is_file():
        raise ValueError("final2048 evaluation summary is missing; refusing a live or partial input")
    summary = evaluator.read_json(path)
    evaluator.require(
        summary.get("status") == "complete"
        and summary.get("stage") == "scaled32_final_2048_evaluation",
        "final2048 evaluation is not terminal and complete",
    )
    return path


def _completed_inputs(evaluator, inputs, runtime):
    summary_path = EVAL_OUT / "summary.json"
    records_path = EVAL_OUT / "seen8/evaluation.jsonl"
    summary = evaluator.read_json(summary_path)
    evaluator.require(
        summary.get("status") == "complete"
        and summary.get("stage") == "scaled32_final_2048_evaluation",
        "final2048 evaluation is not terminal and complete",
    )
    evaluator.require(
        summary.get("training_checkpoint_sha256") == inputs["checkpoint_sha256"]
        and summary.get("fingerprint_before") == summary.get("fingerprint_after") == inputs["fingerprint"],
        "completed evaluation does not match the pinned checkpoint and fingerprint",
    )
    evaluator.require(summary.get("hashes_before") == summary.get("hashes_after"),
                      "completed evaluation input snapshots differ")
    cohort = evaluator._read_cohort(inputs, runtime)
    records = _read_jsonl(records_path)
    expected_ids = [row["id"] for row in cohort["seen8"]]
    evaluator.require(len(expected_ids) == 8 and len(records) == 8,
                      "raw completed evaluation must contain all eight seen cases")
    evaluator.require([record["id"] for record in records] == expected_ids
                      and summary["cohort"]["seen8_case_ids"] == expected_ids
                      and summary["seen8"]["n"] == 8,
                      "raw evaluation rows do not match the frozen seen8 order")
    return summary_path, records_path, summary, cohort, records


def _snapshot(evaluator, inputs, runtime, summary_path, records_path):
    return {
        "evaluation": evaluator._snapshot(inputs, runtime),
        "probe_source_sha256": evaluator.sha_file(__file__),
        "completed_evaluation_summary_sha256": evaluator.sha_file(summary_path),
        "completed_evaluation_seen8_sha256": evaluator.sha_file(records_path),
    }


def _response(model, tok, runner, prompt, mode):
    return runner.attempt(lambda: model.text(prompt, tok, mode), model)


def _grade(response, runner, c2_audit):
    return runner.grade_answer(response["value"], c2_audit) if response.get("ok") is True else None


def run(out=DEFAULT_OUT):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite reader-context probe: {out}")

    evaluator = _load_module("scaled32_evaluate_for_reader_context", SCALED_ROOT / "evaluate.py")
    _require_terminal_summary(evaluator)
    inputs = evaluator._load_inputs(evaluator.DEFAULT_PIN)
    runtime = evaluator._load_runtime()
    device = runtime["joint_eval"].header_probe.require_gpu_allocation()
    evaluator.require(evaluator.sha_file(inputs["paths"]["checkpoint_path"]) == inputs["checkpoint_sha256"],
                      "checkpoint hash differs from the final2048 pin")
    summary_path, records_path, _, cohort, original = _completed_inputs(
        evaluator, inputs, runtime
    )

    runner, prepare = runtime["runner"], runtime["prepare"]
    eligible = []
    for row, record in zip(cohort["seen8"], original):
        edit = record.get("actual_edit", {})
        actual_q2 = record.get("actual_second_read")
        if actual_q2 is None or edit.get("ok") is not True:
            continue
        if record.get("actual_read", {}).get("ok") is not True:
            raise ValueError(f"{row['id']}: Q2 was attempted without a successful actual Q1")
        c2 = edit["value"]["crystal"]
        c2_audit = runner.audit(c2)
        if c2_audit["valid"]:
            eligible.append((row, record, c2, c2_audit))
    evaluator.require(len(eligible) == 5,
                      f"expected five eligible seen8 actual-Q2 cases, found {len(eligible)}")

    before = _snapshot(evaluator, inputs, runtime, summary_path, records_path)
    tok = prepare.tokenizer()
    model, fingerprint_before = evaluator._load_final_model(inputs, runtime, tok)
    metadata = runtime["fast_eval"].install_and_check_fast_metadata(model, "cuda")
    text_mode = runner.CORE_MODES["joint_ar_diffusion_b32"]

    # All replay cases must match before either context contrast starts.
    replayed = []
    for row, record, c2, c2_audit in eligible:
        actual_read = record["actual_read"]["value"]
        generation = record["generation"]["value"]
        edit = record["actual_edit"]["value"]
        history_prefix, q2_suffix, q2_task = _actual_q2_parts(
            runner, prepare, row, record["direction"], generation, actual_read, edit
        )
        prompt = history_prefix + q2_suffix
        replay = _response(model, tok, runner, prompt, text_mode)
        stored = record["actual_second_read"]
        replayed.append({
            "task": "Q2_actual",
            "row": row, "source": record, "c2": c2, "c2_audit": c2_audit,
            "current_crystal": c2, "target_audit": c2_audit,
            "history_prefix": history_prefix, "q2_suffix": q2_suffix, "q2_task": q2_task,
            "prompt": prompt, "stored": stored, "response": replay,
            "attempt_match": _same_attempt(stored, replay),
            "exact_raw_match": _raw_exact(stored, replay),
        })

    seen_by_id = {row["id"]: row for row in cohort["seen8"]}
    swap_replayed = []
    for row, record in zip(cohort["seen8"], original):
        if "swap_read" not in record or not record.get("swap_id"):
            raise ValueError(f"{row['id']}: completed seen8 record lacks its replacement-read attempt")
        partner = seen_by_id[record["swap_id"]]
        if partner["id"] == row["id"]:
            raise ValueError(f"{row['id']}: replacement partner must be another frozen seen8 case")
        target_audit = runner.audit(partner["c1"])
        if not target_audit["valid"]:
            raise ValueError(f"{partner['id']}: frozen swap crystal failed score.audit")
        history_prefix, suffix, partner_q1 = _swap_parts(
            runner, row, partner, record["direction"]
        )
        prompt = history_prefix + suffix
        response = _response(model, tok, runner, prompt, text_mode)
        stored = record["swap_read"]
        swap_replayed.append({
            "task": "Q_swap", "row": row, "source": record, "partner": partner,
            "current_crystal": partner["c1"], "target_audit": target_audit,
            "history_prefix": history_prefix, "suffix": suffix, "partner_q1": partner_q1,
            "prompt": prompt, "stored": stored, "response": response,
            "attempt_match": _same_attempt(stored, response),
            "exact_raw_match": _raw_exact(stored, response),
        })
    replayed.extend(swap_replayed)
    mismatches = [f"{item['task']}:{item['row']['id']}" for item in replayed
                  if not item["attempt_match"]]
    if mismatches:
        raise AssertionError(f"full-history replay failed before contrasts: {mismatches}")
    evaluator.require(runner.fingerprint(model) == fingerprint_before == inputs["fingerprint"],
                      "model fingerprint changed during exact replay")

    train = _load_module("scaled32_train_reader_context", SCALED_ROOT / "train.py")
    assert train.runner is runner
    contrast_rows = []
    for item in replayed:
        row, record = item["row"], item["source"]
        current = item["current_crystal"]
        if item["task"] == "Q2_actual":
            neutral = _neutral_prompt(
                train, runner, row, record["direction"], item["q2_task"], current
            )
            instruction, suffix = INSTRUCTION, item["q2_suffix"]
        else:
            neutral = _neutral_swap_prompt(
                train, runner, item["partner"], record["direction"], item["partner_q1"]
            )
            instruction, suffix = SWAP_INSTRUCTION, item["suffix"]
        instructed = _cue_prompt(item["history_prefix"], suffix, instruction)
        assert all(current in prompt for prompt in (item["prompt"], neutral, instructed))
        neutral_response = _response(model, tok, runner, neutral, text_mode)
        instructed_response = _response(model, tok, runner, instructed, text_mode)
        target_audit = item["target_audit"]
        target_name = "generated_c2" if item["task"] == "Q2_actual" else "swap_partner_c1"
        contrast_rows.append({
            "task": item["task"], "id": row["id"], "direction": record["direction"],
            "swap_partner_id": item["partner"]["id"] if item["task"] == "Q_swap" else None,
            "current_crystal_sha256": evaluator.hashlib.sha256(current.encode("utf-8")).hexdigest(),
            "target_audit": {"target_source": target_name, "valid": target_audit["valid"],
                             "decoded_plan_sha256": target_audit["decoded_plan_sha256"]},
            "stored_read": {key: item["stored"].get(key)
                            for key in ("ok", "value", "error") if key in item["stored"]},
            "full_history_replay": {
                "prompt": item["prompt"], "response": item["response"],
                "attempt_match": item["attempt_match"],
                "exact_raw_match": item["exact_raw_match"],
                "response_success": item["response"].get("ok") is True,
                "grade": _grade(item["response"], runner, target_audit),
            },
            "grounded_training_neutral": {
                "prompt": neutral, "response": neutral_response,
                "response_success": neutral_response.get("ok") is True,
                "grade": _grade(neutral_response, runner, target_audit),
            },
            "full_history_with_instruction": {
                "prompt": instructed, "response": instructed_response,
                "response_success": instructed_response.get("ok") is True,
                "grade": _grade(instructed_response, runner, target_audit),
            },
        })

    fingerprint_after = runner.fingerprint(model)
    evaluator.require(fingerprint_after == fingerprint_before == inputs["fingerprint"],
                      "model fingerprint changed during reader-context probe")
    after = _snapshot(evaluator, inputs, runtime, summary_path, records_path)
    evaluator.assert_snapshot_unchanged(before, after)
    replay_receipts = {
        task: _replay_summary([item for item in replayed if item["task"] == task])
        for task in ("Q2_actual", "Q_swap")
    }

    out.mkdir(parents=True, exist_ok=False)
    with (out / "cases.jsonl").open("x") as stream:
        for row in contrast_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    runner.write_json(out / "summary.json", {
        "status": "complete", "stage": "bounded_reader_context_probe",
        "source_evaluation": str(EVAL_OUT.relative_to(PROJECT)),
        "source_evaluation_summary_sha256": before["completed_evaluation_summary_sha256"],
        "source_seen8_evaluation_sha256": before["completed_evaluation_seen8_sha256"],
        "checkpoint_sha256": inputs["checkpoint_sha256"],
        "fingerprint_before": fingerprint_before, "fingerprint_after": fingerprint_after,
        "device": device, "text_inference": "model.text autoregressive path; frozen hybrid B32 checkpoint mode",
        "fast_metadata_equivalence": metadata,
        "task_denominators": {
            "Q2_actual": {"seen8_cases": 8,
                           "attempted_actual_q2": sum("actual_second_read" in record for record in original),
                           "eligible_attempted_q2_with_valid_c2": len(eligible),
                           "replayed": sum(item["task"] == "Q2_actual" for item in replayed)},
            "Q_swap": {"seen8_cases": 8,
                       "attempted_swap_reads": sum("swap_read" in record for record in original),
                       "eligible_valid_partner_c1": len(swap_replayed),
                       "replayed": len(swap_replayed)},
        },
        "contrasts": ["full_history_replay", "grounded_training_neutral", "full_history_with_instruction"],
        "full_history_replay_attempts_match": all(receipt["attempts_match"]
                                                   for receipt in replay_receipts.values()),
        "task_replay_receipts": replay_receipts,
        "grade_target": "Q2 uses score.audit(actual C2); Q_swap uses score.audit(frozen partner C1); no target facts enter prompts",
        "neutral_interpretation": "Q2 and replacement-read context diagnostics only; no neutral result is a closed-loop success",
        "optimizer_updates": 0, "scope_goal_achieved": False,
        "hashes_before": before, "hashes_after": after,
    })
    del model
    runtime["torch"].cuda.empty_cache()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "complete", "output": str(run(args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
