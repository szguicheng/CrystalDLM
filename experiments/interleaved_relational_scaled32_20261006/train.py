"""Train one scaled32 reader recipe, reusing the frozen joint B32 trainer."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
JOINT_ROOT = PROJECT / "experiments/interleaved_relational_joint_20261006"
sys.path.insert(0, str(JOINT_ROOT))
import run as runner
import prepare_data


DATA_DIR = prepare_data.DATA_DIR
UPDATES = 2048
SEED = 17
READER_MODES = ("plain", "grounded")
EXPECTED_RECORDS_SHA256 = "231f336234d15a216dc9cb800d8dc1c3f30cf90985184ccc0d3e5c26333e95c6"
EXPECTED_MANIFEST_SHA256 = "25f150360f173970895dafd31e390f94b4b01ca174e6ad09b9e288c94b1e48cc"
READER_RECIPE = (
    "Q1/Q2 and Q2 reader_direction follow the frozen parent schedule; grounded contexts "
    "alternate neutral and train-only cyclic-partner history by (epoch//4 + row_index) parity. "
    "Neutral Q2 contains no prior Q1 answer. The final START/END block is authoritative."
)
READER_DIRECTION_NOTE = (
    "Q2 follows the parent reader_direction and receives 16 minus and 16 plus examples per case; "
    "each Q2 direction receives 8 neutral and 8 conflicting contexts. Q1 reads unscaled C1."
)
CURRENT_READER_MODE = "grounded"
DATA_CHECK = None


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_data():
    """Run the reviewed read-only data checks and load its fixed split."""
    global DATA_CHECK
    DATA_CHECK = prepare_data.check_dataset()
    hashes = runner.data_source_hashes(DATA_DIR)
    assert hashes["records_sha256"] == EXPECTED_RECORDS_SHA256
    assert hashes["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    assert manifest["status"] == "data_prepared_not_trained"
    assert manifest["representation"] == "relational_cnl_v1"
    assert manifest["model_for_later_training"] == {
        "name": runner.EXPECTED_MODEL, "revision": runner.EXPECTED_REVISION,
    }
    assert manifest["capacity"] == {"K": runner.K, "MAXLEN": runner.MAXLEN}
    assert manifest["records"]["count"] == 48
    assert manifest["records"]["sha256"] == EXPECTED_RECORDS_SHA256
    budget = manifest["later_training_budget"]
    assert budget["epochs_per_train_case"] == 64 and budget["updates_for_train32"] == UPDATES
    assert budget["scheduled"] is False and budget["status"] == "not_run"
    assert manifest["prefix_preservation"]["prefix_byte_identical"] is True
    assert manifest["prefix_preservation"]["source_records_sha256"] == prepare_data.ORIGINAL24_RECORDS_SHA256
    assert manifest["prefix_preservation"]["output_prefix_sha256"] == prepare_data.ORIGINAL24_RECORDS_SHA256
    assert manifest["prefix_preservation"]["prefix_row_count"] == 24
    assert DATA_CHECK == {"status": "check_passed_data_only", "rows": 48, "train": 32,
                          "heldout": 16, "training_run": "not_run"}
    lines = (DATA_DIR / "records.jsonl").read_text().splitlines()
    assert len(lines) == 48
    train_rows = [json.loads(line) for line in lines[:8] + lines[24:]]
    heldout_ids = manifest["cohort"]["heldout_ids"]
    assert len(train_rows) == 32 and len(heldout_ids) == 16
    assert all(row["split"] == "train" for row in train_rows)
    assert [row["id"] for row in train_rows] == manifest["cohort"]["train_ids"]
    assert {row["id"] for row in train_rows}.isdisjoint(heldout_ids)
    return manifest, train_rows, heldout_ids


def _reader_prefix(base, row, partner, direction, mode, context_mode):
    task = base["task"]
    latest = row["c1"] if task == "Q1" else row[direction]
    suffix = (
        runner._read_question_suffix(base["prefix"], row["c1"])
        if task == "Q1" else runner._q2_suffix(base["prefix"], row[direction])
    )
    if mode == "plain":
        return base["prefix"], latest
    if context_mode == "neutral":
        history = (
            "User: Read the current crystal record below. No earlier answer applies.\n"
            "Assistant: The final crystal block is the current and authoritative record.\n"
        )
        return history + runner.START + latest + runner.END + suffix, latest

    previous = partner["c1"] if task == "Q1" else partner[direction]
    partner_task = runner.tasks(partner, "train", direction, 0)[1 if task == "Q1" else 3]
    history = (
        "User: Here is an earlier crystal record from another training case.\n"
        "Assistant:" + runner.START + previous + runner.END
        + "\nUser: Read that earlier record.\nAssistant:"
        + partner_task["target"]
        + "\nUser: The final crystal block is current and authoritative. Earlier facts may conflict.\n"
        "Assistant: I will use the latest block.\n"
    )
    return history + runner.START + latest + runner.END + suffix, latest


def update_examples(train_rows, update, reader_mode=None):
    """Use the parent G/E tasks and only replace the selected Q context."""
    reader_mode = reader_mode or CURRENT_READER_MODE
    assert reader_mode in READER_MODES
    assert len(train_rows) == 32 and 1 <= update <= UPDATES
    epoch, j = divmod(update - 1, len(train_rows))
    direction = "minus" if (epoch + j) % 2 == 0 else "plus"
    variant = 0
    row = train_rows[j]
    partner = train_rows[(j + 1) % len(train_rows)]
    g1, q1, edit, q2 = runner.tasks(row, "train", direction, variant)
    g2 = runner.tasks(train_rows[(j + len(train_rows) // 2) % len(train_rows)], "train", direction, variant)[0]
    reader_direction = None if epoch % 2 == 0 else (
        "minus" if (epoch // 2 + j) % 2 == 0 else "plus"
    )
    reader = q1 if reader_direction is None else runner.tasks(row, "train", reader_direction, variant)[3]
    occurrence = epoch // 2
    context_mode = (
        "plain" if reader_mode == "plain"
        else "neutral" if (epoch // 4 + j) % 2 == 0
        else "conflict"
    )
    body_direction = reader_direction or direction
    prefix, latest = _reader_prefix(reader, row, partner, body_direction, reader_mode, context_mode)
    reader = dict(
        reader, id=row["id"], prefix=prefix,
        reader_occurrence=occurrence, context_mode=context_mode,
        latest_body_sha256=sha256(latest.encode()),
        partner_id=partner["id"] if context_mode == "conflict" else None,
    )
    if reader_direction is not None:
        reader["reader_direction"] = reader_direction
    assert [g1["task"], g2["task"], edit["task"], reader["task"]] == [
        "G", "G", "E", "Q1" if epoch % 2 == 0 else "Q2",
    ]
    return epoch, j, direction, variant, [g1, g2, edit, reader]


def exposure_receipt(train_rows, reader_mode):
    assert reader_mode in READER_MODES and len(train_rows) == 32
    events = []
    for update in range(1, UPDATES + 1):
        epoch, j, direction, variant, examples = update_examples(train_rows, update, reader_mode)
        for order, example in enumerate(examples):
            material_id = train_rows[(j + len(train_rows) // 2) % len(train_rows)]["id"] if order == 1 else train_rows[j]["id"]
            event = {
                "update": update, "epoch": epoch, "j": j, "order": order,
                "id": material_id, "task": example["task"],
                "direction": direction, "variant": variant,
            }
            if example["task"] in ("Q1", "Q2"):
                event.update(
                    context_mode=example["context_mode"],
                    reader_occurrence=example["reader_occurrence"],
                )
                if "reader_direction" in example:
                    event["reader_direction"] = example["reader_direction"]
                if example["partner_id"] is not None:
                    event["partner_id"] = example["partner_id"]
            events.append(event)
    summary = runner.summarize_exposure(events)
    context_counts = {
        row["id"]: {
            task: dict(Counter(event["context_mode"] for event in events
                               if event["id"] == row["id"] and event["task"] == task))
            for task in ("Q1", "Q2")
        }
        for row in train_rows
    }
    direction_counts = {
        row["id"]: {
            task: dict(Counter(event.get("reader_direction", "source") for event in events
                               if event["id"] == row["id"] and event["task"] == task))
            for task in ("Q1", "Q2")
        }
        for row in train_rows
    }
    return {
        "reader_mode": reader_mode,
        "reader_recipe": READER_RECIPE,
        "reader_direction_note": READER_DIRECTION_NOTE,
        "events_sha256": sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()),
        "events": events,
        "context_counts": context_counts,
        "reader_direction_counts": direction_counts,
        **summary,
    }


def _worst_joint_batch(train_rows, tok, reader_schedule=runner.DEFAULT_READER_SCHEDULE):
    assert reader_schedule == runner.DEFAULT_READER_SCHEDULE and len(train_rows) == 32
    candidates = {role: None for role in ("G", "E", "Q")}
    for epoch in range(8):
        for j in range(32):
            update = epoch * 32 + j + 1
            _, _, direction, variant, examples = update_examples(train_rows, update)
            for order, example in enumerate(examples):
                role = "Q" if example["task"] in ("Q1", "Q2") else example["task"]
                if role not in candidates:
                    continue
                _, packed = runner.task_length(example, tok, runner.ARMS[0])
                current = candidates[role]
                if current is None or packed > current[0]:
                    index = (j + len(train_rows) // 2) % len(train_rows) if order == 1 else j
                    candidates[role] = (packed, train_rows[index]["id"], direction, variant, example)
    assert all(candidates.values())
    batch = [candidates["G"][4], candidates["G"][4], candidates["E"][4], candidates["Q"][4]]
    details = [
        {
            "task": item[4]["task"], "id": item[1], "direction": item[2], "variant": item[3],
            "sequence_kind": "dual_stream" if "crystal" in item[4] else "text",
            "worst_actual_packed_tokens": item[0],
            "actual_sequence_tokens_by_arm": {
                arm: runner.task_length(item[4], tok, arm)[1] for arm in runner.ARMS
            },
        }
        for item in candidates.values()
    ]
    return batch, details


def _reader_metadata(reader_mode, data_hashes, parent_sources):
    return {
        "training_stage": "scaled32_training_only",
        "python_version": sys.version.split()[0],
        "terminal_goal_achieved": False,
        "reader_mode": reader_mode,
        "reader_recipe": READER_RECIPE,
        "reader_direction_note": READER_DIRECTION_NOTE,
        "data_hashes": data_hashes,
        "source_hashes": parent_sources,
    }


def run_training(reader_mode, out):
    global CURRENT_READER_MODE
    assert reader_mode in READER_MODES
    manifest, train_rows, heldout_ids = load_data()
    assert runner.MODEL == runner.EXPECTED_MODEL and runner.REVISION == runner.EXPECTED_REVISION
    data_hashes = runner.data_source_hashes(DATA_DIR)
    source_pin = {
        "data_check": DATA_CHECK,
        "manifest_sha256": data_hashes["manifest_sha256"],
        "records_sha256": data_hashes["records_sha256"],
        "representation": manifest["representation"],
        "model": manifest["model_for_later_training"],
        "old24_prefix": {
            "source_manifest_sha256": manifest["source_pins"]["original24_manifest_sha256"],
            "records_sha256": manifest["prefix_preservation"]["source_records_sha256"],
            "prefix_byte_identical": manifest["prefix_preservation"]["prefix_byte_identical"],
        },
        "blockrunner_code_sha256": manifest["source_pins"]["blockrunner_code_sha256"],
        "runtime_versions": manifest["source_pins"]["runtime_versions"],
        "tokenizer_files": prepare_data.read_json(prepare_data.BASE_MANIFEST_PATH)["provenance"]["tokenizer_files"],
        "initial_fingerprint": runner.EXPECTED_INITIAL_FINGERPRINT,
    }
    CURRENT_READER_MODE = reader_mode
    runner.UPDATES = UPDATES
    runner.update_examples = lambda rows_arg, update, reader_schedule=runner.DEFAULT_READER_SCHEDULE: update_examples(
        rows_arg, update, CURRENT_READER_MODE
    )
    runner._worst_joint_batch = lambda rows_arg, tok, reader_schedule=runner.DEFAULT_READER_SCHEDULE: _worst_joint_batch(
        rows_arg, tok, reader_schedule
    )
    parent_source_hashes = runner.runner_source_hashes

    def frozen_sources():
        return {
            "parent": parent_source_hashes(),
            "wrapper": {
                name: runner.sha_file(ROOT / name)
                for name in ("train.py", "train.slurm", "prepare_data.py")
            },
            "data": runner.data_source_hashes(DATA_DIR),
        }

    runner.runner_source_hashes = frozen_sources
    write_json = runner.write_json

    def training_json(path, value):
        if Path(path).name in ("status.json", "gpu_preflight.json", "checkpoint_receipt.json"):
            value.update(_reader_metadata(reader_mode, data_hashes, frozen_sources()))
        write_json(path, value)

    runner.write_json = training_json
    runner_hashes = frozen_sources()
    exposure = exposure_receipt(train_rows, reader_mode)
    tok = runner.tokenizer()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "training_exposure.json", {
        "updates": UPDATES, "epochs_per_train_case": 64,
        "effective_batch": 4, "microbatch": 1, "accumulation": runner.ACCUMULATION,
        "task_counts_per_update": {"G": 2, "E": 1, "reader": "Q1/Q2 alternate by epoch"},
        **exposure,
    })
    write_json(out / "run_metadata.json", {
        **_reader_metadata(reader_mode, data_hashes, runner_hashes),
        "seed": SEED, "updates": UPDATES, "train_ids": [row["id"] for row in train_rows],
        "heldout_ids_from_manifest_only": heldout_ids,
        "evaluation": {"train32": "not_run", "heldout16": "not_run", "AR_crystal_inference": "not_run"},
    })
    preflight_initial, preflight = runner.gpu_preflight(
        train_rows, tok, out, reader_schedule=runner.DEFAULT_READER_SCHEDULE
    )
    runner.assert_runner_sources(runner_hashes)
    model, tok, receipt = runner.train_one(
        runner.ARMS[0], train_rows, tok, DATA_DIR, manifest, data_hashes, source_pin,
        exposure, runner_hashes, preflight_initial, out,
        reader_schedule=runner.DEFAULT_READER_SCHEDULE,
    )
    assert receipt["status"] == "trained" and receipt["completed_updates"] == UPDATES
    assert receipt["checkpoint_reload_verified"] is True
    assert receipt["initial_fingerprint"] == runner.EXPECTED_INITIAL_FINGERPRINT
    del model
    runner.torch.cuda.empty_cache()
    assert prepare_data.check_dataset() == DATA_CHECK
    runner.assert_runner_sources(runner_hashes)
    assert runner.data_source_hashes(DATA_DIR) == data_hashes
    summary_path = out / "summary.json"
    write_json(summary_path, {
        "status": "training_only_checkpoint_verified",
        "stage": "scaled32_training_only",
        "terminal_goal_achieved": False,
        "reader_mode": reader_mode, "reader_recipe": READER_RECIPE,
        "reader_direction_note": READER_DIRECTION_NOTE,
        "model": runner.MODEL, "revision": runner.REVISION,
        "seed": SEED, "updates": UPDATES, "epochs_per_train_case": 64,
        "K": runner.K, "MAXLEN": runner.MAXLEN, "clean_crystal_ar_weight": runner.CLEAN_CRYSTAL_AR_WEIGHT,
        "initial_fingerprint": receipt["initial_fingerprint"],
        "final_fingerprint": receipt["final_fingerprint"],
        "checkpoint_reload_verified": receipt["checkpoint_reload_verified"],
        "actual_task_exposure_counts": receipt["actual_exposure_counts"],
        "actual_noise_step_exposure": receipt["actual_noise_step_exposure"],
        "source_pin": source_pin, "gpu_preflight": preflight,
        "evaluation": {"train32": "not_run", "heldout16": "not_run", "AR_crystal_inference": "not_run"},
    })
    return summary_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-mode", choices=READER_MODES, default="grounded")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not runner.torch.cuda.is_available():
        raise SystemExit("training requires the requested Slurm A100 allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing to train outside Slurm")
    out = args.out or ROOT / f"training_{args.reader_mode}_001"
    summary = run_training(args.reader_mode, out)
    print(json.dumps({"status": "training_only_checkpoint_verified", "summary": str(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
