"""Train the fixed 2048-update, block-size-8 coverage256 hybrid run."""
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
JOINT_ROOT = PROJECT / "experiments/interleaved_relational_joint_20261006"
SCALED_ROOT = PROJECT / "experiments/interleaved_relational_scaled32_20261006"
DATA_DIR = ROOT / "data_001"
RECORDS_PATH = DATA_DIR / "records.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
OLD48_RECORDS = SCALED_ROOT / "data_001/records.jsonl"
FULL_HISTORY_PATH = SCALED_ROOT / "train_full_history.py"
TRAIN_PATH = SCALED_ROOT / "train.py"
FROZEN_FULL_HISTORY_SHA256 = "e2954cf3eea94639782987596799a772b475d6a3b62eada787ad8a4820633112"
FROZEN_TRAIN_SHA256 = "1bb619fb809d71c68813eb07aa162b2b2bfa1b8f2de7cef1a67b60054d4ae816"
EXPECTED_MANIFEST_SHA256 = "483fbc8646b2223ff86cc15b34172cccf25a8c08f9bcac9b08797247ba9b1bbf"
EXPECTED_RECORDS_SHA256 = "a84ca310fac45944dea7da64aa960fdf7b14891e6d2c1624a1fbc5df6a8c627c"
OLD48_PREFIX_SHA256 = "231f336234d15a216dc9cb800d8dc1c3f30cf90985184ccc0d3e5c26333e95c6"
PREPARE_DATA_SHA256 = "fbd9ffaf2e795ca8d33eb77d6b5a85cdd2c8c30be910ac88b5848d31a0dc7c1d"
UPDATES, SEED, K, MAXLEN = 2048, 17, 2720, 8448
B8_ARM = "joint_ar_diffusion_b8"
TRAINING_STAGE = "coverage256_fixed2048_b8_training_only"
READER_MODE = "grounded"
READER_RECIPE = (
    "Eight epochs over 256 training cases; each update uses G_j, G_(j+128), E_j, and an alternating Q1/Q2 reader. "
    "Each case receives G=16, E=8, Q1=4, Q2=4; Q1 has 2 neutral/2 conflict contexts, and each Q2 direction has 1/1."
)
READER_DIRECTION_NOTE = (
    "The frozen parent alternation is retained: Q1 occurs in even epochs; Q2 occurs in odd epochs, with two "
    "examples per direction per case, split one neutral and one full-history conflict."
)
EXPECTED_DATA_HASHES = {
    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "records_sha256": EXPECTED_RECORDS_SHA256,
}
SPACE_GROUP_POLICY = {
    "new_224_rows": {
        "encoder": {"method": "spglib.get_symmetry_dataset", "symprec": 0.01, "angle_tolerance": 5},
        "physical_audit": {"method": "pymatgen.SpacegroupAnalyzer", "symprec": 0.01, "angle_tolerance": 5},
        "csv_spacegroup_number_role": "selection metadata only; never substituted into the plan",
    },
    "old48_prefix": {
        "status": "verbatim preserved and not re-encoded under the new policy",
        "records_sha256": OLD48_PREFIX_SHA256,
    },
}

for import_root in (str(ROOT), str(SCALED_ROOT), str(JOINT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
assert hashlib.sha256(FULL_HISTORY_PATH.read_bytes()).hexdigest() == FROZEN_FULL_HISTORY_SHA256
assert hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest() == FROZEN_TRAIN_SHA256
spec = importlib.util.spec_from_file_location("_coverage256_frozen_full_history", FULL_HISTORY_PATH)
FULL_HISTORY = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FULL_HISTORY
spec.loader.exec_module(FULL_HISTORY)
BASE = FULL_HISTORY.BASE
runner = FULL_HISTORY.runner
MODEL_MODULE = sys.modules[runner.Hybrid.__module__]

assert runner.MODEL == runner.EXPECTED_MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
assert runner.REVISION == runner.EXPECTED_REVISION == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
assert runner.EXPECTED_INITIAL_FINGERPRINT == "e97aed646b7289c06f10be376b37d67cd7aa6a42ad6214b61da008cd85b8404b"
assert runner.SEED == BASE.SEED == SEED
runner.ARMS = (B8_ARM,)
runner.MODE_OFFSETS = {B8_ARM: 0}
runner.CORE_MODES = {B8_ARM: "hybrid"}
runner.BLOCK_SIZES = {B8_ARM: 8}
runner.UPDATES = BASE.UPDATES = UPDATES
for module in (runner, runner.prepare, MODEL_MODULE):
    module.K, module.MAXLEN = K, MAXLEN
BASE.DATA_DIR = DATA_DIR

DATA_HASHES = dict(EXPECTED_DATA_HASHES)
DATA_CHECK = None
CURRENT_SOURCE_HASHES = None


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _sha(path):
    return sha256(Path(path).read_bytes())


def load_data():
    """Validate coverage256 pins, then return its train split and heldout IDs."""
    global DATA_HASHES, DATA_CHECK
    DATA_HASHES = runner.data_source_hashes(DATA_DIR)
    assert DATA_HASHES == EXPECTED_DATA_HASHES
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["status"] == "data_prepared_not_trained"
    assert manifest["representation"] == "relational_cnl_v1"
    assert manifest["capacity"] == {"K": K, "MAXLEN": MAXLEN}
    assert manifest["records"] == {
        "path": "records.jsonl", "count": 272, "sha256": EXPECTED_RECORDS_SHA256,
    }
    assert manifest["records_sha256"] == EXPECTED_RECORDS_SHA256
    assert manifest["cohort"]["total_records"] == 272
    assert manifest["cohort"]["train_count"] == 256 and manifest["cohort"]["heldout_count"] == 16
    assert manifest["space_group_policy"] == SPACE_GROUP_POLICY
    assert manifest["selection_protocol"]["rule"]["name"] == "coverage256_metadata_round_robin_v1"
    assert manifest["selection_protocol"]["rule"]["seed"] == SEED
    assert manifest["selection_protocol"]["rule"]["selection_failures_replaced"] is False
    assert manifest["selection_protocol"]["replacement_after_encoding_failure"] is False
    assert manifest["later_training"] == {"scheduled": False, "status": "not_run"}
    assert manifest["new_case_validation"]["selected"] == 224
    assert manifest["new_case_validation"]["prepared"] == 224
    assert manifest["new_case_validation"]["failed"] == 0
    assert set(manifest["new_case_validation"]["contracts_passed_per_case"]) >= {
        "exact_plan", "geometry", "canonical_text", "second_decode", "K2720_max8448",
    }
    assert manifest["capacity_measurement"]["maximum_body_tokens_including_END"] <= K
    assert manifest["capacity_measurement"]["maximum_required_context_tokens"] <= MAXLEN

    prefix = b"".join(RECORDS_PATH.read_bytes().splitlines(keepends=True)[:48])
    assert sha256(prefix) == OLD48_PREFIX_SHA256
    assert prefix == OLD48_RECORDS.read_bytes()
    preservation = manifest["prefix_preservation"]
    assert preservation["prefix_byte_identical"] is True
    assert preservation["prefix_rows"] == 48
    assert preservation["source_records_sha256"] == preservation["output_prefix_sha256"] == OLD48_PREFIX_SHA256

    preparation_hash = _sha(ROOT / "prepare_data.py")
    provenance = manifest["provenance"]
    assert preparation_hash == PREPARE_DATA_SHA256
    assert manifest["coverage256_prepare_data_sha256"] == provenance["coverage256_prepare_data_sha256"] == preparation_hash
    assert provenance["old48_records_sha256"] == OLD48_PREFIX_SHA256
    selection_path = PROJECT / manifest["cohort"]["selection_path"]
    assert _sha(selection_path) == manifest["cohort"]["selection_sha256"]
    assert provenance["scaled32_prepare_data_sha256"] == _sha(SCALED_ROOT / "prepare_data.py")
    assert provenance["blockrunner_code_sha256"] == {
        name: _sha(JOINT_ROOT / name) for name in ("prepare.py", "score.py")
    }
    token_files = {path.name: _sha(path) for path in sorted(runner.prepare.TOKENIZER_PATH.iterdir()) if path.is_file()}
    assert token_files == provenance["tokenizer_files"]

    rows = runner.prepare.load_records(path=RECORDS_PATH)
    assert len(rows) == 272 and len({row["id"] for row in rows}) == 272
    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    heldout_ids = [row["id"] for row in heldout_rows]
    assert len(train_rows) == 256 and len(heldout_rows) == 16
    assert [row["id"] for row in train_rows] == manifest["cohort"]["train_ids"]
    assert heldout_ids == manifest["cohort"]["heldout_ids"]
    assert {row["id"] for row in train_rows}.isdisjoint(heldout_ids)
    DATA_CHECK = {
        "status": "check_passed_data_only", "scope": "data_integrity_only",
        "rows": 272, "train": 256, "heldout": 16,
    }
    BASE.DATA_CHECK = DATA_CHECK
    return manifest, train_rows, heldout_ids


def update_examples(train_rows, update, reader_mode=READER_MODE):
    assert reader_mode == READER_MODE and len(train_rows) == 256 and 1 <= update <= UPDATES
    epoch, j = divmod(update - 1, len(train_rows))
    direction = "minus" if (epoch + j) % 2 == 0 else "plus"
    variant = 0
    row, partner = train_rows[j], train_rows[(j + 1) % len(train_rows)]
    g1, q1, edit, q2 = runner.tasks(row, "train", direction, variant)
    g2 = runner.tasks(train_rows[(j + len(train_rows) // 2) % len(train_rows)], "train", direction, variant)[0]
    reader_direction = None if epoch % 2 == 0 else (
        "minus" if (epoch // 2 + j) % 2 == 0 else "plus"
    )
    reader = q1 if reader_direction is None else runner.tasks(row, "train", reader_direction, variant)[3]
    context_mode = "neutral" if (epoch // 4 + j) % 2 == 0 else "conflict"
    prefix, latest = BASE._reader_prefix(
        reader, row, partner, reader_direction or direction, reader_mode, context_mode,
    )
    reader = dict(
        reader, id=row["id"], prefix=prefix, reader_occurrence=epoch // 2,
        context_mode=context_mode, latest_body_sha256=sha256(latest.encode()),
        partner_id=partner["id"] if context_mode == "conflict" and reader["task"] == "Q1" else None,
    )
    if reader_direction is not None:
        reader["reader_direction"] = reader_direction
    assert [item["task"] for item in (g1, g2, edit, reader)] == [
        "G", "G", "E", "Q1" if epoch % 2 == 0 else "Q2",
    ]
    return epoch, j, direction, variant, [g1, g2, edit, reader]


def exposure_receipt(train_rows, reader_mode=READER_MODE):
    assert reader_mode == READER_MODE and len(train_rows) == 256
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
                event.update(context_mode=example["context_mode"], reader_occurrence=example["reader_occurrence"])
                if "reader_direction" in example:
                    event["reader_direction"] = example["reader_direction"]
                if example["partner_id"] is not None:
                    event["partner_id"] = example["partner_id"]
            events.append(event)
    summary = runner.summarize_exposure(events)
    per_material = summary["per_material"]
    assert summary["total"] == {"E": 2048, "G": 4096, "Q1": 1024, "Q2": 1024}
    assert all(value["tasks"] == {"E": 8, "G": 16, "Q1": 4, "Q2": 4} for value in per_material.values())
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
    direction_context_counts = {
        row["id"]: {
            direction: dict(Counter(event["context_mode"] for event in events
                                    if event["id"] == row["id"] and event["task"] == "Q2"
                                    and event.get("reader_direction") == direction))
            for direction in ("minus", "plus")
        }
        for row in train_rows
    }
    for row in train_rows:
        assert context_counts[row["id"]]["Q1"] == {"neutral": 2, "conflict": 2}
        assert context_counts[row["id"]]["Q2"] == {"neutral": 2, "conflict": 2}
        assert direction_counts[row["id"]]["Q2"] == {"minus": 2, "plus": 2}
        assert all(direction_context_counts[row["id"]][direction] == {"neutral": 1, "conflict": 1}
                   for direction in ("minus", "plus"))
    return {
        "reader_mode": reader_mode, "reader_recipe": READER_RECIPE,
        "reader_direction_note": READER_DIRECTION_NOTE,
        "events_sha256": sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()),
        "events": events, "context_counts": context_counts,
        "reader_direction_counts": direction_counts,
        "reader_direction_context_counts": direction_context_counts, **summary,
    }


def worst_joint_batch(train_rows, tok, reader_schedule=runner.DEFAULT_READER_SCHEDULE):
    assert reader_schedule == runner.DEFAULT_READER_SCHEDULE and len(train_rows) == 256
    candidates = {role: None for role in ("G", "E", "Q")}
    for update in range(1, UPDATES + 1):
        _, j, direction, variant, examples = update_examples(train_rows, update)
        for order, example in enumerate(examples):
            role = "Q" if example["task"] in ("Q1", "Q2") else example["task"]
            if role not in candidates:
                continue
            _, packed = runner.task_length(example, tok, B8_ARM)
            if candidates[role] is None or packed > candidates[role][0]:
                index = (j + len(train_rows) // 2) % len(train_rows) if order == 1 else j
                candidates[role] = (packed, train_rows[index]["id"], direction, variant, example, update)
    assert all(candidates.values())
    batch = [candidates["G"][4], candidates["G"][4], candidates["E"][4], candidates["Q"][4]]
    details = [{
        "task": item[4]["task"], "id": item[1], "direction": item[2], "variant": item[3],
        "candidate_updates_checked": UPDATES, "worst_update": item[5],
        "sequence_kind": "dual_stream" if "crystal" in item[4] else "text",
        "worst_actual_packed_tokens": item[0],
        "actual_sequence_tokens_by_arm": {
            arm: runner.task_length(item[4], tok, arm)[1] for arm in runner.ARMS
        },
    } for item in candidates.values()]
    return batch, details


BASE.update_examples = update_examples
BASE.exposure_receipt = exposure_receipt
BASE._worst_joint_batch = worst_joint_batch
BASE.READER_RECIPE = READER_RECIPE
BASE.READER_DIRECTION_NOTE = READER_DIRECTION_NOTE
_PARENT_SOURCE_HASHES = runner.runner_source_hashes


def runner_source_hashes():
    return {
        "parent": _PARENT_SOURCE_HASHES(),
        "coverage256": {
            name: _sha(ROOT / name)
            for name in ("train.py", "train.slurm", "test_train.py", "prepare_data.py")
        },
    }


runner.runner_source_hashes = runner_source_hashes
runner.update_examples = lambda rows, update, reader_schedule=runner.DEFAULT_READER_SCHEDULE: update_examples(rows, update)
runner._worst_joint_batch = lambda rows, tok, reader_schedule=runner.DEFAULT_READER_SCHEDULE: worst_joint_batch(rows, tok, reader_schedule)
_ORIGINAL_WRITE_JSON = runner.write_json


def _training_json(path, value):
    if Path(path).name in ("status.json", "gpu_preflight.json", "checkpoint_receipt.json"):
        value.update(_reader_metadata(READER_MODE, DATA_HASHES, CURRENT_SOURCE_HASHES))
    _ORIGINAL_WRITE_JSON(path, value)


runner.write_json = _training_json


def _reader_metadata(reader_mode, data_hashes, source_hashes):
    return {
        "training_stage": TRAINING_STAGE,
        "python_version": sys.version.split()[0],
        "terminal_goal_achieved": False,
        "reader_mode": reader_mode,
        "reader_recipe": READER_RECIPE,
        "reader_direction_note": READER_DIRECTION_NOTE,
        "data_hashes": data_hashes,
        "source_hashes": source_hashes,
    }


def run_training(out=ROOT / "training_fixed2048_b8_001"):
    global CURRENT_SOURCE_HASHES
    manifest, train_rows, heldout_ids = load_data()
    source_pin = {
        "data_check": DATA_CHECK,
        "manifest_sha256": DATA_HASHES["manifest_sha256"],
        "records_sha256": DATA_HASHES["records_sha256"],
        "representation": manifest["representation"],
        "cohort": {"train_count": 256, "heldout_count": 16, "train_ids": manifest["cohort"]["train_ids"],
                   "heldout_ids": heldout_ids},
        "space_group_policy": manifest["space_group_policy"],
        "selection_rule": manifest["selection_protocol"]["rule"],
        "prefix_preservation": manifest["prefix_preservation"],
        "preparation_source_sha256": PREPARE_DATA_SHA256,
        "blockrunner_code_sha256": manifest["provenance"]["blockrunner_code_sha256"],
        "tokenizer_files": manifest["provenance"]["tokenizer_files"],
        "model": runner.MODEL, "revision": runner.REVISION,
        "seed": SEED, "updates": UPDATES, "epochs_per_train_case": 8,
        "arm": B8_ARM, "prediction_offset": 0, "block_size": 8,
        "clean_crystal_ar_weight": runner.CLEAN_CRYSTAL_AR_WEIGHT,
        "K": K, "MAXLEN": MAXLEN,
        "initial_fingerprint": runner.EXPECTED_INITIAL_FINGERPRINT,
    }
    source_hashes = runner.runner_source_hashes()
    CURRENT_SOURCE_HASHES = source_hashes
    exposure = exposure_receipt(train_rows)
    tok = runner.tokenizer()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    _ORIGINAL_WRITE_JSON(out / "training_exposure.json", {
        "training_stage": TRAINING_STAGE, "updates": UPDATES, "epochs_per_train_case": 8,
        "effective_batch": 4, "microbatch": 1, "accumulation": runner.ACCUMULATION,
        "task_counts_per_update": {"G": 2, "E": 1, "reader": "Q1/Q2 alternate by epoch"},
        **exposure,
    })
    _ORIGINAL_WRITE_JSON(out / "run_metadata.json", {
        **_reader_metadata(READER_MODE, DATA_HASHES, source_hashes),
        "seed": SEED, "updates": UPDATES, "epochs_per_train_case": 8,
        "train_ids": [row["id"] for row in train_rows], "heldout_ids_from_manifest_only": heldout_ids,
        "arm": B8_ARM, "prediction_offset": 0, "block_size": 8,
        "evaluation": {"train256": "not_run", "heldout16": "not_run", "AR_crystal_inference": "not_run"},
    })
    preflight_initial, preflight = runner.gpu_preflight(
        train_rows, tok, out, reader_schedule=runner.DEFAULT_READER_SCHEDULE,
    )
    runner.assert_runner_sources(source_hashes)
    model, tok, receipt = runner.train_one(
        B8_ARM, train_rows, tok, DATA_DIR, manifest, DATA_HASHES, source_pin,
        exposure, source_hashes, preflight_initial, out,
        reader_schedule=runner.DEFAULT_READER_SCHEDULE,
    )
    assert receipt["status"] == "trained" and receipt["completed_updates"] == UPDATES
    assert receipt["checkpoint_reload_verified"] is True
    assert receipt["initial_fingerprint"] == runner.EXPECTED_INITIAL_FINGERPRINT
    del model
    runner.torch.cuda.empty_cache()
    assert load_data()[0]["cohort"]["train_ids"] == manifest["cohort"]["train_ids"]
    runner.assert_runner_sources(source_hashes)
    assert runner.data_source_hashes(DATA_DIR) == DATA_HASHES
    summary_path = out / "summary.json"
    _ORIGINAL_WRITE_JSON(summary_path, {
        "status": "training_only_checkpoint_verified", "stage": TRAINING_STAGE,
        "terminal_goal_achieved": False, "reader_mode": READER_MODE,
        "reader_recipe": READER_RECIPE, "reader_direction_note": READER_DIRECTION_NOTE,
        "model": runner.MODEL, "revision": runner.REVISION, "seed": SEED,
        "updates": UPDATES, "epochs_per_train_case": 8, "effective_batch": 4,
        "arm": B8_ARM, "prediction_offset": 0, "block_size": 8,
        "K": runner.K, "MAXLEN": runner.MAXLEN,
        "clean_crystal_ar_weight": runner.CLEAN_CRYSTAL_AR_WEIGHT,
        "initial_fingerprint": receipt["initial_fingerprint"],
        "final_fingerprint": receipt["final_fingerprint"],
        "checkpoint_reload_verified": receipt["checkpoint_reload_verified"],
        "actual_task_exposure_counts": receipt["actual_exposure_counts"],
        "actual_noise_step_exposure": receipt["actual_noise_step_exposure"],
        "source_pin": source_pin, "gpu_preflight": preflight,
        "evaluation": {"train256": "not_run", "heldout16": "not_run", "AR_crystal_inference": "not_run"},
    })
    return summary_path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training_fixed2048_b8_001")
    args = parser.parse_args(argv)
    if not runner.torch.cuda.is_available():
        raise SystemExit("training requires the requested Slurm A100 allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing to train outside Slurm")
    summary = run_training(args.out)
    print(json.dumps({"status": "training_only_checkpoint_verified", "summary": str(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
