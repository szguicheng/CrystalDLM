"""Fixed 8-seen / 16-heldout relational CNL four-arm AR/block-diffusion pilot."""
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import time
import warnings

import numpy as np
import torch

from model import Hybrid
import prepare
from prepare import (
    DATA_DIR,
    END,
    EOS,
    K,
    MASK,
    MAXLEN,
    MODEL,
    PAD,
    REVISION,
    ROOT,
    START,
    STEPS,
    TOKENIZER_PATH,
    edit_factor,
    load_records,
    tasks,
    tokenizer,
)
from score import audit, grade_answer, grade_edit

UPDATES = 512
SEED = 17
ACCUMULATION = 4
READER_SCHEDULES = ("alternating_q1_q2",)
DEFAULT_READER_SCHEDULE = "alternating_q1_q2"
ARMS = ("ar", "diffusion_full", "diffusion_b32", "diffusion_b64")
MODE_OFFSETS = {arm: 0 for arm in ARMS}
CORE_MODES = {"ar": "ar", **{arm: "hybrid" for arm in ARMS[1:]}}
BLOCK_SIZES = {"ar": K, "diffusion_full": K, "diffusion_b32": 32, "diffusion_b64": 64}
EXPECTED_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECTED_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DATA_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
OUTPUT_KEYS = (
    "G", "Q_reference", "E_reference", "closed_loop", "flow_complete",
    "Q_swap", "Q_no_crystal",
)
PROJECT = Path(__file__).resolve().parents[2]
RELATIONAL = PROJECT / "experiments/naturecry_relational_20261004"
OLD_EXPERIMENT = PROJECT / "experiments/interleaved_relational_20261004"
PIN_RUN = RELATIONAL / "run100_001"
PIN_SUMMARY = PIN_RUN / "summary.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runner_source_hashes():
    sources = {path.name: sha(path.read_bytes()) for path in ROOT.glob("*.py")}
    sources.update({path.name: sha(path.read_bytes()) for path in ROOT.glob("*.slurm")})
    return dict(sorted(sources.items()))


def assert_runner_sources(expected):
    current = runner_source_hashes()
    assert current == expected, "runner or adapter source changed during this pilot"


def data_source_hashes(data):
    return {
        "manifest_sha256": sha((data / "manifest.json").read_bytes()),
        "records_sha256": sha((data / "records.jsonl").read_bytes()),
    }


def verify_codec_pin(data_manifest):
    """Verify the active adapter inputs against the frozen run100 source snapshot."""
    summary = json.loads(PIN_SUMMARY.read_text())
    expected = summary["runtime_hashes"]
    paths = {
        "relational_codec.py": RELATIONAL / "relational_codec.py",
        "run_roundtrip.py": RELATIONAL / "run_roundtrip.py",
        "geometry.py": PROJECT / "experiments/naturecry_inline_20261003/geometry.py",
        "base_cohort.py": PROJECT / "experiments/naturecry_llm_100_20261002/cohort.py",
        "oldcodec.py": PROJECT / "experiments/naturecry_inline_20261003/codec.py",
        "api_runner.py": PROJECT / "experiments/naturecry_inline_20261003/api_runner.py",
        "prepare.py": PROJECT / "experiments/naturecry_inline_20261003/prepare.py",
        "public_wyckoff_baseline.py": PROJECT / "experiments/explicit_full_mp20_20260923/codec/wyckoff_baseline.py",
        "public_affine_parser.py": PROJECT / "experiments/explicit_full_mp20_20260923/codec/audit_wyckoff_families.py",
        "cif_parser.py": PROJECT / "experiments/explicit_full_mp20_20260923/codec/test_program.py",
        "public_wyckoff_list.csv": PROJECT / "experiments/explicit_full_mp20_20260923/codec/fixtures/wyckoff_list.csv",
    }
    assert summary["runtime_hashes_stable"] is True, "run100 source pin was not stable"
    assert summary["runtime_hashes_after"] == expected, "run100 sources changed during its frozen run"
    assert summary["all_contracts_passed"] is True and summary["passed"] == 100
    for name, path in paths.items():
        digest = sha(path.read_bytes())
        assert digest == expected[name], f"pinned codec source changed: {name}"
        snapshot = PIN_RUN / "sources" / name
        assert sha(snapshot.read_bytes()) == expected[name], f"run100 source snapshot mismatch: {name}"
    assert data_manifest["cohort"]["selection_sha256"] == summary["selection_sha256"], "pilot cohort selection differs from run100"
    provenance = data_manifest["provenance"]
    data_files = provenance["files"]
    assert data_files["selection"] == summary["selection_sha256"]
    assert data_files["relational_codec"] == expected["relational_codec.py"]
    assert data_files["geometry"] == expected["geometry.py"]
    assert data_files["source_facts_helper"] == expected["oldcodec.py"]
    assert data_files["public_wyckoff_fixture"] == expected["public_wyckoff_list.csv"]
    assert data_files["adapter_prepare"] == sha((OLD_EXPERIMENT / "prepare.py").read_bytes())
    old_score_hash = sha((OLD_EXPERIMENT / "score.py").read_bytes())
    assert data_files["adapter_score"] == old_score_hash
    assert sha((ROOT / "score.py").read_bytes()) == old_score_hash, "new scorer differs from the frozen scorer"
    assert provenance["public_table_sha256"] == prepare.relational_codec.geometry.FROZEN_TABLE_SHA256
    assert provenance["model"] == DATA_MODEL and provenance["revision"] == DATA_REVISION
    assert MODEL == EXPECTED_MODEL and REVISION == EXPECTED_REVISION
    assert provenance["tokenizer_path"] == Path(TOKENIZER_PATH).relative_to(PROJECT).as_posix()
    tokenizer_hashes = {
        path.name: sha(path.read_bytes())
        for path in Path(TOKENIZER_PATH).iterdir() if path.is_file()
    }
    assert tokenizer_hashes == provenance["tokenizer_files"], "native tokenizer files differ from frozen data manifest"
    return {
        "summary_sha256": sha(PIN_SUMMARY.read_bytes()),
        "selection_sha256": summary["selection_sha256"],
        "adapter_prepare_sha256": data_files["adapter_prepare"],
        "adapter_score_sha256": data_files["adapter_score"],
        "data_model": provenance["model"],
        "data_revision": provenance["revision"],
        "data_tokenizer_path": provenance["tokenizer_path"],
        "data_tokenizer_files": provenance["tokenizer_files"],
        "runtime_hashes": {name: expected[name] for name in paths},
        "runtime_hashes_stable": True,
    }


def read_data(data):
    manifest_path = data / "manifest.json"
    rows_path = data / "records.jsonl"
    manifest = json.loads(manifest_path.read_text())
    rows = load_records(path=rows_path)
    selected_ids = [
        item["id"]
        for item in json.loads((PROJECT / manifest["cohort"]["selection_path"]).read_text())["cases"]
    ]
    row_ids = [row["id"] for row in rows]
    assert len(selected_ids) == 100
    assert row_ids == selected_ids[:24], "pilot rows are not the frozen selection prefix"
    assert manifest["cohort"]["excluded_from_this_pilot"] == selected_ids[24:100]
    assert manifest["capacity"] == {"K": K, "MAXLEN": MAXLEN}
    assert manifest["diffusion"]["STEPS"] == STEPS
    assert manifest["cohort"]["selected_total"] == 100 and manifest["cohort"]["included"] == 24
    assert len(manifest["cohort"]["excluded_from_this_pilot"]) == 76
    assert len(rows) == 24, f"pilot must load only 24 records, got {len(rows)}"
    assert [row["split"] for row in rows] == ["train"] * 8 + ["heldout"] * 16
    assert len({row["id"] for row in rows}) == 24
    train, heldout = rows[:8], rows[8:]
    assert all(row["split"] == "train" for row in train)
    assert all(row["split"] == "heldout" for row in heldout)
    assert manifest["cohort"]["train_ids"] == row_ids[:8]
    assert manifest["cohort"]["heldout_ids"] == row_ids[8:24]
    return manifest, rows, train, heldout


def task_length(example, tok, arm):
    assert arm in ARMS
    encode = lambda value: len(tok.encode(value, add_special_tokens=False))
    if "crystal" in example:
        body = encode(example["crystal"]) + 1
        streams = 1 if arm == "ar" else 2
        packed = encode(example["prefix"]) + encode(example["lead"]) + streams * K
        return body, packed
    return None, encode(example["prefix"]) + encode(example["target"])


def pack_audit(rows, tok):
    """Audit the actual one-stream AR and dual-stream diffusion sequence lengths."""
    stats = {
        arm: {split: {name: [] for name in ("G", "Q1", "E", "Q2")}
              for split in ("train", "heldout")}
        for arm in ARMS
    }
    variants = {"train": (0, 1), "heldout": (0,)}
    for row in rows:
        split = row["split"]
        for direction in ("minus", "plus"):
            for variant in variants[split]:
                for name, example in zip(("G", "Q1", "E", "Q2"), tasks(row, split, direction, variant)):
                    for arm in ARMS:
                        body, packed = task_length(example, tok, arm)
                        if body is not None:
                            assert body <= K, f"{row['id']} {split}/{name}/{direction}: body {body} > K={K}"
                        assert packed <= MAXLEN, (
                            f"{row['id']} {split}/{name}/{direction}/{arm}: "
                            f"actual packed length {packed} > MAXLEN={MAXLEN}"
                        )
                        stats[arm][split][name].append(packed)
    return {
        arm: {
            split: {
                name: {
                    "n": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": float(np.mean(values)),
                    "p95": float(np.percentile(values, 95)),
                }
                for name, values in roles.items()
            }
            for split, roles in splits.items()
        }
        for arm, splits in stats.items()
    }


def update_examples(rows, update, reader_schedule=DEFAULT_READER_SCHEDULE):
    """Keep G_j, G_(j+4), E_j fixed; select the reader target by schedule."""
    assert len(rows) == 8 and 1 <= update <= UPDATES
    assert reader_schedule in READER_SCHEDULES
    epoch, j = divmod(update - 1, 8)
    direction = "minus" if (epoch + j) % 2 == 0 else "plus"
    variant = 0
    g1 = tasks(rows[j], "train", direction, variant)[0]
    g2 = tasks(rows[(j + 4) % 8], "train", direction, variant)[0]
    task_set = tasks(rows[j], "train", direction, variant)
    if epoch % 2 == 0:
        reader = task_set[1]
    else:
        reader_direction = "minus" if (epoch // 2 + j) % 2 == 0 else "plus"
        reader = dict(tasks(rows[j], "train", reader_direction, variant)[3],
                      reader_direction=reader_direction)
    examples = [g1, g2, task_set[2], reader]
    expected_reader = "Q1" if epoch % 2 == 0 else "Q2"
    assert [example["task"] for example in examples] == ["G", "G", "E", expected_reader]
    return epoch, j, direction, variant, examples


def exposure_receipt(rows, reader_schedule=DEFAULT_READER_SCHEDULE):
    assert reader_schedule in READER_SCHEDULES
    events = []
    counts = {row["id"]: Counter() for row in rows}
    factors = {row["id"]: Counter() for row in rows}
    reader_factors = {row["id"]: Counter() for row in rows}
    for update in range(1, UPDATES + 1):
        epoch, j, direction, variant, examples = update_examples(rows, update, reader_schedule)
        for order, example in enumerate(examples):
            role = example["task"]
            material_id = rows[j]["id"] if role != "G" or order == 0 else rows[(j + 4) % 8]["id"]
            counts[material_id][role] += 1
            if role == "E":
                factors[material_id][direction] += 1
            event = {
                "update": update,
                "epoch": epoch,
                "j": j,
                "order": order,
                "id": material_id,
                "task": role,
                "direction": direction,
                "variant": variant,
            }
            if role == "Q2":
                reader_factors[material_id][example["reader_direction"]] += 1
                event["reader_direction"] = example["reader_direction"]
            events.append(event)
    for material_id in counts:
        assert counts[material_id] == Counter({"G": 128, "E": 64, "Q1": 32, "Q2": 32})
        assert reader_factors[material_id] == Counter({"minus": 16, "plus": 16})
        assert factors[material_id] == Counter({"minus": 32, "plus": 32})
    assert len(events) == UPDATES * 4
    result = {
        "reader_schedule": reader_schedule,
        "events_sha256": sha(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()),
        "events": events,
        "per_material": {
            material_id: {
                "tasks": dict(sorted(counts[material_id].items())),
                "E_factors": dict(sorted(factors[material_id].items())),
            }
            for material_id in counts
        },
        "total": dict(sorted(Counter(event["task"] for event in events).items())),
    }
    if reader_schedule == "alternating_q1_q2":
        for material_id in result["per_material"]:
            result["per_material"][material_id]["Q2_factors"] = dict(
                sorted(reader_factors[material_id].items())
            )
    return result


def summarize_exposure(events):
    by_material = {}
    by_factor = {}
    by_reader_factor = {}
    total = Counter()
    for event in events:
        material_id, role = event["id"], event["task"]
        by_material.setdefault(material_id, Counter())[role] += 1
        total[role] += 1
        if role == "E":
            by_factor.setdefault(material_id, Counter())[event["direction"]] += 1
        elif role == "Q2":
            by_reader_factor.setdefault(material_id, Counter())[event["reader_direction"]] += 1
    result = {
        "per_material": {
            material_id: {
                "tasks": dict(sorted(counts.items())),
                "E_factors": dict(sorted(by_factor.get(material_id, Counter()).items())),
            }
            for material_id, counts in sorted(by_material.items())
        },
        "total": dict(sorted(total.items())),
    }
    for material_id, factors in by_reader_factor.items():
        result["per_material"][material_id]["Q2_factors"] = dict(sorted(factors.items()))
    return result


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def attempt(function, model=None):
    started = time.monotonic()
    before = model.forward_count if model is not None else 0
    try:
        with warnings.catch_warnings(), autocast():
            warnings.simplefilter("ignore")
            result = {"ok": True, "value": function()}
    except Exception as error:
        result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        raw_ids = getattr(error, "raw_ids", None)
        if raw_ids is not None:
            result["raw_token_ids"] = list(raw_ids)
    result["elapsed_seconds"] = time.monotonic() - started
    if model is not None:
        result["forward_calls"] = model.forward_count - before
    return result


def generate(model, tok, mode, prefix, rng):
    lead = model.text(prefix, tok, mode, trigger=True)
    crystal, tokens = model.crystal(prefix + lead, tok, mode, rng)
    return {"lead": lead, "crystal": crystal, "tokens": tokens}


def _suffix_after(prefix, marker, label):
    before, found, after = prefix.rpartition(marker)
    if not found:
        raise ValueError(f"cannot locate {label} in task prefix")
    return after


def _read_question_suffix(q_prefix, c1):
    suffix = _suffix_after(q_prefix, c1 + END, "reference crystal")
    if not suffix.startswith("\nUser: ") or not suffix.endswith("\nAssistant:"):
        raise ValueError("unexpected Q1 prompt boundary")
    return suffix


def _edit_instruction_suffix(e_prefix, q_target):
    suffix = _suffix_after(e_prefix, q_target, "reference Q1 answer")
    if not suffix.startswith("\nUser: ") or not suffix.endswith("\nAssistant:"):
        raise ValueError("unexpected E prompt boundary")
    return suffix


def _q2_suffix(q2_prefix, edited):
    suffix = _suffix_after(q2_prefix, edited + END, "reference edited crystal")
    if not suffix.startswith("\nUser: ") or not suffix.endswith("\nAssistant:"):
        raise ValueError("unexpected Q2 prompt boundary")
    return suffix


def _plan_signature(checked):
    plan = checked["decoded_plan"]
    return (
        tuple(sorted(plan["composition"].items())),
        plan["space_group_number"],
        plan["input_atom_count"],
        len(plan["orbits"]),
        tuple(plan["lattice"][:3]),
    )


def _swap_partner(row, candidates, checks):
    signature = _plan_signature(checks[row["id"]])
    for other in candidates:
        if other["id"] != row["id"]:
            changed = sum(a != b for a, b in zip(signature, _plan_signature(checks[other["id"]])))
            if changed >= 2:
                return other
    return None


def _read_text_question(suffix):
    return suffix[len("\nUser: "):-len("\nAssistant:")]


def check_runner_boundaries(rows):
    for row in rows:
        for direction in ("minus", "plus"):
            _, q_task, e_task, q2_task = tasks(row, row["split"], direction, 0)
            _read_question_suffix(q_task["prefix"], row["c1"])
            _edit_instruction_suffix(e_task["prefix"], q_task["target"])
            _q2_suffix(q2_task["prefix"], row[direction])
            assert grade_edit(row[direction], row["c1"], edit_factor(direction))["success"], (
                f"{row['id']} {direction}: adapter E target disagrees with runner factor contract"
            )


def evaluate(model, tok, mode, rows, out):
    """Run every queue once; all 24 records remain in their split denominator."""
    out.mkdir(parents=True, exist_ok=False)
    model_mode = CORE_MODES[mode]
    source_checks = {row["id"]: audit(row["c1"]) for row in rows}
    assert all(check["valid"] for check in source_checks.values()), "frozen CNL failed its own decode/execution audit"
    counters = []
    with (out / "evaluation.jsonl").open("w") as log:
        for index, row in enumerate(rows):
            split = row["split"]
            direction = "minus" if int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % 2 == 0 else "plus"
            variant = 0
            g_task, q_task, e_task, q2_task = tasks(row, split, direction, variant)
            rng_seed = int(hashlib.sha256((row["id"] + "20261004").encode()).hexdigest()[:8], 16)
            rng = torch.Generator(device="cuda").manual_seed(rng_seed)
            before = model.forward_count
            record = {
                "id": row["id"], "split": split, "direction": direction,
                **{key: False for key in OUTPUT_KEYS},
                "failure_classes": [],
            }

            g = attempt(lambda: generate(model, tok, model_mode, g_task["prefix"], rng), model)
            record["generation"] = g
            g_audit = None
            if g["ok"]:
                c1 = g["value"]["crystal"]
                g_audit = audit(c1, requested=row)
                record["generation_audit"] = {key: value for key, value in g_audit.items() if key != "structure"}
                record["G"] = bool(g_audit["valid"])
                if not record["G"]:
                    record["failure_classes"].append({"stage": "G", "kind": "generation_audit", "reasons": g_audit.get("reasons", [])})
            else:
                record["failure_classes"].append({"stage": "G", "kind": "generation_error", "error": g["error"]})

            q = attempt(lambda: model.text(q_task["prefix"], tok, model_mode), model)
            record["reference_read"] = q
            if q["ok"]:
                record["reference_read_grade"] = grade_answer(q["value"], source_checks[row["id"]])
                record["Q_reference"] = bool(record["reference_read_grade"]["success"])
                if not record["Q_reference"]:
                    record["failure_classes"].append({"stage": "Q_reference", "kind": "answer_mismatch"})
            else:
                record["failure_classes"].append({"stage": "Q_reference", "kind": "read_error", "error": q["error"]})

            e = attempt(lambda: generate(model, tok, model_mode, e_task["prefix"], rng), model)
            record["reference_edit"] = e
            if e["ok"]:
                record["reference_edit_grade"] = grade_edit(
                    e["value"]["crystal"], row["c1"], edit_factor(direction)
                )
                record["E_reference"] = bool(record["reference_edit_grade"]["success"])
                if not record["E_reference"]:
                    record["failure_classes"].append({"stage": "E_reference", "kind": "edit_mismatch"})
            else:
                record["failure_classes"].append({"stage": "E_reference", "kind": "edit_error", "error": e["error"]})

            if record["G"] and g_audit is not None:
                c1 = g["value"]["crystal"]
                q_suffix = _read_question_suffix(q_task["prefix"], row["c1"])
                actual_history = g_task["prefix"] + g["value"]["lead"] + c1 + END
                actual_q_prefix = actual_history + q_suffix
                actual_q = attempt(lambda: model.text(actual_q_prefix, tok, model_mode), model)
                record["actual_read"] = actual_q
                if actual_q["ok"]:
                    actual_q_grade = grade_answer(actual_q["value"], g_audit)
                    record["actual_read_grade"] = actual_q_grade
                    if not actual_q_grade["success"]:
                        record["failure_classes"].append({"stage": "Q1_actual", "kind": "answer_mismatch"})
                    edit_suffix = _edit_instruction_suffix(e_task["prefix"], q_task["target"])
                    actual_edit_prefix = actual_q_prefix + actual_q["value"] + edit_suffix
                    actual_e = attempt(lambda: generate(model, tok, model_mode, actual_edit_prefix, rng), model)
                    record["actual_edit"] = actual_e
                    if actual_e["ok"]:
                        c2 = actual_e["value"]["crystal"]
                        actual_e_grade = grade_edit(c2, c1, edit_factor(direction))
                        record["actual_edit_grade"] = actual_e_grade
                        if not actual_e_grade["success"]:
                            record["failure_classes"].append({"stage": "E_actual", "kind": "edit_mismatch"})
                        c2_audit = audit(c2)
                        record["actual_edit_audit"] = {key: value for key, value in c2_audit.items() if key != "structure"}
                        if c2_audit["valid"]:
                            q2_suffix = _q2_suffix(q2_task["prefix"], row[direction])
                            actual_q2_prefix = actual_edit_prefix + actual_e["value"]["lead"] + c2 + END + q2_suffix
                            actual_q2 = attempt(lambda: model.text(actual_q2_prefix, tok, model_mode), model)
                            record["actual_second_read"] = actual_q2
                            if actual_q2["ok"]:
                                record["flow_complete"] = True
                                actual_q2_grade = grade_answer(actual_q2["value"], c2_audit)
                                record["actual_second_read_grade"] = actual_q2_grade
                                if not actual_q2_grade["success"]:
                                    record["failure_classes"].append({"stage": "Q2_actual", "kind": "answer_mismatch"})
                                record["closed_loop"] = bool(
                                    actual_q_grade["success"] and actual_e_grade["success"] and actual_q2_grade["success"]
                                )
                            else:
                                record["failure_classes"].append({"stage": "Q2_actual", "kind": "read_error", "error": actual_q2["error"]})
                        else:
                            record["failure_classes"].append({"stage": "E_actual", "kind": "generated_edit_audit", "reasons": c2_audit.get("reasons", [])})
                    else:
                        record["failure_classes"].append({"stage": "E_actual", "kind": "edit_error", "error": actual_e["error"]})
                else:
                    record["failure_classes"].append({"stage": "Q1_actual", "kind": "read_error", "error": actual_q["error"]})
            elif not record["G"]:
                record["failure_classes"].append({"stage": "closed_loop", "kind": "skipped_invalid_generated_c1"})

            candidates = [item for item in rows if item["split"] == split]
            other = _swap_partner(row, candidates, source_checks)
            if other is None:
                record["failure_classes"].append({"stage": "Q_swap", "kind": "no_valid_swap_partner"})
            else:
                read_question = _read_text_question(_read_question_suffix(q_task["prefix"], row["c1"]))
                swap_question = "Read the current replacement crystal, ignoring the earlier design conditions. " + read_question
                prior = g_task["prefix"] + " I generated a crystal.\n"
                replacement = prior + START + other["c1"] + END + "\nUser: " + swap_question + "\nAssistant:"
                swap = attempt(lambda: model.text(replacement, tok, model_mode), model)
                record["swap_id"] = other["id"]
                record["swap_read"] = swap
                if swap["ok"]:
                    record["swap_read_grade"] = grade_answer(swap["value"], source_checks[other["id"]])
                    record["Q_swap"] = bool(record["swap_read_grade"]["success"])
                    if not record["Q_swap"]:
                        record["failure_classes"].append({"stage": "Q_swap", "kind": "answer_mismatch"})
                else:
                    record["failure_classes"].append({"stage": "Q_swap", "kind": "read_error", "error": swap["error"]})
                no_crystal = prior + "\nUser: " + swap_question + "\nAssistant:"
                absent = attempt(lambda: model.text(no_crystal, tok, model_mode), model)
                record["no_crystal_read"] = absent
                if absent["ok"]:
                    # Diagnostic only: exact match without the structure is not an abstention score.
                    record["no_crystal_fact_match"] = grade_answer(absent["value"], source_checks[other["id"]])
                    record["Q_no_crystal"] = bool(record["no_crystal_fact_match"]["success"])
                else:
                    record["failure_classes"].append({"stage": "Q_no_crystal", "kind": "read_error", "error": absent["error"]})

            record["forward_calls"] = model.forward_count - before
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            counters.append(record)
    counts = {key: sum(bool(row[key]) for row in counters) for key in OUTPUT_KEYS}
    summary = {
        "arm": mode,
        "core_mode": model_mode,
        "block_size": BLOCK_SIZES[mode],
        "model": MODEL,
        "revision": REVISION,
        "n": len(rows),
        "split": rows[0]["split"] if rows else None,
        "counts": counts,
        "rates": {key: value / len(rows) for key, value in counts.items()},
        "attempts_retried": 0,
        "failure_classes": dict(Counter(
            failure["stage"] + ":" + failure["kind"]
            for row in counters for failure in row["failure_classes"]
        )),
        "forward_calls": sum(row["forward_calls"] for row in counters),
        "attempt_seconds": sum(
            attempt_value["elapsed_seconds"]
            for row in counters
            for key, attempt_value in row.items()
            if isinstance(attempt_value, dict) and "elapsed_seconds" in attempt_value
        ),
        "no_crystal_semantics": "Q_no_crystal is exact target-fact match without a structure; raw response is retained and this is not an abstention/success score.",
    }
    write_json(out / "evaluation_summary.json", summary)
    return summary


def train_one(
    mode, train_rows, tok, data, data_manifest, data_hashes, source_pin,
    exposure, runner_hashes, expected_initial, out,
    reader_schedule=DEFAULT_READER_SCHEDULE,
):
    assert reader_schedule in READER_SCHEDULES
    mode_dir = out / mode
    mode_dir.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), "Training must run on an allocated GPU compute node"
    assert os.environ.get("SLURM_JOB_ID"), "Training must run inside a Slurm allocation"
    assert_runner_sources(runner_hashes)
    assert data_source_hashes(data) == data_hashes, "frozen pilot records changed during this run"
    seed_all(SEED)
    offset = MODE_OFFSETS[mode]
    block_size = BLOCK_SIZES[mode]
    model = Hybrid.pretrained(tok, prediction_offset=offset, block_size=block_size)
    model.base.gradient_checkpointing_enable()
    model = model.cuda().train()
    assert model.base.is_gradient_checkpointing
    assert next(model.parameters()).dtype == torch.float32, "base parameters must remain FP32"
    assert not torch.count_nonzero(model.time.weight).item(), "time embedding initialization must remain zero"
    initial = fingerprint(model)
    if expected_initial is not None:
        assert initial == expected_initial, f"{mode} initial fingerprint differs from AR control"
    rng = torch.Generator().manual_seed(SEED + 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    receipt = {
        "mode": CORE_MODES[mode], "arm": mode, "status": "training", "updates": UPDATES, "seed": SEED,
        "reader_schedule": reader_schedule,
        "prediction_offset": offset, "block_size": block_size, "initial_fingerprint": initial,
        "model": MODEL, "model_revision": REVISION,
        "tokenizer_path": str(TOKENIZER_PATH),
        "base_parameter_dtype": "float32", "autocast": "bfloat16",
        "gradient_checkpointing": True,
        "optimizer": "AdamW", "learning_rate": 2e-5, "weight_decay": 0.01,
        "scheduler": "linear warmup 40 updates then cosine to zero",
        "gradient_clip_norm": 1.0, "effective_batch": 4, "microbatch": 1,
        "gradient_accumulation": ACCUMULATION, "K": K, "MAXLEN": MAXLEN, "STEPS": STEPS,
        "task_exposure_sha256": sha(json.dumps(exposure["events"], sort_keys=True, separators=(",", ":")).encode()),
        "scheduled_exposure_counts": {"per_material": exposure["per_material"], "total": exposure["total"]},
        "dataset_manifest_sha256": data_hashes["manifest_sha256"],
        "dataset_records_sha256": data_hashes["records_sha256"],
        "data_manifest_sha256": sha(json.dumps(data_manifest, sort_keys=True, separators=(",", ":")).encode()),
        "source_pin": source_pin,
        "code_hashes": runner_hashes,
        "host": socket.gethostname(), "slurm_job": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(), "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy", "pyxtal", "pymatgen", "spglib")
        },
    }
    write_json(mode_dir / "status.json", receipt)
    curve = []
    actual_events = []
    training_forward_start = model.forward_count
    torch.cuda.reset_peak_memory_stats()
    begun = time.monotonic()
    with (mode_dir / "curve.jsonl").open("w") as log:
        for update in range(1, UPDATES + 1):
            torch.cuda.synchronize()
            started = time.monotonic()
            forward_before_update = model.forward_count
            multiplier = update / 40 if update <= 40 else 0.5 * (
                1 + math.cos(math.pi * (update - 40) / (UPDATES - 40))
            )
            for group in optimizer.param_groups:
                group["lr"] = 2e-5 * multiplier
            optimizer.zero_grad(set_to_none=True)
            epoch, j, direction, variant, examples = update_examples(train_rows, update, reader_schedule)
            values = []
            task_loss = Counter()
            for example in examples:
                with autocast():
                    loss, stats = model.loss([example], tok, CORE_MODES[mode], rng)
                assert torch.isfinite(loss), "nonfinite objective"
                (loss / ACCUMULATION).backward()
                values.append(float(loss.detach()))
                task_loss[example["task"]] += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), "nonfinite gradient norm"
            optimizer.step()
            actual_events.extend(exposure["events"][(update - 1) * 4:update * 4])
            torch.cuda.synchronize()
            value = {
                "update": update, "epoch": epoch, "j": j, "direction": direction,
                "variant": variant, "loss": sum(values) / len(values),
                "task_loss": dict(task_loss), "seconds": time.monotonic() - started,
                "forward_calls": model.forward_count - forward_before_update,
                "lr": optimizer.param_groups[0]["lr"],
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
            }
            curve.append(value)
            log.write(json.dumps(value, allow_nan=False) + "\n")
            log.flush()
            if update % 10 == 0 or update == UPDATES:
                assert_runner_sources(runner_hashes)
                assert data_source_hashes(data) == data_hashes, "frozen pilot records changed during training"
                receipt.update(
                    completed_updates=update,
                    elapsed_seconds=time.monotonic() - begun,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                    training_forward_calls=model.forward_count - training_forward_start,
                    actual_exposure_counts=summarize_exposure(actual_events),
                )
                write_json(mode_dir / "status.json", receipt)
                print(json.dumps({"mode": mode, **value}), flush=True)
            if update % 200 == 0 or update == UPDATES:
                final_update = update == UPDATES
                if final_update:
                    final_fingerprint = fingerprint(model)
                    receipt["final_fingerprint"] = final_fingerprint
                    write_json(mode_dir / "status.json", receipt)
                temporary = mode_dir / "checkpoint.tmp"
                torch.save({
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "update": update, "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "corruption_rng": rng.get_state(),
                    "config": receipt,
                }, temporary)
                temporary.replace(mode_dir / "checkpoint.pt")
                if final_update:
                    checkpoint_path = mode_dir / "checkpoint.pt"
                    checkpoint_sha = sha_file(checkpoint_path)
                    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    restored_config = model.restore_checkpoint(
                        saved, expected_mode=CORE_MODES[mode], expected_offset=offset,
                        expected_block_size=block_size,
                    )
                    reloaded_fingerprint = fingerprint(model)
                    assert restored_config["final_fingerprint"] == final_fingerprint
                    assert reloaded_fingerprint == final_fingerprint, f"{mode} checkpoint reload fingerprint mismatch"
                    del saved
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    checkpoint_receipt = {
                        "status": "verified",
                        "path": str(checkpoint_path),
                        "checkpoint_update": update,
                        "checkpoint_sha256": checkpoint_sha,
                        "reader_schedule": reader_schedule,
                        "task_exposure_sha256": receipt["task_exposure_sha256"],
                        "mode": CORE_MODES[mode],
                        "arm": mode,
                        "prediction_offset": offset,
                        "block_size": block_size,
                        "final_fingerprint": final_fingerprint,
                        "reload_verified": True,
                        "reload_fingerprint": reloaded_fingerprint,
                        "strict_state_load": True,
                    }
                    write_json(mode_dir / "checkpoint_receipt.json", checkpoint_receipt)
                    receipt.update(
                        checkpoint_sha256=checkpoint_sha,
                        checkpoint_reload_verified=True,
                        checkpoint_reload_fingerprint=reloaded_fingerprint,
                        checkpoint_reload_mode=restored_config["mode"],
                        checkpoint_reload_offset=offset,
                        checkpoint_reload_block_size=restored_config["block_size"],
                    )
                    write_json(mode_dir / "status.json", receipt)
    receipt.update(
        status="trained", completed_updates=UPDATES,
        training_seconds=time.monotonic() - begun,
        training_forward_calls=model.forward_count - training_forward_start,
        final_fingerprint=receipt["final_fingerprint"],
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
        actual_exposure_counts=summarize_exposure(actual_events),
        actual_exposure_sha256=sha(json.dumps(actual_events, sort_keys=True, separators=(",", ":")).encode()),
    )
    assert_runner_sources(runner_hashes)
    assert data_source_hashes(data) == data_hashes, "frozen pilot records changed during training"
    assert receipt["actual_exposure_counts"]["per_material"] == exposure["per_material"]
    assert receipt["actual_exposure_counts"]["total"] == exposure["total"]
    write_json(mode_dir / "status.json", receipt)
    return model.eval(), tok, receipt


def aggregate_teacher_forced(examples):
    result = {}
    for role in ("body", "end", "pad"):
        count = sum(item["metrics"][role]["count"] for item in examples)
        correct = sum(item["metrics"][role]["correct"] for item in examples)
        nll_sum = sum(item["metrics"][role]["nll_sum"] for item in examples)
        result[role] = {
            "count": count, "correct": correct, "nll_sum": nll_sum,
            "accuracy": correct / count if count else None,
            "mean_nll": nll_sum / count if count else None,
        }
    return result


def teacher_forced_seen8(model, tok, rows, arm, out):
    """Measure fixed-noise G accuracy/NLL and normalized CE contributions."""
    roles = ("body", "end", "pad")
    contributions = ("crystal", "crystal_body", "crystal_end", "crystal_pad")
    attempts, grouped = [], {step: [] for step in (32, 16, 1)}
    for row in rows:
        example = tasks(row, "train", "minus", 0)[0]
        for step in grouped:
            seeded = dict(example, noise_step=step)
            seed = int(hashlib.sha256(f"{row['id']}:{step}".encode()).hexdigest()[:8], 16)
            rng = torch.Generator().manual_seed(seed)
            started, before = time.monotonic(), model.forward_count
            try:
                with torch.inference_mode(), autocast():
                    loss, stats = model.loss([seeded], tok, "hybrid", rng)
                torch.cuda.synchronize()
                values = {key: stats[key] for key in contributions}
                assert all(value is not None and math.isfinite(value) for value in values.values())
                assert abs(sum(values[key] for key in contributions[1:]) - values["crystal"]) < 1e-4
                class_metrics = {
                    role: {
                        "count": int(stats["crystal_metrics"][role]["count"]),
                        "correct": int(stats["crystal_metrics"][role]["correct"]),
                        "nll_sum": float(stats["crystal_metrics"][role]["nll_sum"]),
                    }
                    for role in roles
                }
                attempt = {
                    "ok": True, "id": row["id"], "noise_step": step,
                    "mask_seed": seed, "loss": float(loss),
                    "normalized_ce_contributions": values,
                    "crystal_metrics": class_metrics,
                }
                grouped[step].append({"contributions": values, "metrics": class_metrics})
            except Exception as error:
                attempt = {
                    "ok": False, "id": row["id"], "noise_step": step,
                    "error": f"{type(error).__name__}: {error}",
                }
            attempt.update(
                elapsed_seconds=time.monotonic() - started,
                forward_calls=model.forward_count - before,
            )
            attempts.append(attempt)
    summary = {
        "arm": arm, "block_size": BLOCK_SIZES[arm], "split": "seen8",
        "target": "G", "noise_steps": {},
        "forward_calls": sum(item["forward_calls"] for item in attempts),
        "attempt_seconds": sum(item["elapsed_seconds"] for item in attempts),
        "attempts": attempts,
        "metric_semantics": "class accuracy and raw NLL are aggregated by masked target count; CE contributions use the whole-target normalizer",
        "logits_persisted": False,
    }
    for step, values in grouped.items():
        if not values:
            summary["noise_steps"][str(step)] = {"n": 0}
            continue
        summary["noise_steps"][str(step)] = {
            "n": len(values),
            "mean_normalized_ce_contributions": {
                key: float(np.mean([item["contributions"][key] for item in values]))
                for key in contributions
            },
            "crystal_metrics": aggregate_teacher_forced(values),
        }
    write_json(out / "teacher_forced_seen8.json", summary)
    return summary


def _worst_joint_batch(train_rows, tok, reader_schedule=DEFAULT_READER_SCHEDULE):
    assert reader_schedule in READER_SCHEDULES
    candidates = {role: None for role in ("G", "E", "Q")}
    for row in train_rows:
        for direction in ("minus", "plus"):
            for variant in (0, 1):
                for example in tasks(row, "train", direction, variant):
                    role = example["task"]
                    role_key = "Q" if role in ("Q1", "Q2") else role
                    if role_key not in candidates:
                        continue
                    body, packed = task_length(example, tok, "diffusion_full")
                    if candidates[role_key] is None or packed > candidates[role_key][0]:
                        candidates[role_key] = (packed, row["id"], direction, variant, example)
    assert all(candidates.values()), f"preflight lacks a target for {reader_schedule}"
    selected = [candidates["G"][4], candidates["G"][4], candidates["E"][4], candidates["Q"][4]]
    details = [
        {
            "task": entry[4]["task"], "id": entry[1],
            "direction": entry[2], "variant": entry[3],
            "sequence_kind": "dual_stream" if "crystal" in entry[4] else "text",
            "worst_actual_packed_tokens": entry[0],
            "actual_sequence_tokens_by_arm": {
                arm: task_length(entry[4], tok, arm)[1] for arm in ARMS
            },
        }
        for role, entry in candidates.items()
    ]
    return selected, details


def preflight_examples(examples):
    return [dict(example, noise_step=STEPS) if "crystal" in example else example for example in examples]


def gpu_preflight(train_rows, tok, out, reader_schedule=DEFAULT_READER_SCHEDULE):
    """One disposable optimizer step per arm; it never contributes to pilot updates."""
    assert reader_schedule in READER_SCHEDULES
    device = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if "A100" not in device or total_gib < 79:
        report = {
            "status": "failed",
            "device": device,
            "device_total_GiB": total_gib,
            "error": f"requires one A100 80GB, found {device} ({total_gib:.1f} GiB)",
            "optimizer_step_counted_in_pilot": False,
        }
        write_json(out / "gpu_preflight.json", report)
        raise RuntimeError(report["error"])
    batch, examples = _worst_joint_batch(train_rows, tok, reader_schedule)
    batch = preflight_examples(batch)
    report = {
        "status": "running",
        "device": device,
        "device_total_GiB": total_gib,
        "reserved_limit_GiB": 70.0,
        "optimizer_step_counted_in_pilot": False,
        "effective_batch": 4,
        "microbatch": 1,
        "accumulation": ACCUMULATION,
        "reader_schedule": reader_schedule,
        "preflight_noise_step": STEPS,
        "worst_train_examples": examples,
        "worst_actual_dual_stream_tokens": max(
            item["actual_sequence_tokens_by_arm"]["diffusion_full"]
            for item in examples if item["sequence_kind"] == "dual_stream"
        ),
        "arms": {},
    }
    write_json(out / "gpu_preflight.json", report)
    common_initial = None
    try:
        for arm in ARMS:
            seed_all(SEED)
            block_size = BLOCK_SIZES[arm]
            model = Hybrid.pretrained(
                tok, prediction_offset=MODE_OFFSETS[arm], block_size=block_size
            )
            model.base.gradient_checkpointing_enable()
            model = model.cuda().train()
            assert model.base.is_gradient_checkpointing
            assert next(model.parameters()).dtype == torch.float32
            assert not torch.count_nonzero(model.time.weight).item()
            initial = fingerprint(model)
            if common_initial is None:
                common_initial = initial
            assert initial == common_initial, f"{arm} preflight initial fingerprint differs from AR"
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5 / 40, weight_decay=0.01)
            rng = torch.Generator().manual_seed(SEED + 1000)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            forward_before = model.forward_count
            for example in batch:
                with autocast():
                    loss, _ = model.loss([example], tok, CORE_MODES[arm], rng)
                assert torch.isfinite(loss), f"{arm} preflight loss is nonfinite"
                (loss / ACCUMULATION).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(norm), f"{arm} preflight gradient norm is nonfinite"
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            allocated = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            report["arms"][arm] = {
                "core_mode": CORE_MODES[arm],
                "prediction_offset": MODE_OFFSETS[arm],
                "block_size": block_size,
                "gradient_checkpointing": True,
                "preflight_noise_step": STEPS,
                "initial_fingerprint": initial,
                "initial_parameters": "FP32, zero time embedding",
                "one_effective_batch_optimizer_step": True,
                "forward_calls": model.forward_count - forward_before,
                "actual_sequence_tokens": [task_length(example, tok, arm)[1] for example in batch],
                "max_dual_stream_tokens": max(
                    task_length(example, tok, arm)[1]
                    for example in batch if "crystal" in example
                ),
                "seconds": seconds,
                "peak_allocated_GiB": allocated,
                "peak_reserved_GiB": reserved,
            }
            del loss, norm, model, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            write_json(out / "gpu_preflight.json", report)
            assert reserved <= report["reserved_limit_GiB"], (
                f"{arm} preflight peak reserved {reserved:.2f} GiB exceeds 70 GiB; training stopped"
            )
        report["status"] = "passed"
        report["shared_initial_fingerprint"] = common_initial
        report["initial_fingerprints_equal"] = len({value["initial_fingerprint"] for value in report["arms"].values()}) == 1
        assert report["initial_fingerprints_equal"] and len(report["arms"]) == len(ARMS)
        write_json(out / "gpu_preflight.json", report)
        return common_initial, report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(out / "gpu_preflight.json", report)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        raise


def run(data, out, reader_schedule=DEFAULT_READER_SCHEDULE):
    assert reader_schedule in READER_SCHEDULES
    runner_hashes = runner_source_hashes()
    assert MODEL == EXPECTED_MODEL and REVISION == EXPECTED_REVISION, (
        f"unexpected pretrained pin: {MODEL}@{REVISION}"
    )
    manifest, rows, train_rows, heldout = read_data(data)
    tok = tokenizer()
    packing = pack_audit(rows, tok)
    source_pin = verify_codec_pin(manifest)
    exposure = exposure_receipt(train_rows, reader_schedule)
    data_hashes = data_source_hashes(data)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "pack_audit.json", {
        "K": K, "MAXLEN": MAXLEN, "STEPS": STEPS, "block_sizes": BLOCK_SIZES,
        "row_count": len(rows), "lengths": packing,
    })
    write_json(out / "training_exposure.json", {
        "reader_schedule": reader_schedule,
        "updates": UPDATES, "effective_batch": 4,
        "task_counts_per_update": {"G": 2, "E": 1, "reader": "Q1/Q2 alternating by epoch"},
        **exposure,
    })
    preflight_initial, preflight = gpu_preflight(train_rows, tok, out, reader_schedule)
    assert_runner_sources(runner_hashes)
    assert data_source_hashes(data) == data_hashes, "frozen pilot records changed during preflight"
    results, common_initial = {}, preflight_initial
    for mode in ARMS:
        model, tok, receipt = train_one(
            mode, train_rows, tok, data, manifest, data_hashes, source_pin,
            exposure, runner_hashes, common_initial, out, reader_schedule=reader_schedule
        )
        if common_initial is None:
            common_initial = receipt["initial_fingerprint"]
        receipt["other_arm_initial_fingerprint"] = common_initial
        teacher_forced = None
        if mode != "ar":
            teacher_forced = teacher_forced_seen8(model, tok, train_rows, mode, out / mode)
        seen = evaluate(model, tok, mode, train_rows, out / mode / "seen8")
        held = evaluate(model, tok, mode, heldout, out / mode / "heldout16")
        assert_runner_sources(runner_hashes)
        assert data_source_hashes(data) == data_hashes, "frozen pilot records changed during evaluation"
        receipt.update(status="evaluated", seen8=seen, heldout16=held)
        write_json(out / mode / "status.json", receipt)
        results[mode] = {
            "core_mode": CORE_MODES[mode],
            "block_size": BLOCK_SIZES[mode],
            "prediction_offset": MODE_OFFSETS[mode],
            "gradient_checkpointing": receipt["gradient_checkpointing"],
            "initial_fingerprint": receipt["initial_fingerprint"],
            "final_fingerprint": receipt["final_fingerprint"],
            "training_forward_calls": receipt["training_forward_calls"],
            "teacher_forced_seen8": teacher_forced,
            "seen8": seen, "heldout16": held,
        }
        write_json(out / "summary.json", {
            "status": "partial", "model": MODEL, "revision": REVISION,
            "seed": SEED, "updates": UPDATES, "K": K, "MAXLEN": MAXLEN,
            "arms": ARMS, "block_sizes": BLOCK_SIZES,
            "reader_schedule": reader_schedule,
            "source_pin": source_pin, "gpu_preflight": preflight, "results": results,
        })
        del model
        torch.cuda.empty_cache()
    assert len({results[arm]["initial_fingerprint"] for arm in ARMS}) == 1
    write_json(out / "summary.json", {
        "status": "complete", "model": MODEL, "revision": REVISION,
        "seed": SEED, "updates": UPDATES, "K": K, "MAXLEN": MAXLEN,
        "arms": ARMS, "block_sizes": BLOCK_SIZES,
        "reader_schedule": reader_schedule,
        "initial_fingerprints_equal": True, "source_pin": source_pin,
        "gpu_preflight": preflight, "results": results,
        "length_shift_note": "seen8 and heldout16 are reported separately; heldout CNL blocks may be longer.",
    })
    return out / "summary.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=ROOT / "pilot_001")
    parser.add_argument("--reader-schedule", choices=READER_SCHEDULES, default=DEFAULT_READER_SCHEDULE)
    parser.add_argument("--check-pack", action="store_true", help="Run tokenizer/trajectory length checks without loading a model or using CUDA.")
    args = parser.parse_args(argv)
    manifest, rows, _, _ = read_data(args.data)
    tok = tokenizer()
    packing = pack_audit(rows, tok)
    print(json.dumps({"status": "pack_ok", "K": K, "MAXLEN": MAXLEN, "lengths": packing}, ensure_ascii=False))
    check_runner_boundaries(rows)
    if args.check_pack:
        verify_codec_pin(manifest)
        exposure = exposure_receipt(rows[:8], args.reader_schedule)
        print(json.dumps({
            "status": "offline_checks_ok",
            "reader_schedule": args.reader_schedule,
            "training_exposure_counts": {
                "total": exposure["total"],
                "per_material": exposure["per_material"],
            },
        }, ensure_ascii=False))
        return 0
    assert torch.cuda.is_available(), "Training requires a CUDA allocation"
    assert os.environ.get("SLURM_JOB_ID"), "Refusing to train outside Slurm"
    result = run(args.data, args.out.resolve(), reader_schedule=args.reader_schedule)
    print(json.dumps({"status": "complete", "summary": str(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
