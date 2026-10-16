"""Prepare 32-train/16-heldout relational-CNL data; never trains a model."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
BASE_ROOT = PROJECT / "experiments/interleaved_relational_20261004"
BLOCK_ROOT = PROJECT / "experiments/interleaved_relational_blocks_20261005"
RELATIONAL_ROOT = PROJECT / "experiments/naturecry_relational_20261004"
INLINE_ROOT = PROJECT / "experiments/naturecry_inline_20261003"
ROUNDTRIP_ROOT = PROJECT / "experiments/naturecry_roundtrip_20261001/run_001"
OLD_TOKENIZER_ROOT = PROJECT / "experiments/interleaved_ar_diffusion_20261002/data_002/tokenizer"
DATA_DIR = ROOT / "data_001"

SELECTION_PATH = ROUNDTRIP_ROOT / "selection.json"
BASE_MANIFEST_PATH = BASE_ROOT / "data_pilot_001/manifest.json"
BASE_RECORDS_PATH = BASE_ROOT / "data_pilot_001/records.jsonl"
INLINE_MANIFEST_PATH = INLINE_ROOT / "prepared_final/manifest.json"
RUN100_SUMMARY_PATH = RELATIONAL_ROOT / "run100_001/summary.json"
RUN100_RECORDS_PATH = RELATIONAL_ROOT / "run100_001/records.json"
TRAIN_CSV_PATH = PROJECT / "work/PLaID/data/basic/train.csv"

K, MAXLEN = 2560, 8192
ADDED_IDS = 24
TOTAL_ROWS = 48
TRAIN_ROWS = 32
HELDOUT_ROWS = 16
ORIGINAL24_RECORDS_SHA256 = "ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f"
ORIGINAL24_MANIFEST_SHA256 = "b8c290dd64f210dd469b83cbf41ad321ae4d7c5c40303c70e8cda0e4335cd0c3"
EXPECTED_BLOCK_CODE_HASHES = {
    "prepare.py": "4889b777e79b15c1761698375f9b1f48316e648129cce4f0439a30714fb1a30b",
    "score.py": "3c1760481872726d798dd18ec4fb3557d6940e669e4c4d818a07da276d9c477a",
    "run.py": "1be7999e5d297f3a3129b513be704eef7ec8ad6c5e7f8e1d62a6befe2a542f62",
    "model.py": "0b27ee6f952ebbdaea599b67da1e7b1061e3dd4b43bbe730bf2291eed0811349",
}
BASE_PROVENANCE_PATHS = {
    "selection": SELECTION_PATH,
    "prepared_manifest": INLINE_MANIFEST_PATH,
    "prepared_source_hashes": INLINE_ROOT / "prepared_final/source_hashes.json",
    "relational_codec": RELATIONAL_ROOT / "relational_codec.py",
    "geometry": INLINE_ROOT / "geometry.py",
    "source_facts_helper": INLINE_ROOT / "codec.py",
    "public_wyckoff_fixture": PROJECT / "experiments/explicit_full_mp20_20260923/codec/fixtures/wyckoff_list.csv",
    "preflight": BASE_ROOT / "preflight_001/relational_preflight.json",
    "adapter_prepare": BASE_ROOT / "prepare.py",
    "adapter_score": BASE_ROOT / "score.py",
}
RUNTIME_PACKAGES = ("numpy", "scipy", "pymatgen", "spglib", "torch", "transformers")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_paths(case_id):
    return {
        "selection": SELECTION_PATH,
        "plan": INLINE_ROOT / "prepared_final/preflight_only/oracle_plans" / f"{case_id}.json",
        "facts": INLINE_ROOT / "prepared_final/cases" / case_id / "encoder_facts.json",
        "text": RELATIONAL_ROOT / "run100_001/texts" / f"{case_id}.txt",
        "cif": ROUNDTRIP_ROOT / "inputs" / f"{case_id}.cif",
    }


def verify_frozen_sources():
    require(sha256(BASE_MANIFEST_PATH.read_bytes()) == ORIGINAL24_MANIFEST_SHA256,
            "original 24-case manifest differs from the reviewed frozen pin")
    require(sha256(BASE_RECORDS_PATH.read_bytes()) == ORIGINAL24_RECORDS_SHA256,
            "original 24-case records differ from the reviewed frozen pin")
    selection_bytes = SELECTION_PATH.read_bytes()
    selection = json.loads(selection_bytes)
    base_manifest = read_json(BASE_MANIFEST_PATH)
    base_records = BASE_RECORDS_PATH.read_bytes()
    base_rows = [json.loads(line) for line in base_records.splitlines()]
    inline_manifest = read_json(INLINE_MANIFEST_PATH)
    run100_summary = read_json(RUN100_SUMMARY_PATH)
    run100_records = read_json(RUN100_RECORDS_PATH)

    ids = [case["id"] for case in selection["cases"]]
    require(selection.get("selected") == 100 and len(ids) == 100 and len(set(ids)) == 100,
            "source selection must contain 100 unique ordered IDs")
    require(sha256(selection_bytes) == base_manifest["cohort"]["selection_sha256"],
            "selection hash differs from the frozen 24-case source manifest")
    require(sha256(base_records) == base_manifest["records_sha256"] and len(base_rows) == 24,
            "original 24-record prefix does not match its frozen manifest")
    require(base_records.endswith(b"\n"), "original JSONL prefix must end at a record boundary")
    require([row["id"] for row in base_rows] == ids[:24],
            "original records do not match the first 24 selected source IDs")
    require(base_manifest["cohort"]["train_ids"] == ids[:8]
            and base_manifest["cohort"]["heldout_ids"] == ids[8:24],
            "original train8/heldout16 split differs from the frozen selection")
    require(inline_manifest["case_ids_in_original_order"] == ids,
            "prepared facts are not in the frozen 100-case selection order")
    require(run100_summary["completed"] == run100_summary["passed"] == 100
            and run100_summary["failed"] == 0 and run100_summary["all_contracts_passed"],
            "relational_cnl_v1 source cohort is not the verified 100/100 run")
    require(run100_summary["prepared_runtime_hashes"]["selection_json"] == sha256(selection_bytes)
            and run100_summary["prepared_runtime_hashes"]["selected_dataset"] == selection["dataset_sha256"],
            "run100 source selection or dataset hash differs")
    require(len(run100_records) == 100 and len({item["case"] for item in run100_records}) == 100,
            "run100 records must contain exactly 100 unique source cases")
    require([item["case"] for item in run100_records] == ids,
            "run100 result records are not in the frozen source-selection order")

    expected_files = base_manifest["provenance"]["files"]
    for name, path in BASE_PROVENANCE_PATHS.items():
        actual = sha256(path.read_bytes())
        require(actual == expected_files[name], f"frozen source pin mismatch for {name}: {path}")
    for name, expected in base_manifest["provenance"]["tokenizer_files"].items():
        actual = sha256((OLD_TOKENIZER_ROOT / name).read_bytes())
        require(actual == expected, f"pinned tokenizer file changed: {name}")
    runtime_versions = base_manifest["provenance"]["runtime_versions"]
    require(sha256(TRAIN_CSV_PATH.read_bytes()) == selection["dataset_sha256"],
            "MP20 train CSV differs from the frozen source-selection dataset")

    inline_cases = {case["id"]: case for case in inline_manifest["cases"]}
    run_records = {item["case"]: item for item in run100_records}
    selected_cases = {case["id"]: case for case in selection["cases"]}
    source_hashes = {}
    for case_id in ids:
        paths = source_paths(case_id)
        inline_case = inline_cases[case_id]
        run_record = run_records[case_id]
        selection_case = selected_cases[case_id]
        require(run_record["status"] == "pass" and all(run_record["contracts"].values()),
                f"stored relational codec result is not a full pass for {case_id}")
        values = {
            "selection_sha256": sha256(selection_bytes),
            "source_plan_sha256": sha256(paths["plan"].read_bytes()),
            "source_facts_sha256": sha256(paths["facts"].read_bytes()),
            "source_text_sha256": sha256(paths["text"].read_bytes()),
            "source_cif_sha256": sha256(paths["cif"].read_bytes()),
        }
        require(values["source_plan_sha256"] == inline_case["oracle_plan_sha256"],
                f"prepared source plan hash mismatch for {case_id}")
        require(values["source_facts_sha256"] == inline_case["facts_sha256"]
                == run_record["facts_sha256"], f"prepared facts hash mismatch for {case_id}")
        require(values["source_text_sha256"] == run_record["text_sha256"],
                f"relational CNL text hash mismatch for {case_id}")
        require(values["source_cif_sha256"] == selection_case["input_cif_sha256"]
                == run_record["input_sha256"], f"source CIF hash mismatch for {case_id}")
        source_hashes[case_id] = values
    for row in base_rows:
        pinned = source_hashes[row["id"]]
        for name, digest in pinned.items():
            require(row["source_hashes"][name] == digest,
                    f"original record source hash differs from frozen inputs for {row['id']} ({name})")

    return {
        "selection": selection,
        "selection_bytes": selection_bytes,
        "ids": ids,
        "base_manifest": base_manifest,
        "base_records": base_records,
        "base_rows": base_rows,
        "inline_manifest": inline_manifest,
        "run100_summary": run100_summary,
        "run100_records": run_records,
        "source_hashes": source_hashes,
        "runtime_versions": runtime_versions,
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_adapter():
    for name, expected in EXPECTED_BLOCK_CODE_HASHES.items():
        actual = sha256((BLOCK_ROOT / name).read_bytes())
        require(actual == expected, f"blockrunner source pin mismatch for {name}")
    if str(BLOCK_ROOT) not in sys.path:
        sys.path.insert(0, str(BLOCK_ROOT))
    module = importlib.import_module("prepare")
    require(Path(module.__file__).resolve() == (BLOCK_ROOT / "prepare.py").resolve(),
            "imported adapter is not the pinned blockrunner prepare.py")
    require(module.K == K and module.MAXLEN == MAXLEN,
            "blockrunner K/MAXLEN changed from the scaled-data protocol")
    return module


def require_geometry(result, label):
    coordinate_error, lattice_error = result
    require(lattice_error < 1e-8 and coordinate_error < 1e-6,
            f"{label} geometry failed: lattice={lattice_error}, coordinates={coordinate_error}")
    return {
        "lattice_component_error_A": lattice_error,
        "coordinate_error_A": coordinate_error,
    }


def prepare_row(case_id, selection_bytes, source_hashes, adapter, score):
    paths = adapter._source_paths(case_id)
    source_plan_bytes = paths["plan"].read_bytes()
    source_facts_bytes = paths["facts"].read_bytes()
    source_text_bytes = paths["text"].read_bytes()
    source_cif_bytes = paths["cif"].read_bytes()
    require(sha256(source_plan_bytes) == source_hashes[case_id]["source_plan_sha256"],
            f"source plan changed before row preparation for {case_id}")
    require(sha256(source_facts_bytes) == source_hashes[case_id]["source_facts_sha256"],
            f"source facts changed before row preparation for {case_id}")
    require(sha256(source_text_bytes) == source_hashes[case_id]["source_text_sha256"],
            f"source text changed before row preparation for {case_id}")
    require(sha256(source_cif_bytes) == source_hashes[case_id]["source_cif_sha256"],
            f"source CIF changed before row preparation for {case_id}")

    plan = json.loads(source_plan_bytes)
    facts = json.loads(source_facts_bytes)
    require(facts["input_atoms"] == plan["input_atom_count"]
            and facts["space_group_number"] == plan["space_group_number"],
            f"prepared source facts disagree with plan for {case_id}")
    source_text = source_text_bytes.decode("utf-8")
    require(adapter._render_plan(plan) == source_text,
            f"stored relational text differs from facts-rendered source plan for {case_id}")
    require(adapter.relational_codec.normalize(source_text, table=adapter.TABLE) == source_text,
            f"stored source text is not canonical for {case_id}")
    require(adapter.relational_codec.decode(source_text, table=adapter.TABLE) == plan,
            f"stored source text does not decode to its exact plan for {case_id}")

    source_cif = adapter.CifParser(str(paths["cif"]), frac_tolerance=0).parse_structures(primitive=False)[0]
    source_exec = adapter._structure(plan)
    source_geometry = require_geometry(
        adapter.relational_codec.geometry.compare_structures(source_cif, source_exec),
        f"{case_id} source",
    )
    source_audit = score.audit(source_text, plan)
    require(source_audit["valid"], f"{case_id} source physical audit failed: {source_audit.get('reasons')}")

    target_plans, target_texts, target_audits, edit_geometry, target_hashes = {}, {}, {}, {}, {}
    lengths = {"c1": len(adapter.TOK.encode(source_text, add_special_tokens=False)) + 1}
    require(lengths["c1"] <= K, f"{case_id} source requires {lengths['c1']} tokens; K={K}")
    for direction, factor in (("plus", "1.05"), ("minus", "0.95")):
        target_plan = adapter._scaled_plan(plan, factor)
        require(all(target_plan[key] == value for key, value in plan.items() if key != "lattice")
                and target_plan["lattice"][3:] == plan["lattice"][3:],
                f"{case_id} {direction} target changed a non-length field")
        target_text = adapter._render_plan(target_plan)
        decoded = adapter.relational_codec.decode(target_text, table=adapter.TABLE)
        require(decoded == target_plan, f"{case_id} {direction} target plan did not round-trip exactly")
        require(adapter.relational_codec.normalize(target_text, table=adapter.TABLE) == target_text,
                f"{case_id} {direction} text is not canonical")
        target_structure = adapter._structure(decoded)
        geometry = require_geometry(
            adapter.relational_codec.geometry.compare_structures(
                adapter._scaled_reference(source_cif, factor), target_structure
            ),
            f"{case_id} {direction} edit",
        )
        target_audit = score.audit(target_text, target_plan)
        require(target_audit["valid"],
                f"{case_id} {direction} physical audit failed: {target_audit.get('reasons')}")
        target_plans[direction] = target_plan
        target_texts[direction] = target_text
        target_audits[direction] = adapter._audit_record(target_audit)
        edit_geometry[direction] = geometry
        lengths[direction] = len(adapter.TOK.encode(target_text, add_special_tokens=False)) + 1
        require(lengths[direction] <= K,
                f"{case_id} {direction} requires {lengths[direction]} tokens; K={K}, no truncation")
        target_hashes[direction] = {
            "plan_sha256": sha256(json.dumps(target_plan, sort_keys=True, separators=(",", ":")).encode()),
            "text_sha256": sha256(target_text.encode()),
        }

    return {
        "id": case_id,
        "split": "train",
        "c1": source_text,
        "plus": target_texts["plus"],
        "minus": target_texts["minus"],
        "condition": adapter._condition(plan),
        "plans": {"source": plan, "plus": target_plans["plus"], "minus": target_plans["minus"]},
        "source_hashes": source_hashes[case_id],
        "source_files": {key: path.relative_to(PROJECT).as_posix() for key, path in paths.items()},
        "target_hashes": target_hashes,
        "token_lengths": lengths,
        "baseline_audit": {
            "source": adapter._audit_record(source_audit),
            "plus": target_audits["plus"],
            "minus": target_audits["minus"],
            "source_cif_geometry": source_geometry,
            "edit_geometry": edit_geometry,
        },
    }


def audit_packing(rows, adapter):
    arms = ("ar", "diffusion_full", "diffusion_b32", "diffusion_b64")
    variants = {"train": (0, 1), "heldout": (0,)}
    directions = ("minus", "plus")
    stats = {arm: {split: {task: [] for task in ("G", "Q1", "E", "Q2")}
                   for split in ("train", "heldout")} for arm in arms}
    dual_concat = []
    dual_actual = []
    encode = lambda text: len(adapter.TOK.encode(text, add_special_tokens=False))

    for row in rows:
        split = row["split"]
        for direction in directions:
            for variant in variants[split]:
                for name, example in zip(("G", "Q1", "E", "Q2"),
                                         adapter.tasks(row, split, direction, variant)):
                    if "crystal" in example:
                        body = encode(example["crystal"]) + 1
                        require(body <= K, f"{row['id']} {split}/{name}/{direction}: body={body} > K={K}")
                        separate_prompt = encode(example["prefix"]) + encode(example["lead"])
                        combined_prompt = encode(example["prefix"] + example["lead"])
                        separate_dual = separate_prompt + 2 * K
                        combined_dual = combined_prompt + 2 * K
                        dual_actual.append(separate_dual)
                        dual_concat.append(combined_dual)
                        require(separate_dual <= MAXLEN and combined_dual <= MAXLEN,
                                f"{row['id']} {split}/{name}/{direction}/v{variant}: dual stream exceeds MAXLEN")
                        packed = {
                            "ar": separate_prompt + K,
                            "diffusion_full": separate_dual,
                            "diffusion_b32": separate_dual,
                            "diffusion_b64": separate_dual,
                        }
                    else:
                        packed_len = encode(example["prefix"]) + encode(example["target"])
                        packed = {arm: packed_len for arm in arms}
                    for arm, length in packed.items():
                        require(length <= MAXLEN,
                                f"{row['id']} {split}/{name}/{direction}/v{variant}/{arm}: {length} > MAXLEN={MAXLEN}")
                        stats[arm][split][name].append(length)

    return {
        "directions": list(directions),
        "variants_by_split": {key: list(value) for key, value in variants.items()},
        "max_dual_stream_prefix_plus_2K": max(dual_concat),
        "max_actual_dual_pack_prefix_streams_plus_2K": max(dual_actual),
        "arms": {
            arm: {
                split: {
                    task: {"n": len(values), "min": min(values), "max": max(values)}
                    for task, values in tasks.items()
                }
                for split, tasks in splits.items()
            }
            for arm, splits in stats.items()
        },
    }


def write_dataset():
    require(not DATA_DIR.exists(), f"refusing to overwrite existing data: {DATA_DIR}")
    frozen = verify_frozen_sources()
    actual_versions = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    require(actual_versions == frozen["runtime_versions"],
            "runtime versions differ from the frozen relational preparation")
    require(sys.version.split()[0] == frozen["base_manifest"]["provenance"]["python_version"],
            "Python version differs from the frozen relational preparation")
    adapter = load_adapter()
    import score

    require(adapter.MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
            and adapter.REVISION == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "current blockrunner model pin changed")
    appended_ids = frozen["ids"][24:48]
    heldout_ids = frozen["ids"][8:24]
    added_rows = [
        prepare_row(case_id, frozen["selection_bytes"], frozen["source_hashes"], adapter, score)
        for case_id in appended_ids
    ]
    rows = frozen["base_rows"] + added_rows
    train_ids = [row["id"] for row in rows if row["split"] == "train"]
    held_ids = [row["id"] for row in rows if row["split"] == "heldout"]
    require(len(rows) == TOTAL_ROWS and len(train_ids) == TRAIN_ROWS and held_ids == heldout_ids,
            "scaled cohort must be exactly train32 plus the unchanged heldout16")
    require(set(train_ids).isdisjoint(held_ids), "train and heldout source IDs overlap")
    held_hashes = {row["source_hashes"]["source_cif_sha256"] for row in rows if row["split"] == "heldout"}
    train_hashes = {row["source_hashes"]["source_cif_sha256"] for row in rows if row["split"] == "train"}
    require(train_hashes.isdisjoint(held_hashes), "train and heldout source CIF hashes overlap")
    require([row["id"] for row in rows[:24]] == frozen["ids"][:24],
            "original 24 rows are not the unchanged selection prefix")

    pack_audit = audit_packing(rows, adapter)
    records_bytes = frozen["base_records"] + b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        for row in added_rows
    )
    manifest = {
        "status": "data_prepared_not_trained",
        "representation": "relational_cnl_v1",
        "model_for_later_training": {"name": adapter.MODEL, "revision": adapter.REVISION},
        "cohort": {
            "selection_path": SELECTION_PATH.relative_to(PROJECT).as_posix(),
            "selection_sha256": sha256(frozen["selection_bytes"]),
            "selected_total": 100,
            "included": TOTAL_ROWS,
            "train_ids": train_ids,
            "heldout_ids": heldout_ids,
            "appended_train_ids": appended_ids,
            "untouched_ids": frozen["ids"][48:],
            "train_heldout_id_overlap": [],
            "train_heldout_source_cif_hash_overlap": [],
        },
        "prefix_preservation": {
            "source_records_path": BASE_RECORDS_PATH.relative_to(PROJECT).as_posix(),
            "source_records_sha256": frozen["base_manifest"]["records_sha256"],
            "output_prefix_sha256": sha256(records_bytes[:len(frozen["base_records"])]),
            "prefix_byte_identical": records_bytes.startswith(frozen["base_records"]),
            "prefix_row_count": 24,
        },
        "records": {"path": "records.jsonl", "count": TOTAL_ROWS, "sha256": sha256(records_bytes)},
        "source_pins": {
            "selection_dataset_sha256": frozen["selection"]["dataset_sha256"],
            "run100_summary_sha256": sha256(RUN100_SUMMARY_PATH.read_bytes()),
            "run100_records_sha256": sha256(RUN100_RECORDS_PATH.read_bytes()),
            "original24_manifest_sha256": sha256(BASE_MANIFEST_PATH.read_bytes()),
            "original24_records_sha256": frozen["base_manifest"]["records_sha256"],
            "original_codec_files": frozen["base_manifest"]["provenance"]["files"],
            "source_hashes_for_appended_ids": {row["id"]: row["source_hashes"] for row in added_rows},
            "blockrunner_code_sha256": EXPECTED_BLOCK_CODE_HASHES,
            "runtime_versions": frozen["runtime_versions"],
            "codec": "relational_cnl_v1",
            "source_codec_run": "100/100 exact-plan, geometry, canonical-text, second-decode contracts passed; no LLM calls",
        },
        "capacity": {"K": K, "MAXLEN": MAXLEN},
        "dual_stream_pack_audit": pack_audit,
        "added_reference_audit": {
            "source_cases": ADDED_IDS,
            "plus_cases": ADDED_IDS,
            "minus_cases": ADDED_IDS,
            "failed_cases": [],
            "filtered_or_replaced": False,
            "truncated": False,
        },
        "later_training_budget": {
            "epochs_per_train_case": 64,
            "updates_for_train32": 2048,
            "scheduled": False,
            "status": "not_run",
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=False)
    (DATA_DIR / "records.jsonl").write_bytes(records_bytes)
    temp_manifest = DATA_DIR / "manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp_manifest.replace(DATA_DIR / "manifest.json")
    return manifest


def check_dataset():
    frozen = verify_frozen_sources()
    records_path = DATA_DIR / "records.jsonl"
    manifest_path = DATA_DIR / "manifest.json"
    records_bytes = records_path.read_bytes()
    manifest = read_json(manifest_path)
    rows = [json.loads(line) for line in records_bytes.splitlines()]
    ids = frozen["ids"]
    row_ids = [row["id"] for row in rows]
    require(manifest["status"] == "data_prepared_not_trained"
            and manifest["capacity"] == {"K": K, "MAXLEN": MAXLEN},
            "output status or capacity differs from this data-only stage")
    require(records_bytes.startswith(frozen["base_records"]),
            "output does not begin with the original 24 record bytes")
    require(sha256(records_bytes[:len(frozen["base_records"])]) == frozen["base_manifest"]["records_sha256"],
            "original 24-record prefix hash changed")
    require(row_ids == ids[:48] and len(rows) == TOTAL_ROWS,
            "output IDs must be original prefix 24 plus selected IDs [24:48]")
    require([row["split"] for row in rows] == ["train"] * 8 + ["heldout"] * 16 + ["train"] * 24,
            "output split must preserve heldout16 and append exactly 24 train cases")
    train_ids = row_ids[:8] + row_ids[24:]
    heldout_ids = row_ids[8:24]
    require(manifest["cohort"]["train_ids"] == train_ids
            and manifest["cohort"]["heldout_ids"] == heldout_ids == ids[8:24],
            "manifest train/heldout IDs differ from the frozen split")
    require(manifest["cohort"]["appended_train_ids"] == ids[24:48]
            and manifest["cohort"]["untouched_ids"] == ids[48:],
            "manifest must append only selection IDs [24:48]")
    train_hashes = {row["source_hashes"]["source_cif_sha256"] for row in rows if row["split"] == "train"}
    held_hashes = {row["source_hashes"]["source_cif_sha256"] for row in rows if row["split"] == "heldout"}
    require(set(train_ids).isdisjoint(heldout_ids) and train_hashes.isdisjoint(held_hashes),
            "train and heldout sources overlap")
    selection_hashes = {case["id"]: case["input_cif_sha256"] for case in frozen["selection"]["cases"]}
    for row in rows:
        require(row["source_hashes"]["source_cif_sha256"] == selection_hashes[row["id"]],
                f"source CIF selection hash differs for {row['id']}")
        if row["id"] in ids[24:48]:
            require(row["source_hashes"] == frozen["source_hashes"][row["id"]],
                    f"appended source hashes differ from the verified source pool for {row['id']}")
    require(manifest["records"]["count"] == TOTAL_ROWS
            and manifest["records"]["sha256"] == sha256(records_bytes),
            "records file does not match its output manifest")
    require(manifest["prefix_preservation"]["prefix_byte_identical"]
            and manifest["prefix_preservation"]["source_records_sha256"]
            == manifest["prefix_preservation"]["output_prefix_sha256"],
            "manifest prefix-preservation proof failed")
    return {"status": "check_passed_data_only", "rows": len(rows), "train": len(train_ids),
            "heldout": len(heldout_ids), "training_run": "not_run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="build the frozen data_001 output")
    mode.add_argument("--check", action="store_true", help="check prefix, split, source separation, and file hashes")
    args = parser.parse_args()
    result = write_dataset() if args.prepare else check_dataset()
    if args.prepare:
        result = {"status": result["status"], "records": result["records"],
                  "train": len(result["cohort"]["train_ids"]),
                  "heldout": len(result["cohort"]["heldout_ids"]),
                  "training_run": "not_run"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
