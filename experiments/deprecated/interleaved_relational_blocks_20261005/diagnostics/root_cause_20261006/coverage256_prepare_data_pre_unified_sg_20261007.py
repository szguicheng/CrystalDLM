"""Freeze a metadata-selected 256-train/16-heldout relational-CNL dataset."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
PREPARE_DIR = ROOT / "prepare_001"
DATA_DIR = ROOT / "data_001"
CSV_PATH = PROJECT / "work/PLaID/data/basic/train.csv"
OLD_SELECTION_PATH = PROJECT / "experiments/naturecry_roundtrip_20261001/run_001/selection.json"
OLD_RECORDS_PATH = PROJECT / "experiments/interleaved_relational_scaled32_20261006/data_001/records.jsonl"
OLD_MANIFEST_PATH = OLD_RECORDS_PATH.with_name("manifest.json")
RUN100_SUMMARY_PATH = PROJECT / "experiments/naturecry_relational_20261004/run100_001/summary.json"
RUN100_DIR = RUN100_SUMMARY_PATH.parent
OLD48_SHA256 = "231f336234d15a216dc9cb800d8dc1c3f30cf90985184ccc0d3e5c26333e95c6"
CSV_SHA256 = "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d"
EXPECTED_HELPER_SHA256 = "3323b74d93ac7895c9c1233c018d1c58341743eb1bb23d1df1835eaf5573c101"
EXPECTED_INLINE_SHA256 = "c77ea38b734fe0a1accab663cfa8f1b071ff7c4784513681af0d4bb89a4a6a65"
EXPECTED_BLOCK_CODE_HASHES = {
    "prepare.py": "4889b777e79b15c1761698375f9b1f48316e648129cce4f0439a30714fb1a30b",
    "score.py": "3c1760481872726d798dd18ec4fb3557d6940e669e4c4d818a07da276d9c477a",
}
HELPER_PATH = PROJECT / "experiments/interleaved_relational_scaled32_20261006/prepare_data.py"
INLINE_PATH = PROJECT / "experiments/naturecry_inline_20261003/prepare.py"
BLOCK_ROOT = PROJECT / "experiments/interleaved_relational_blocks_20261005"
CNL_RUNTIME_SOURCE_PATHS = {
    "relational_codec.py": PROJECT / "experiments/naturecry_relational_20261004/relational_codec.py",
    "run_roundtrip.py": PROJECT / "experiments/naturecry_relational_20261004/run_roundtrip.py",
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
RAW_CIF_PREPARATION_DEPENDENCIES = {
    "pilot.py": (PROJECT / "experiments/naturecry_llm_pilot_20261002/pilot.py",
                 "0aad3088782cb38abf80949f439d915cef407075aaad3697cf6fd6fc3f626b81"),
    "benchmark.py": (PROJECT / "experiments/naturecry_roundtrip_20261001/benchmark.py",
                      "603cc3a0923d7171c1507ff2503574cdab13f69f005635b8af417716356b69e9"),
    "naturecry_codec.py": (PROJECT / "experiments/naturecry_roundtrip_20261001/naturecry_codec.py",
                           "5421ceac327c00d8a9d783685be9a02a16d1d16b727414c4a227060715a202ee"),
}
RUNTIME_EXACT_PACKAGES = ("numpy", "scipy", "pymatgen", "spglib", "transformers")
TRAIN_CSV_ROWS = 27136
NEW_TRAIN_ROWS = 224
SEED = 17


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append_jsonl(prefix, rows):
    if not prefix.endswith(b"\n"):
        raise ValueError("The preserved JSONL prefix must end at a record boundary")
    return prefix + b"".join(
        (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def select_candidates(rows, excluded_ids=frozenset(), excluded_cif_hashes=frozenset(),
                      heldout_element_sets=frozenset(), limit=NEW_TRAIN_ROWS, seed=SEED):
    reasons = Counter()
    buckets = defaultdict(list)
    seen_ids, seen_cifs = set(), set()
    excluded_rows = 0
    for row in rows:
        flags = []
        if row["id"] in excluded_ids:
            flags.append("old_selection_id")
        if row["cif_sha256"] in excluded_cif_hashes:
            flags.append("old_selection_cif_sha256")
        if frozenset(row["elements"]) in heldout_element_sets:
            flags.append("heldout_element_set")
        if flags:
            excluded_rows += 1
            reasons.update(flags)
            continue
        if row["id"] in seen_ids:
            reasons["duplicate_candidate_id"] += 1
            excluded_rows += 1
            continue
        seen_ids.add(row["id"])
        if row["cif_sha256"] in seen_cifs:
            reasons["duplicate_candidate_cif_sha256"] += 1
            excluded_rows += 1
            continue
        seen_cifs.add(row["cif_sha256"])
        buckets[int(row["space_group_number"])].append(row)

    for group in buckets.values():
        group.sort(key=lambda row: (sha256((row["id"] + str(seed)).encode()), row["id"],
                                   row["source_row_zero_based"]))
    selected = []
    positions = Counter()
    groups = sorted(buckets)
    while len(selected) < limit:
        advanced = False
        for sg in groups:
            index = positions[sg]
            if index < len(buckets[sg]):
                selected.append(buckets[sg][index])
                positions[sg] += 1
                advanced = True
                if len(selected) == limit:
                    break
        if not advanced:
            break
    if len(selected) != limit:
        raise ValueError(f"Only {len(selected)} eligible unique rows are available; need {limit}")
    return selected, {
        "input_rows": sum(len(group) for group in buckets.values()) + excluded_rows,
        "excluded_rows": excluded_rows,
        "excluded_by_reason": {key: reasons[key] for key in (
            "old_selection_id", "old_selection_cif_sha256", "heldout_element_set",
            "duplicate_candidate_id", "duplicate_candidate_cif_sha256")},
        "eligible_unique_rows": sum(map(len, buckets.values())),
        "selected": len(selected),
        "unselected_eligible_rows": sum(map(len, buckets.values())) - len(selected),
        "eligible_space_group_count": len(buckets),
        "eligible_space_groups": groups,
    }


def _metadata_rows():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected = {"", "material_id", "pretty_formula", "elements", "cif", "spacegroup.number"}
        if not expected.issubset(reader.fieldnames or ()):
            raise ValueError(f"Unexpected MP20 train CSV columns: {reader.fieldnames}")
        source_rows = set()
        for row in reader:
            case_id = row["material_id"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", case_id):
                raise ValueError(f"Invalid MP20 material_id: {case_id!r}")
            source_row = int(row[""])
            if source_row in source_rows:
                raise ValueError(f"Duplicate source row index: {source_row}")
            source_rows.add(source_row)
            elements = ast.literal_eval(row["elements"])
            if not isinstance(elements, list) or not elements or any(
                not isinstance(element, str) or not re.fullmatch(r"[A-Z][a-z]?", element)
                for element in elements
            ):
                raise ValueError(f"Invalid element metadata for {case_id}")
            cif = row["cif"]
            if not cif:
                raise ValueError(f"Empty CIF for {case_id}")
            rows.append({
                "id": case_id,
                "source_row_zero_based": source_row,
                "space_group_number": int(row["spacegroup.number"]),
                "pretty_formula": row["pretty_formula"],
                "elements": sorted(set(elements)),
                "cif_sha256": sha256(cif.encode("utf-8")),
            })
    if len(rows) != TRAIN_CSV_ROWS:
        raise ValueError(f"MP20 train CSV has {len(rows)} rows; expected {TRAIN_CSV_ROWS}")
    return rows


def _old_cohort():
    selection_bytes = OLD_SELECTION_PATH.read_bytes()
    selection = json.loads(selection_bytes)
    records_bytes = OLD_RECORDS_PATH.read_bytes()
    if sha256(records_bytes) != OLD48_SHA256 or not records_bytes.endswith(b"\n"):
        raise ValueError("The frozen old48 records prefix changed")
    rows = [json.loads(line) for line in records_bytes.splitlines()]
    ids = [case["id"] for case in selection["cases"]]
    if selection.get("selected") != 100 or len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("The frozen old100 selection is invalid")
    if len(rows) != 48 or len({row["id"] for row in rows}) != 48:
        raise ValueError("The frozen source records are not exactly 48 unique rows")
    if [row["id"] for row in rows] != ids[:48]:
        raise ValueError("The old48 records are not the old100 selection prefix")
    if sum(row["split"] == "train" for row in rows) != 32 or sum(
        row["split"] == "heldout" for row in rows
    ) != 16:
        raise ValueError("The old48 split is not train32/heldout16")
    heldout_element_sets = {
        frozenset(row["plans"]["source"]["composition"])
        for row in rows if row["split"] == "heldout"
    }
    return {
        "selection": selection,
        "selection_bytes": selection_bytes,
        "records_bytes": records_bytes,
        "rows": rows,
        "ids": set(ids),
        "cif_hashes": {case["input_cif_sha256"] for case in selection["cases"]},
        "heldout_element_sets": heldout_element_sets,
    }


def select_stage():
    if PREPARE_DIR.exists():
        raise FileExistsError(f"Selection is frozen; refusing to replace {PREPARE_DIR}")
    csv_bytes = CSV_PATH.read_bytes()
    if sha256(csv_bytes) != CSV_SHA256:
        raise ValueError("MP20 train CSV does not match its pinned SHA-256")
    old = _old_cohort()
    selected, exclusions = select_candidates(
        _metadata_rows(), old["ids"], old["cif_hashes"], old["heldout_element_sets"]
    )
    selection = {
        "rule": {
            "name": "coverage256_metadata_round_robin_v1",
            "source": "MP20 train CSV public metadata only",
            "exclude": ["all old100 IDs", "all old100 source CIF SHA-256 values",
                        "element-set matches to any heldout plan composition"],
            "bucket": "ascending CSV spacegroup.number",
            "within_bucket": "ascending SHA256(material_id + seed), then material_id and source row",
            "round_robin": "one item per ascending space-group bucket per round",
            "seed": SEED,
            "selection_failures_replaced": False,
        },
        "source_dataset": "work/PLaID/data/basic/train.csv",
        "source_dataset_sha256": CSV_SHA256,
        "source_rows": TRAIN_CSV_ROWS,
        "previous_selection": {
            "path": OLD_SELECTION_PATH.relative_to(PROJECT).as_posix(),
            "sha256": sha256(old["selection_bytes"]),
            "selected": 100,
        },
        "selected": len(selected),
        "exclusions": exclusions,
        "cases": [{
            "id": row["id"],
            "source": "work/PLaID/data/basic/train.csv",
            "source_row_zero_based": row["source_row_zero_based"],
            "kind": "MP20_train",
            "dataset_space_group": row["space_group_number"],
            "dataset_formula": row["pretty_formula"],
            "composition_elements": row["elements"],
            "input_cif_sha256": row["cif_sha256"],
            "input_cif_path": (PREPARE_DIR / "inputs" / f"{row['id']}.cif").relative_to(PROJECT).as_posix(),
        } for row in selected],
    }
    PREPARE_DIR.mkdir(parents=True, exist_ok=False)
    input_dir = PREPARE_DIR / "inputs"
    input_dir.mkdir()
    wanted = {case["id"]: case for case in selection["cases"]}
    with CSV_PATH.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            case = wanted.get(row["material_id"])
            if case is None:
                continue
            cif_bytes = row["cif"].encode("utf-8")
            if sha256(cif_bytes) != case["input_cif_sha256"] or int(row[""]) != case["source_row_zero_based"]:
                raise ValueError(f"Selected CSV row changed for {case['id']}")
            (input_dir / f"{case['id']}.cif").write_bytes(cif_bytes)
            del wanted[case["id"]]
    if wanted:
        raise ValueError(f"Selected CSV rows were not found: {sorted(wanted)}")
    selection_bytes = (json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    (PREPARE_DIR / "selection.json").write_bytes(selection_bytes)
    (PREPARE_DIR / "selection.sha256").write_text(sha256(selection_bytes) + "\n")
    return {
        "status": "selection_frozen",
        "selection_sha256": sha256(selection_bytes),
        "selected": len(selected),
        "exclusions": exclusions,
        "selected_metadata_space_groups": sorted({row["space_group_number"] for row in selected}),
    }


def _load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_adapters(helper):
    if sha256(INLINE_PATH.read_bytes()) != EXPECTED_INLINE_SHA256:
        raise ValueError("Frozen inline prepare.py hash changed")
    for filename, expected in EXPECTED_BLOCK_CODE_HASHES.items():
        if sha256((BLOCK_ROOT / filename).read_bytes()) != expected:
            raise ValueError(f"Frozen blockrunner {filename} hash changed")

    # Preserve inline.prepare_case before registering the block adapter as `prepare`.
    inline = _load_file_module("coverage256_inline_prepare", INLINE_PATH)
    adapter = _load_file_module("prepare", BLOCK_ROOT / "prepare.py")
    sys.path.insert(0, str(BLOCK_ROOT))
    sys.modules.pop("score", None)
    score = _load_file_module("score", BLOCK_ROOT / "score.py")
    return inline, helper, adapter, score


def _next_attempt_dir():
    index = 1
    while True:
        path = PREPARE_DIR / f"prepare_attempt_{index:03d}"
        try:
            path.mkdir(parents=False, exist_ok=False)
            return path
        except FileExistsError:
            index += 1


def _write_case_jsonl(path, results):
    path.write_bytes(b"".join(
        (json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        for result in results
    ))


def _source_paths(cases, attempt_dir):
    cache = attempt_dir / "cache"
    paths = {}
    for case in cases:
        case_id = case["id"]
        item_dir = cache / case_id
        paths[case_id] = {
            "selection": PREPARE_DIR / "selection.json",
            "plan": item_dir / "plan.json",
            "facts": item_dir / "facts.json",
            "text": item_dir / "text.txt",
            "cif": PROJECT / case["input_cif_path"],
        }
    return paths


def _coverage(rows):
    def summarize(subset):
        space_groups = Counter()
        sg_wyckoff = Counter()
        for row in subset:
            plan = row["plans"]["source"]
            sg = int(plan["space_group_number"])
            space_groups[str(sg)] += 1
            for orbit in plan["orbits"]:
                sg_wyckoff[f"{sg}:{orbit['wyckoff']}"] += 1
        return {
            "encoded_space_group_counts": dict(sorted(space_groups.items(), key=lambda pair: int(pair[0]))),
            "encoded_space_group_count": len(space_groups),
            "encoded_sg_wyckoff_pair_counts": dict(sorted(sg_wyckoff.items())),
            "encoded_sg_wyckoff_pair_count": len(sg_wyckoff),
        }

    train = [row for row in rows if row["split"] == "train"]
    heldout = [row for row in rows if row["split"] == "heldout"]
    train_sgs, train_pairs = set(), set()
    for row in train:
        plan = row["plans"]["source"]
        sg = int(plan["space_group_number"])
        train_sgs.add(sg)
        train_pairs.update((sg, orbit["wyckoff"]) for orbit in plan["orbits"])
    unseen_sgs, unseen_pairs = [], {}
    for row in heldout:
        plan = row["plans"]["source"]
        sg = int(plan["space_group_number"])
        missing = sorted({f"{pair_sg}:{wyckoff}" for pair_sg, wyckoff in
                          ((sg, orbit["wyckoff"]) for orbit in plan["orbits"])
                          if (pair_sg, wyckoff) not in train_pairs})
        if sg not in train_sgs:
            unseen_sgs.append(row["id"])
        if missing:
            unseen_pairs[row["id"]] = missing
    return {
        "by_split": {
            "train": summarize(train),
            "heldout": summarize(heldout),
            "all": summarize(rows),
        },
        "heldout_generalization_from_train": {
            "heldout_ids_with_unseen_space_group": unseen_sgs,
            "heldout_ids_with_unseen_sg_wyckoff_pair": list(unseen_pairs),
            "unseen_sg_wyckoff_pairs_by_heldout_id": unseen_pairs,
        },
    }


def _runtime_source_state():
    summary_bytes = RUN100_SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_bytes)
    expected = summary["runtime_hashes"]
    snapshots = summary["source_snapshots"]
    current_hashes, snapshot_hashes, errors = {}, {}, {}
    for name, path in CNL_RUNTIME_SOURCE_PATHS.items():
        try:
            current_hashes[name] = sha256(path.read_bytes())
        except OSError as exc:
            current_hashes[name] = None
            errors[f"current:{name}"] = f"{type(exc).__name__}: {exc}"
        try:
            snapshot_path = RUN100_DIR / snapshots[name]
            snapshot_hashes[name] = sha256(snapshot_path.read_bytes())
        except (OSError, KeyError) as exc:
            snapshot_hashes[name] = None
            errors[f"snapshot:{name}"] = f"{type(exc).__name__}: {exc}"
    return {
        "run100_summary_sha256": sha256(summary_bytes),
        "run100_runtime_hashes_expected": expected,
        "run100_runtime_hashes_after": summary.get("runtime_hashes_after"),
        "run100_runtime_hashes_stable": summary.get("runtime_hashes_stable"),
        "run100_passed": summary.get("passed"),
        "run100_all_contracts_passed": summary.get("all_contracts_passed"),
        "runtime_hashes_current": current_hashes,
        "run100_source_snapshot_hashes": snapshot_hashes,
        "runtime_source_paths": {name: path.relative_to(PROJECT).as_posix()
                                  for name, path in CNL_RUNTIME_SOURCE_PATHS.items()},
        "run100_source_snapshot_paths": snapshots,
        "runtime_hash_errors": errors,
    }


def _runtime_state_matches_pin(state):
    expected = state["run100_runtime_hashes_expected"]
    return (
        state["run100_runtime_hashes_stable"] is True
        and state["run100_runtime_hashes_after"] == expected
        and state["run100_passed"] == 100
        and state["run100_all_contracts_passed"] is True
        and state["runtime_hashes_current"] == expected
        and state["run100_source_snapshot_hashes"] == expected
        and not state["runtime_hash_errors"]
    )


def _runtime_policy(baseline_versions, actual_versions, baseline_python, actual_python):
    package_set_matches = set(baseline_versions) == set(actual_versions) == {
        *RUNTIME_EXACT_PACKAGES, "torch"
    }
    exact_package_matches = {
        package: baseline_versions.get(package) == actual_versions.get(package)
        for package in RUNTIME_EXACT_PACKAGES
    }
    baseline_python_mm = ".".join(baseline_python.split(".")[:2])
    actual_python_mm = ".".join(actual_python.split(".")[:2])
    python_major_minor_match = baseline_python_mm == actual_python_mm
    return {
        "accepted": package_set_matches and all(exact_package_matches.values()) and python_major_minor_match,
        "baseline_package_versions": baseline_versions,
        "actual_package_versions": actual_versions,
        "exact_package_matches": exact_package_matches,
        "baseline_python": baseline_python,
        "actual_python": actual_python,
        "python_major_minor_match": python_major_minor_match,
        "python_patch_difference_allowed": python_major_minor_match,
        "baseline_torch": baseline_versions.get("torch"),
        "actual_torch": actual_versions.get("torch"),
        "torch_difference_allowed": True,
        "arithmetic_scope": "data preparation performs no Torch/model arithmetic",
    }


def _raw_cif_dependency_state():
    expected = {name: digest for name, (_, digest) in RAW_CIF_PREPARATION_DEPENDENCIES.items()}
    current, errors = {}, {}
    for name, (path, _) in RAW_CIF_PREPARATION_DEPENDENCIES.items():
        try:
            current[name] = sha256(path.read_bytes())
        except OSError as exc:
            current[name] = None
            errors[name] = f"{type(exc).__name__}: {exc}"
    return {
        "expected_sha256": expected,
        "current_sha256": current,
        "source_paths": {name: path.relative_to(PROJECT).as_posix()
                          for name, (path, _) in RAW_CIF_PREPARATION_DEPENDENCIES.items()},
        "errors": errors,
    }


def _raw_cif_dependencies_match(state):
    return state["current_sha256"] == state["expected_sha256"] and not state["errors"]


def _raw_cif_dependency_audit(before, after):
    return {
        "expected_sha256": before["expected_sha256"],
        "before_sha256": before["current_sha256"],
        "after_sha256": after["current_sha256"],
        "source_paths": before["source_paths"],
        "stable_and_pinned": (
            _raw_cif_dependencies_match(before)
            and _raw_cif_dependencies_match(after)
            and before["current_sha256"] == after["current_sha256"]
        ),
        "errors_before": before["errors"],
        "errors_after": after["errors"],
    }


def _runtime_audit(before, after):
    stable = (
        _runtime_state_matches_pin(before)
        and _runtime_state_matches_pin(after)
        and before["runtime_hashes_current"] == after["runtime_hashes_current"]
        and before["run100_summary_sha256"] == after["run100_summary_sha256"]
    )
    return {
        "run100_summary_sha256": before["run100_summary_sha256"],
        "run100_runtime_hashes_expected": before["run100_runtime_hashes_expected"],
        "runtime_hashes_before": before["runtime_hashes_current"],
        "runtime_hashes_after": after["runtime_hashes_current"],
        "run100_source_snapshot_hashes": before["run100_source_snapshot_hashes"],
        "runtime_source_paths": before["runtime_source_paths"],
        "run100_source_snapshot_paths": before["run100_source_snapshot_paths"],
        "runtime_hashes_stable": stable,
        "runtime_hash_errors_before": before["runtime_hash_errors"],
        "runtime_hash_errors_after": after["runtime_hash_errors"],
    }


def _append_case_jsonl(path, result):
    line = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


def _fail_summary(attempt_dir, selection_hash, results, stage, error, extra=None):
    summary = {
        "status": "failed",
        "selection_sha256": selection_hash,
        "selected": NEW_TRAIN_ROWS,
        "processed": len(results),
        "prepared": sum(item["status"] == "pass" for item in results),
        "failed": sum(item["status"] == "fail" for item in results),
        "failure_stage_counts": dict(Counter(item["stage"] for item in results if item["status"] == "fail")),
        "global_error": {"stage": stage, "error": error} if error else None,
        "data_written": False,
    }
    if extra:
        summary.update(extra)
    _write_case_jsonl(attempt_dir / "cases.jsonl", results)
    write_json(attempt_dir / "summary.json", summary)
    return summary


def prepare_stage():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("--prepare must run inside an allocated Slurm job")
    if DATA_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {DATA_DIR}")
    selection_path = PREPARE_DIR / "selection.json"
    selection_bytes = selection_path.read_bytes()
    selection_hash = sha256(selection_bytes)
    if (PREPARE_DIR / "selection.sha256").read_text().strip() != selection_hash:
        raise ValueError("Frozen selection.json differs from its SHA-256 sidecar")
    if sha256(CSV_PATH.read_bytes()) != CSV_SHA256:
        raise ValueError("MP20 train CSV does not match its pinned SHA-256")
    selection = json.loads(selection_bytes)
    cases = selection["cases"]
    if selection.get("selected") != NEW_TRAIN_ROWS or len(cases) != NEW_TRAIN_ROWS:
        raise ValueError("Frozen selection must contain exactly 224 cases")
    if len({case["id"] for case in cases}) != NEW_TRAIN_ROWS or len(
        {case["input_cif_sha256"] for case in cases}
    ) != NEW_TRAIN_ROWS:
        raise ValueError("Frozen selection IDs and CIF hashes must each be unique")

    old = _old_cohort()
    dependency_before = _raw_cif_dependency_state()
    if not _raw_cif_dependencies_match(dependency_before):
        raise ValueError("Raw-CIF preparation dependencies differ from their fixed source hashes")
    runtime_before = _runtime_source_state()
    if not _runtime_state_matches_pin(runtime_before):
        raise ValueError("Current CNL runtime sources or run100 snapshots differ from the stable run100 pins")
    if sha256(HELPER_PATH.read_bytes()) != EXPECTED_HELPER_SHA256:
        raise ValueError("Frozen scaled32 prepare_data.py helper hash changed")
    frozen_helper = _load_file_module("coverage256_scaled32_prepare_data", HELPER_PATH)
    frozen = frozen_helper.verify_frozen_sources()
    if not {case["id"] for case in cases}.isdisjoint(old["ids"]):
        raise ValueError("New selected IDs overlap the complete old100 selection")
    if not {case["input_cif_sha256"] for case in cases}.isdisjoint(old["cif_hashes"]):
        raise ValueError("New selected CIF hashes overlap the complete old100 selection")

    attempt_dir = _next_attempt_dir()
    case_log = attempt_dir / "cases.jsonl"
    case_log.touch(exist_ok=False)
    results, new_rows = [], []
    global_stage, global_error = None, None
    runtime_policy = None
    try:
        actual_versions = {name: importlib.metadata.version(name) for name in frozen_helper.RUNTIME_PACKAGES}
        expected_python = frozen["base_manifest"]["provenance"]["python_version"]
        runtime_policy = _runtime_policy(
            frozen["runtime_versions"], actual_versions, expected_python, sys.version.split()[0]
        )
        if not runtime_policy["accepted"]:
            raise ValueError(f"Runtime policy rejected preparation environment: {runtime_policy}")
        inline, helper, adapter, score = _load_adapters(frozen_helper)
        case_paths = _source_paths(cases, attempt_dir)
        adapter._source_paths = case_paths.__getitem__

        for case in cases:
            case_id = case["id"]
            result = {
                "id": case_id,
                "status": "fail",
                "stage": "input_integrity",
                "metadata_space_group": case["dataset_space_group"],
                "input_cif_sha256": case["input_cif_sha256"],
            }
            try:
                paths = case_paths[case_id]
                cif_bytes = paths["cif"].read_bytes()
                if sha256(cif_bytes) != case["input_cif_sha256"]:
                    raise ValueError("Selected CIF hash changed")
                result["stage"] = "facts_plan"
                facts, plan, _, _ = inline.prepare_case(str(paths["cif"]), adapter.TABLE)
                if frozenset(plan["composition"]) in old["heldout_element_sets"]:
                    raise ValueError("Encoded plan element set matches a heldout composition")
                result["stage"] = "relational_encode"
                text = adapter.relational_codec.encode(facts, table=adapter.TABLE)
                if text != adapter._render_plan(plan):
                    raise ValueError("Facts encoder and blockrunner plan renderer differ")
                result["stage"] = "cache_write"
                for key in ("plan", "facts"):
                    paths[key].parent.mkdir(parents=True, exist_ok=True)
                plan_bytes = (json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
                facts_bytes = (json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
                text_bytes = text.encode("utf-8")
                paths["plan"].write_bytes(plan_bytes)
                paths["facts"].write_bytes(facts_bytes)
                paths["text"].write_bytes(text_bytes)
                hashes = {
                    "selection_sha256": selection_hash,
                    "source_plan_sha256": sha256(plan_bytes),
                    "source_facts_sha256": sha256(facts_bytes),
                    "source_text_sha256": sha256(text_bytes),
                    "source_cif_sha256": case["input_cif_sha256"],
                }
                result["stage"] = "contracts_geometry_physical_capacity"
                row = helper.prepare_row(case_id, selection_bytes, {case_id: hashes}, adapter, score)
                row["source_files"]["selection"] = selection_path.relative_to(PROJECT).as_posix()
                row["source_files"]["cif"] = paths["cif"].relative_to(PROJECT).as_posix()
                new_rows.append(row)
                result.update({
                    "status": "pass",
                    "stage": "complete",
                    "encoded_space_group": int(plan["space_group_number"]),
                    "encoded_wyckoff_labels": sorted({orbit["wyckoff"] for orbit in plan["orbits"]}),
                    "token_lengths": row["token_lengths"],
                    "contracts_passed": ["exact_plan", "geometry", "canonical_text", "second_decode",
                                         "source_plus_minus_physical_audit", "K2560_max8192"],
                })
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            _append_case_jsonl(case_log, result)
            if len(results) % 10 == 0 or len(results) == NEW_TRAIN_ROWS:
                passed = sum(item["status"] == "pass" for item in results)
                failed = len(results) - passed
                print(f"coverage256 prepare: {len(results)}/{NEW_TRAIN_ROWS} cases, pass={passed}, fail={failed}",
                      flush=True)
    except Exception as exc:
        global_stage = "runtime_or_source_load"
        global_error = f"{type(exc).__name__}: {exc}"

    packing, packing_error, pack_failures = None, None, []
    old_rows = old["rows"]
    all_rows = old_rows + new_rows
    if not global_error and len(results) == NEW_TRAIN_ROWS and all(item["status"] == "pass" for item in results):
        try:
            packing = helper.audit_packing(all_rows, adapter)
        except Exception as exc:
            packing_error = f"{type(exc).__name__}: {exc}"
            for row in all_rows:
                try:
                    helper.audit_packing([row], adapter)
                except Exception as row_exc:
                    pack_failures.append({"id": row["id"], "error": f"{type(row_exc).__name__}: {row_exc}"})
                    if row["id"] in {item["id"] for item in results}:
                        item = next(item for item in results if item["id"] == row["id"])
                        item.update({"status": "fail", "stage": "packing_audit", "error": pack_failures[-1]["error"]})

    runtime_after = _runtime_source_state()
    runtime_audit = _runtime_audit(runtime_before, runtime_after)
    dependency_after = _raw_cif_dependency_state()
    dependency_audit = _raw_cif_dependency_audit(dependency_before, dependency_after)
    preparation_audits = {
        "runtime_sources": runtime_audit,
        "raw_cif_preparation_dependencies": dependency_audit,
        "runtime_environment_policy": runtime_policy,
    }
    if not runtime_audit["runtime_hashes_stable"]:
        global_stage = global_stage or "runtime_source_hash_after"
        global_error = global_error or "CNL runtime source hashes changed or no longer match run100 pins"
    if not dependency_audit["stable_and_pinned"]:
        global_stage = global_stage or "raw_cif_dependency_hash_after"
        global_error = global_error or "Raw-CIF preparation dependency hashes changed during preparation"
    if global_error:
        return _fail_summary(attempt_dir, selection_hash, results, global_stage, global_error,
                             preparation_audits)
    if len(results) != NEW_TRAIN_ROWS or any(item["status"] != "pass" for item in results):
        return _fail_summary(attempt_dir, selection_hash, results, None, None,
                             {"all_selected_cases_attempted": len(results) == NEW_TRAIN_ROWS,
                              **preparation_audits})
    if packing_error:
        return _fail_summary(attempt_dir, selection_hash, results, "aggregate_packing_audit", packing_error,
                             {"packing_failures": pack_failures, **preparation_audits})

    output_bytes = append_jsonl(old["records_bytes"], new_rows)
    old_ids = {row["id"] for row in old_rows}
    old_cif_hashes = {row["source_hashes"]["source_cif_sha256"] for row in old_rows}
    new_ids = {row["id"] for row in new_rows}
    new_cif_hashes = {row["source_hashes"]["source_cif_sha256"] for row in new_rows}
    if not new_ids.isdisjoint(old_ids) or not new_cif_hashes.isdisjoint(old_cif_hashes):
        return _fail_summary(attempt_dir, selection_hash, results, "novelty_audit",
                             "New IDs or CIF hashes overlap the preserved old48 records",
                             preparation_audits)

    train_ids = [row["id"] for row in all_rows if row["split"] == "train"]
    heldout_ids = [row["id"] for row in all_rows if row["split"] == "heldout"]
    if len(all_rows) != 272 or len(train_ids) != 256 or len(heldout_ids) != 16:
        return _fail_summary(attempt_dir, selection_hash, results, "split_audit",
                             f"Expected 272 rows with train256/heldout16; got {len(all_rows)}/{len(train_ids)}/{len(heldout_ids)}",
                             preparation_audits)

    selected_hashes = {case["id"]: case["input_cif_sha256"] for case in cases}
    selection_counts = selection["exclusions"]
    coverage = _coverage(all_rows)
    manifest = {
        "status": "data_prepared_not_trained",
        "representation": "relational_cnl_v1",
        "cohort": {
            "selected_new_rows": NEW_TRAIN_ROWS,
            "total_records": len(all_rows),
            "train_count": len(train_ids),
            "heldout_count": len(heldout_ids),
            "train_ids": train_ids,
            "heldout_ids": heldout_ids,
            "selection_path": selection_path.relative_to(PROJECT).as_posix(),
            "selection_sha256": selection_hash,
            "excluded_rows": {
                "unique_union_count": selection_counts["excluded_rows"],
                "criterion_match_counts_nonexclusive": selection_counts["excluded_by_reason"],
                "criterion_counts_overlap": True,
            },
        },
        "prefix_preservation": {
            "source_records_path": OLD_RECORDS_PATH.relative_to(PROJECT).as_posix(),
            "source_records_sha256": OLD48_SHA256,
            "output_prefix_sha256": sha256(output_bytes[:len(old["records_bytes"])]),
            "prefix_byte_identical": output_bytes.startswith(old["records_bytes"]),
            "prefix_bytes": len(old["records_bytes"]),
            "prefix_rows": 48,
        },
        "records": {"path": "records.jsonl", "count": len(all_rows), "sha256": sha256(output_bytes)},
        "records_sha256": sha256(output_bytes),
        "selection_protocol": {
            "rule": selection["rule"],
            "source_csv_sha256": CSV_SHA256,
            "eligible_unique_rows": selection_counts["eligible_unique_rows"],
            "selected_metadata_space_groups": sorted({case["dataset_space_group"] for case in cases}),
            "selected_cif_sha256_by_id": selected_hashes,
            "replacement_after_encoding_failure": False,
        },
        "novelty_audit": {
            "new_ids_unique": len(new_ids) == NEW_TRAIN_ROWS,
            "new_ids_disjoint_from_old100": new_ids.isdisjoint(old["ids"]),
            "new_cif_hashes_unique": len(new_cif_hashes) == NEW_TRAIN_ROWS,
            "new_cif_hashes_disjoint_from_old100": new_cif_hashes.isdisjoint(old["cif_hashes"]),
            "new_ids_disjoint_from_old48_records": new_ids.isdisjoint(old_ids),
            "new_cif_hashes_disjoint_from_old48_records": new_cif_hashes.isdisjoint(old_cif_hashes),
            "new_actual_compositions_disjoint_from_heldout_element_sets": True,
        },
        "coverage": coverage,
        "new_case_validation": {
            "selected": NEW_TRAIN_ROWS,
            "prepared": len(new_rows),
            "failed": 0,
            "contracts_passed_per_case": ["exact_plan", "geometry", "canonical_text", "second_decode",
                                          "source_plus_minus_physical_audit", "K2560_max8192"],
            "packing": packing,
        },
        "capacity": {"K": helper.K, "MAXLEN": helper.MAXLEN},
        "provenance": {
            "csv_sha256": CSV_SHA256,
            "old100_selection_sha256": sha256(old["selection_bytes"]),
            "old48_records_sha256": OLD48_SHA256,
            "scaled32_prepare_data_sha256": EXPECTED_HELPER_SHA256,
            "inline_prepare_sha256": EXPECTED_INLINE_SHA256,
            "blockrunner_code_sha256": EXPECTED_BLOCK_CODE_HASHES,
            "public_wyckoff_table_sha256": adapter.relational_codec.geometry.FROZEN_TABLE_SHA256,
            "codec": "relational_cnl_v1",
            "tokenizer_files": frozen["base_manifest"]["provenance"]["tokenizer_files"],
            "runtime_versions": actual_versions,
            "python_version": sys.version.split()[0],
            "cnl_runtime_sources": runtime_audit,
            "raw_cif_preparation_dependencies": dependency_audit,
            "runtime_environment_policy": runtime_policy,
        },
        "later_training": {"scheduled": False, "status": "not_run"},
    }
    temp_dir = ROOT / ".data_001.tmp"
    if temp_dir.exists():
        raise FileExistsError(f"Refusing to overwrite temporary output {temp_dir}")
    temp_dir.mkdir()
    try:
        (temp_dir / "records.jsonl").write_bytes(output_bytes)
        write_json(temp_dir / "manifest.json", manifest)
        temp_dir.rename(DATA_DIR)
    except Exception:
        import shutil
        shutil.rmtree(temp_dir)
        raise

    _write_case_jsonl(attempt_dir / "cases.jsonl", results)
    write_json(attempt_dir / "summary.json", {
        "status": "prepared_not_trained",
        "selection_sha256": selection_hash,
        "selected": NEW_TRAIN_ROWS,
        "processed": NEW_TRAIN_ROWS,
        "prepared": len(new_rows),
        "failed": 0,
        "data_path": DATA_DIR.relative_to(PROJECT).as_posix(),
        "records_sha256": sha256(output_bytes),
        "manifest_sha256": sha256((DATA_DIR / "manifest.json").read_bytes()),
        **preparation_audits,
    })
    return {"status": "prepared_not_trained", "data_path": DATA_DIR.relative_to(PROJECT).as_posix(),
            "records_sha256": sha256(output_bytes), "manifest_sha256": sha256((DATA_DIR / "manifest.json").read_bytes())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--select", action="store_true", help="freeze the 224 metadata-selected inputs")
    stages.add_argument("--prepare", action="store_true", help="prepare and validate all frozen rows inside Slurm")
    args = parser.parse_args(argv)
    result = select_stage() if args.select else prepare_stage()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
