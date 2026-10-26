"""Offline round-trip benchmark for the relational NatureCry codec."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
INLINE = PROJECT / "experiments/naturecry_inline_20261003"
BASE = PROJECT / "experiments/naturecry_llm_100_20261002"
TABLE_ROOT = PROJECT / "experiments/explicit_full_mp20_20260923/codec"
PREPARED_DEFAULT = INLINE / "prepared_final"
CONTRACTS = ("exact_plan_match", "geometry_match", "canonical_text_match", "second_decode_match")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def exact_plan_equal(actual, expected):
    dumps = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False)
    return dumps(actual) == dumps(expected)


def require_pilot_for_limit(limit, pilot):
    if limit not in (30, 100):
        raise ValueError("--limit must be 30 or 100")
    if limit == 100 and pilot is None:
        raise ValueError("A completed 30-case --pilot summary is required for --limit 100")


def select_prefix(case_ids, limit):
    if limit not in (30, 100):
        raise ValueError("--limit must be 30 or 100")
    if len(case_ids) != 100:
        raise ValueError("The frozen selection must contain exactly 100 cases")
    return list(case_ids[:limit])


def source_paths():
    return {
        "relational_codec.py": HERE / "relational_codec.py",
        "run_roundtrip.py": Path(__file__).resolve(),
        "geometry.py": INLINE / "geometry.py",
        "base_cohort.py": BASE / "cohort.py",
        "oldcodec.py": INLINE / "codec.py",
        "api_runner.py": INLINE / "api_runner.py",
        "prepare.py": INLINE / "prepare.py",
        "public_wyckoff_baseline.py": TABLE_ROOT / "wyckoff_baseline.py",
        "public_affine_parser.py": TABLE_ROOT / "audit_wyckoff_families.py",
        "cif_parser.py": TABLE_ROOT / "test_program.py",
        "public_wyckoff_list.csv": TABLE_ROOT / "fixtures/wyckoff_list.csv",
    }


def hash_sources(paths=None):
    paths = paths or source_paths()
    return {name: sha256(path.read_bytes()) for name, path in paths.items()}


def snapshot_sources(output, paths, hashes):
    root = output / "sources"
    for name, path in paths.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if sha256(target.read_bytes()) != hashes[name]:
            raise RuntimeError(f"Source snapshot changed while being copied: {name}")
    return {name: str((root / name).relative_to(output)) for name in paths}


def validate_pilot(pilot, selection_sha256, runtime_hashes):
    path = Path(pilot).expanduser().resolve()
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise ValueError(f"Pilot summary does not exist: {path}")
    summary = json.loads(path.read_text())
    if (summary.get("limit") != 30 or summary.get("selected_denominator") != 30
            or summary.get("completed") != 30 or summary.get("passed") != 30
            or summary.get("failed") != 0 or summary.get("all_contracts_passed") is not True):
        raise ValueError("Pilot must record 30/30 completed cases with every contract passing")
    counts = summary.get("contract_pass_counts", {})
    if any(counts.get(contract) != 30 for contract in CONTRACTS):
        raise ValueError("Pilot summary does not show 30/30 passes for every contract")
    if summary.get("selection_sha256") != selection_sha256:
        raise ValueError("Pilot selection hash differs from the frozen selection")
    pilot_hashes = summary.get("runtime_hashes", {})
    if (summary.get("runtime_hashes_stable") is not True
            or summary.get("runtime_hashes_after") != pilot_hashes):
        raise ValueError("Pilot runtime hashes were not stable across execution")
    if pilot_hashes != runtime_hashes:
        raise ValueError("Pilot runtime hashes differ from the current codec, runner, or shared sources")
    return {"path": str(path), "completed": 30, "passed": 30,
            "selection_sha256": selection_sha256, "runtime_hashes_match": True}


def load_modules():
    for path in (INLINE, BASE, HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    api_runner = importlib.import_module("api_runner")
    relational_codec = importlib.import_module("relational_codec")
    return api_runner, api_runner.prepare, api_runner.cohort, api_runner.geometry, api_runner.codec, relational_codec


def frame_support(facts):
    frame = facts["frame_transform"]
    matrix = frame["matrix"]
    shift = frame["origin_shift"]
    expected = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    changed = sum(value != expected[i][j] for i, row in enumerate(matrix) for j, value in enumerate(row))
    changed += sum(value != 0 for value in shift)
    corrections = len(facts.get("precision_corrections", []))
    return changed, corrections


def run(output, prepared=PREPARED_DEFAULT, limit=30, pilot=None):
    require_pilot_for_limit(limit, pilot)
    output = Path(output).expanduser().resolve()
    prepared = Path(prepared).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite relational round-trip output: {output}")

    api_runner, prepare, cohort, geometry, oldcodec, relational_codec = load_modules()
    prepared_hashes, manifest, selection, hash_verification = api_runner.verify_prepared(
        prepared, include_preflight=False)
    case_ids = select_prefix(manifest["case_ids_in_original_order"], limit)
    selection_sha = manifest["selection_sha256"]

    paths = source_paths()
    runtime_hashes = hash_sources(paths)
    pilot_receipt = validate_pilot(pilot, selection_sha, runtime_hashes) if limit == 100 else None
    output.mkdir(parents=True)
    snapshots = snapshot_sources(output, paths, runtime_hashes)
    prepared_records = {record["id"]: record for record in manifest["cases"]}
    input_root = prepare.FROZEN / "inputs"
    table = geometry.verify_public_table()
    records = []
    support = Counter()
    started = time.perf_counter()

    for case_id in case_ids:
        source_record = prepared_records[case_id]
        record = {
            "case": case_id,
            "status": "fail",
            "stage": "source_preparation",
            "failure_stage": None,
            "api_requests": 0,
            "llm_calls": 0,
            "contracts": {contract: None for contract in CONTRACTS},
        }
        if source_record["status"] != "prepared":
            record.update(failure_stage="source_preparation", failure_kind="source_preparation",
                          error="Prepared facts unavailable; retained in the selected denominator")
            records.append(record)
            write_json(output / "records.json", records)
            continue

        case_dir = prepared / "cases" / case_id
        facts_path = case_dir / "encoder_facts.json"
        source_path = input_root / f"{case_id}.cif"
        try:
            record["facts_sha256"] = source_record["facts_sha256"]
            record["input_sha256"] = source_record["input_sha256_actual"]
            facts = json.loads(facts_path.read_text())
            exact_nonidentity_components, residual_count = frame_support(facts)
            support["prepared_fact_cases"] += 1
            manifest_nonidentity = bool(source_record["nonidentity_frame"])
            support["cases_with_nonidentity_frame"] += manifest_nonidentity
            support["cases_with_exact_nonidentity_frame"] += exact_nonidentity_components > 0
            support["exact_nonidentity_frame_components"] += exact_nonidentity_components
            support["precision_correction_count"] += residual_count
            support["cases_with_precision_corrections"] += residual_count > 0
            record["nonidentity_frame"] = manifest_nonidentity
            record["exact_nonidentity_frame_components"] = exact_nonidentity_components
            record["precision_correction_count"] = residual_count

            record["stage"] = "encode"
            text = relational_codec.encode(facts, table=table)
            if not isinstance(text, str) or not text:
                raise ValueError("Encoder returned no text")
            text_path = output / "texts" / f"{case_id}.txt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text)
            final_text = text_path.read_text()
            record["text_sha256"] = sha256(final_text.encode("utf-8"))
            record["text_characters"] = len(final_text)

            record["stage"] = "decode"
            decoded_plan = relational_codec.decode(final_text, table=table)
            write_json(output / "plans" / f"{case_id}.json", decoded_plan)
            expected_plan = oldcodec.plan_from_facts(facts)
            record["contracts"]["exact_plan_match"] = exact_plan_equal(decoded_plan, expected_plan)

            record["stage"] = "geometry_execution"
            from pymatgen.io.cif import CifWriter
            reconstructed = geometry.execute_geometry_plan(decoded_plan, table=table)
            cif_path = output / "cifs" / f"{case_id}.cif"
            cif_path.parent.mkdir(parents=True, exist_ok=True)
            CifWriter(reconstructed, symprec=None, significant_figures=14).write_file(cif_path)
            record["decoded_cif_sha256"] = sha256(cif_path.read_bytes())

            record["stage"] = "original_structure_match"
            original = cohort.parse(source_path)
            reparsed = cohort.parse(cif_path)
            coord_error, lattice_error = geometry.compare_structures(original, reparsed)
            record["contracts"]["geometry_match"] = True
            record["coordinate_error_A"] = coord_error
            record["lattice_error_A"] = lattice_error
            record["atom_count"] = len(reparsed)

            record["stage"] = "canonicalization"
            canonical = relational_codec.normalize(final_text, table=table)
            record["contracts"]["canonical_text_match"] = canonical == final_text

            record["stage"] = "second_decode"
            second_plan = relational_codec.decode(canonical, table=table)
            write_json(output / "second_plans" / f"{case_id}.json", second_plan)
            record["contracts"]["second_decode_match"] = exact_plan_equal(second_plan, expected_plan)

            record["contract_failures"] = [name for name, passed in record["contracts"].items() if not passed]
            record["stage"] = "complete"
            record["status"] = "pass" if not record["contract_failures"] else "fail"
            if record["contract_failures"]:
                record["failure_kind"] = "contract_mismatch"
                record["failure_stage"] = record["contract_failures"][0]
        except Exception as error:
            record["failure_stage"] = record["stage"]
            record["failure_kind"] = "processing_error"
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
        write_json(output / "records.json", records)

    runtime_hashes_after = hash_sources(paths)
    changed_sources = [name for name, digest in runtime_hashes.items()
                       if runtime_hashes_after.get(name) != digest]
    passed = [record for record in records if record["status"] == "pass"]
    contract_counts = {
        name: sum(record["contracts"].get(name) is True for record in records)
        for name in CONTRACTS
    }
    geometry_records = [record for record in records if record.get("coordinate_error_A") is not None]
    failure_stages = Counter(record.get("failure_stage") or record["stage"]
                             for record in records if record["status"] != "pass")
    support_counts = {
        "prepared_fact_cases": support["prepared_fact_cases"],
        "cases_with_nonidentity_frame": support["cases_with_nonidentity_frame"],
        "frame_nonidentity_rule": "prepared manifest allclose(matrix, I, atol=1e-12, rtol=0) and allclose(shift, 0, atol=1e-12, rtol=0)",
        "cases_with_exact_nonidentity_frame": support["cases_with_exact_nonidentity_frame"],
        "exact_nonidentity_frame_components": support["exact_nonidentity_frame_components"],
        "cases_with_precision_corrections": support["cases_with_precision_corrections"],
        "precision_correction_count": support["precision_correction_count"],
    }
    summary = {
        "kind": "offline relational controlled-natural-language codec round-trip",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "selected_denominator": len(case_ids),
        "completed": len(records),
        "passed": len(passed),
        "failed": len(records) - len(passed),
        "failures_by_stage": dict(failure_stages),
        "contract_pass_counts": contract_counts,
        "all_contracts_passed": len(records) == limit and all(value == limit for value in contract_counts.values()),
        "exact_plan_count": contract_counts["exact_plan_match"],
        "geometry_match_count": contract_counts["geometry_match"],
        "canonical_text_count": contract_counts["canonical_text_match"],
        "second_decode_count": contract_counts["second_decode_match"],
        "support_counts": support_counts,
        "maximum_coordinate_error_A": max((record["coordinate_error_A"] for record in geometry_records), default=None),
        "maximum_lattice_error_A": max((record["lattice_error_A"] for record in geometry_records), default=None),
        "selection_sha256": selection_sha,
        "selection_provenance": {
            "rule": "Unfiltered prefix of the unchanged ordered 100-case selection; no retry or replacement",
            "full_selected": len(manifest["case_ids_in_original_order"]),
            "requested_case_ids": case_ids,
            "requested_case_ids_sha256": sha256(json.dumps(case_ids, separators=(",", ":")).encode()),
        },
        "prepared_input": str(prepared),
        "prepared_hash_verification": hash_verification,
        "prepared_runtime_hashes": prepared_hashes,
        "runtime_hashes": runtime_hashes,
        "runtime_hashes_after": runtime_hashes_after,
        "runtime_hashes_stable": not changed_sources,
        "runtime_hash_changes": changed_sources,
        "source_snapshots": snapshots,
        "pilot_gate": pilot_receipt,
        "api_requests": 0,
        "llm_calls": 0,
        "not_an_llm_result": True,
        "naturalness_validated": False,
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    if changed_sources:
        raise RuntimeError(f"Runtime sources changed during execution: {changed_sources}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PREPARED_DEFAULT)
    parser.add_argument("--limit", type=int, choices=(30, 100), default=30)
    parser.add_argument("--pilot", type=Path, help="Completed 30-case summary required for --limit 100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args.output, prepared=args.prepared, limit=args.limit, pilot=args.pilot)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    return 0 if summary["all_contracts_passed"] and summary["runtime_hashes_stable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
