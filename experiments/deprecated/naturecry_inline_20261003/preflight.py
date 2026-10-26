"""Run the local source-to-inline-to-public-geometry preflight; never calls an API."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments/naturecry_llm_100_20261002"))
sys.path.insert(0, str(HERE))

import cohort
import api_runner
import codec
import geometry
import prepare
from pymatgen.io.cif import CifWriter


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run(output=HERE / "preflight/final_002", prepared=HERE / "prepared_final"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preflight output: {output}")
    hashes, manifest, selection, hash_verification = api_runner.verify_prepared(prepared, include_preflight=True)
    output.mkdir(parents=True)
    prepare.update_progress("preflight_shared_entry", "in_progress", "Running deterministic local fixtures through the shared decoder gate and public geometry executor after entry hash verification.")
    ids = manifest["case_ids_in_original_order"]
    if len(ids) != 100 or len(set(ids)) != 100 or ids != [case["id"] for case in selection["cases"]]:
        raise ValueError("Prepared cohort is not the unchanged frozen 100-case selection")
    cases = {case["id"]: case for case in manifest["cases"]}
    source_root = prepare.FROZEN / "inputs"
    table = geometry.verify_public_table()
    records = []
    example_written = False
    for case_id in ids:
        source_record = cases[case_id]
        record = {"case": case_id, "status": "fail", "stage": "source_preparation",
                  "encoder_api_requests": 0, "decoder_api_requests": 0}
        if source_record["status"] != "prepared":
            record["failure_kind"] = "source_preparation"
            record["error"] = source_record.get("error", "source preparation failed")
            records.append(record)
            continue
        try:
            source = source_root / f"{case_id}.cif"
            source_sha = source_record["input_sha256_actual"]
            case_dir = prepared / "cases" / case_id
            oracle_path = prepared / "preflight_only/oracle_plans" / f"{case_id}.json"
            facts = json.loads((case_dir / "encoder_facts.json").read_text())
            inventory_file = json.loads((case_dir / "inventory.json").read_text())
            inventory = {field["name"]: {key: field[key] for key in ("type", "unit", "frame", "value")}
                         for field in inventory_file["fields"]}
            narrative = codec.fixture_narrative(facts)
            document = codec.bind_inline(narrative, inventory)
            codec.validate_encoder_document(document, inventory)
            oracle = json.loads(oracle_path.read_text())
            record["stage"] = "shared_decoder_gate"
            decoded_plan = codec.decode_plan_response(json.dumps(oracle, ensure_ascii=False, allow_nan=False), document, table)
            write_json(output / "decoded_plans" / f"{case_id}.json", decoded_plan)
            record["stage"] = "public_geometry_execution"
            rebuilt = geometry.execute_geometry_plan(decoded_plan, table)
            decoded_cif = output / "decoded_cifs" / f"{case_id}.cif"
            decoded_cif.parent.mkdir(parents=True, exist_ok=True)
            CifWriter(rebuilt, symprec=None, significant_figures=14).write_file(decoded_cif)
            record["stage"] = "original_structure_match"
            original = cohort.parse(source)
            reparsed = cohort.parse(decoded_cif)
            coord_error, lattice_error = geometry.compare_structures(original, reparsed)
            record.update(status="pass", stage="complete", input_sha256=source_sha,
                          facts_sha256=source_record["facts_sha256"],
                          inline_sha256=digest(document.encode()), coordinate_error_A=coord_error,
                          lattice_error_A=lattice_error, atom_count=len(reparsed),
                          decoded_cif_sha256=digest(decoded_cif.read_bytes()))
            if not example_written and facts["precision_corrections"]:
                example = output / "example_with_correction.md"
                example.write_text("# Deterministic inline binding fixture\n\n"
                                   "This is generated from frozen source facts without an LLM call. It demonstrates binding and is not a generated description or an LLM result.\n\n"
                                   "## Placeholder prose\n\n" + narrative + "\n\n## Bound inline document\n\n" + document + "\n")
                example_written = True
        except Exception as error:
            record.update(failure_kind=getattr(error, "stage", "preflight"),
                          error=f"{type(error).__name__}: {error}")
        records.append(record)

    passed = [record for record in records if record["status"] == "pass"]
    summary = {
        "kind": "offline deterministic protocol and public geometry preflight",
        "not_an_llm_result": True,
        "uses_same_decoder_gate_and_geometry_executor": True,
        "scores_reparsed_14_significant_figure_cif": True,
        "selected_denominator": 100,
        "completed": len(records),
        "passed": len(passed),
        "failed": len(records) - len(passed),
        "failures_by_stage": dict(Counter(record["stage"] for record in records if record["status"] != "pass")),
        "failures_by_kind": dict(Counter(record.get("failure_kind", "unspecified") for record in records if record["status"] != "pass")),
        "api_requests": {"encoder": 0, "decoder": 0, "total": 0},
        "per_case_api_requests_reconciled": all(record["encoder_api_requests"] == 0 and record["decoder_api_requests"] == 0 for record in records),
        "selection_sha256": manifest["selection_sha256"],
        "hash_verification": hash_verification,
        "runtime_hashes": hashes,
        "coordinate_tolerance_A": 1e-6,
        "lattice_component_tolerance_A": 1e-8,
        "maximum_coordinate_error_A": max((record["coordinate_error_A"] for record in passed), default=None),
        "maximum_lattice_error_A": max((record["lattice_error_A"] for record in passed), default=None),
        "fixture_narrative": "deterministic placeholder fixture; it does not test natural-language semantic understanding",
        "semantic_prose_audit": "not_executed; tag accuracy does not establish correctness of surrounding prose",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(output / "records.json", records)
    write_json(output / "summary.json", summary)
    prepare.update_progress("preflight_shared_entry", "completed" if summary["passed"] == 100 else "completed_with_failures",
                            f"Offline preflight: {summary['passed']}/100; API calls 0; maximum coordinate error {summary['maximum_coordinate_error_A']} A.")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    prepared_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "prepared_final"
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else HERE / "preflight/final_002"
    result = run(output_path, prepared_path)
    raise SystemExit(0 if result["passed"] == 100 and result["failed"] == 0 else 1)
