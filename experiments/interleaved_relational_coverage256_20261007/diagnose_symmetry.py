"""Compare the unchanged source and reconstructed geometry under both SG protocols."""
import json
import os
from pathlib import Path
import prepare_data as prep


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Run geometry diagnostics on an allocated node"
    helper = prep._load_file_module("coverage_symmetry_helper", prep.HELPER_PATH)
    inline, _, adapter, score = prep._load_adapters(helper)
    import spglib

    attempt = prep.PREPARE_DIR / "prepare_attempt_002"
    failures = [x for x in map(json.loads, (attempt / "cases.jsonl").read_text().splitlines())
                if x["status"] == "fail"]
    cases = {x["id"]: x for x in prep.read_json(prep.PREPARE_DIR / "selection.json")["cases"]}
    assert len(failures) == 13
    results = []
    for failure in failures:
        case_id = failure["id"]
        cache = attempt / "cache" / case_id
        plan = prep.read_json(cache / "plan.json")
        text = (cache / "text.txt").read_text()
        assert adapter.relational_codec.decode(text, table=adapter.TABLE) == plan
        original = inline.cohort.parse(str(prep.PROJECT / cases[case_id]["input_cif_path"]))
        reconstructed = adapter._structure(plan)
        errors = adapter.relational_codec.geometry.compare_structures(original, reconstructed)
        assert errors[0] < 1e-6 and errors[1] < 1e-8
        detections = {}
        for name, structure in (("source", original), ("reconstructed", reconstructed)):
            cell = (structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers)
            detections[name] = {
                f"symprec={tol},angle={angle}": int(spglib.get_symmetry_dataset(
                    cell, symprec=tol, angle_tolerance=angle).number)
                for tol in (0.001, 0.01) for angle in (-1, 5)
            }
        results.append({"id": case_id, "CSV_SG": failure["metadata_space_group"],
                        "declared_plan_SG": plan["space_group_number"],
                        "geometry_error_A": {"coordinates": errors[0], "lattice": errors[1]},
                        "detections": detections,
                        "frozen_audit": {k: v for k, v in score.audit(text).items()
                                         if k in ("valid", "failure_stage", "reasons")}})
    output = {"scope": "13 fixed preparation failures; no geometry/scorer/selection edits",
              "selection_sha256": prep.sha256((prep.PREPARE_DIR / "selection.json").read_bytes()),
              "results": results}
    path = prep.ROOT / "symmetry_protocol_diagnostic_001.json"
    with path.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
