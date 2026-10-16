"""Small offline checks for the frozen relational pilot adapter."""
from copy import deepcopy
from decimal import Decimal
import json

from prepare import DATA_DIR, K, MAXLEN, ROUNDTRIP_ROOT, TOK, load_records, tasks
import prepare
import score


def main():
    rows = load_records()
    assert len(rows) == 24
    assert [row["split"] for row in rows] == ["train"] * 8 + ["heldout"] * 16
    assert len({row["id"] for row in rows}) == 24

    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    selected_ids = [case["id"] for case in json.loads((ROUNDTRIP_ROOT / "selection.json").read_text())["cases"]]
    assert manifest["cohort"]["selected_total"] == 100 and manifest["cohort"]["included"] == 24
    assert manifest["cohort"]["train_ids"] == selected_ids[:8] == [row["id"] for row in rows[:8]]
    assert manifest["cohort"]["heldout_ids"] == selected_ids[8:24] == [row["id"] for row in rows[8:]]
    assert manifest["cohort"]["excluded_from_this_pilot"] == selected_ids[24:]
    assert len(manifest["cohort"]["excluded_from_this_pilot"]) == 76
    assert manifest["capacity"] == {"K": K, "MAXLEN": MAXLEN}
    for key in ("source", "plus", "minus"):
        report = manifest["reference_failure_report"][key]
        assert report["denominator"] == 24 and not report["failed_ids"] and not report["classes"]
        assert manifest["physical_reference_audit"][f"{key}_cases"] == 24

    for row in rows:
        source = row["plans"]["source"]
        assert prepare.relational_codec.decode(row["c1"], table=prepare.TABLE) == source
        assert score.audit(row["c1"], source)["valid"]
        assert score.grade_answer(score.answer(row["c1"]), source)["success"]
        for direction, key, factor in (("plus", "plus", "1.05"), ("minus", "minus", "0.95")):
            target = deepcopy(source)
            target["lattice"][:3] = [
                float(Decimal(repr(length)) * Decimal(factor)) for length in source["lattice"][:3]
            ]
            assert row["plans"][key] == target
            assert prepare.relational_codec.decode(row[key], table=prepare.TABLE) == target
            assert score.audit(row[key], target)["valid"]
            assert score.grade_edit(row[key], row["c1"], factor)["success"]
            assert len(TOK.encode(row[key], add_special_tokens=False)) + 1 <= K
            for example in tasks(row, row["split"], direction):
                used = prepare.encoded_length(example)
                assert used <= MAXLEN, (row["id"], direction, example["task"], used)

    bad_text = "not a relational crystal description"
    failure = score.audit(bad_text, rows[0]["plans"]["source"])
    assert not failure["valid"] and failure["raw_text"] == bad_text and failure["reasons"]
    invalid_formula = score.grade_answer("The composition is Qq999.", rows[0]["plans"]["source"])
    assert not invalid_formula["success"] and invalid_formula["fields"]["formula"] is False
    assert not score.grade_edit(bad_text, rows[0]["c1"], "1.05")["success"]

    good_edit = score.grade_edit(rows[0]["plus"], rows[0]["c1"], "1.05")
    assert good_edit["success"]
    json.dumps(good_edit, allow_nan=False)
    wrong_but_valid = rows[1]["plus"]
    assert score.audit(wrong_but_valid)["valid"]
    wrong_grade = score.grade_edit(wrong_but_valid, rows[0]["c1"], "1.05")
    assert not wrong_grade["success"] and wrong_grade["raw_text"] == wrong_but_valid
    scale_errors = wrong_grade["diagnostics"]["whole_geometry_scale_errors_A"]
    assert not scale_errors["defined"] and scale_errors["undefined_reason"] == "atom_count_or_composition_mismatch"
    assert scale_errors["coordinate_max"] is None
    json.dumps(wrong_grade, allow_nan=False)

    degenerate_plan = deepcopy(rows[1]["plans"]["source"])
    oxygen_orbit = next(
        orbit for orbit in degenerate_plan["orbits"]
        if orbit["element"] == "O" and orbit["wyckoff"] == "8i"
    )
    oxygen_orbit["parameters"][0] = 0.0
    degenerate_text = prepare.relational_codec.encode(
        prepare.source_facts_from_plan(degenerate_plan), table=prepare.TABLE
    )
    geometry_failure = score.audit(degenerate_text)
    assert geometry_failure["failure_stage"] == "geometry" and geometry_failure["reasons"]
    assert geometry_failure["decoded_plan"]
    failed_edit = score.grade_edit(degenerate_text, rows[1]["c1"], "1.05")
    assert not failed_edit["success"] and failed_edit["raw_text"] == degenerate_text
    assert failed_edit["output_audit"]["failure_stage"] == "geometry"
    assert failed_edit["output_audit"]["reasons"] == geometry_failure["reasons"]
    json.dumps(failed_edit, allow_nan=False)
    print("adapter checks passed: 24 fixed rows, 48 exact edits, strict failures retained")


if __name__ == "__main__":
    main()
