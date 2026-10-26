import sys
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare


TARGET = """Li2O2
Spacegroup: Pm-3m
abc: 4.00 4.00 4.00
angles: 90.00 90.00 90.00
Sites (4)
Li 0.000 0.000 0.000 2a
O 0.500 0.500 0.500 2b"""


def simple_audit(text):
    lines = text.splitlines()
    reasons = []
    if lines[0] != "Li2O2":
        reasons.append("formula_header_mismatch")
    if lines[4] != "Sites (4)":
        reasons.append("sites_header_mismatch")
    if lines[-1].split()[-1] != "2b":
        reasons.append("ValueError: Wyckoff multiplicity does not match label")
    return {"valid": not reasons, "reasons": reasons}


class RepairPreparationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(prepare, "corrupt_structure"),
                        "repair corruption helper is not implemented yet")

    def test_formula_corruption_changes_only_count_and_keeps_original_target(self):
        corrupted, edit, audit = prepare.corrupt_structure(
            TARGET, "formula_count", "mp-1", 42, simple_audit)

        self.assertEqual(TARGET.splitlines()[0], "Li2O2")
        self.assertIn(f"{edit['element']}{edit['corrupted_count']}", corrupted.splitlines()[0])
        self.assertEqual(corrupted.splitlines()[1:], TARGET.splitlines()[1:])
        self.assertEqual(edit["target_count"], 2)
        self.assertEqual(edit["corrupted_count"], edit["target_count"] + edit["delta"])
        self.assertIn(edit["delta"], prepare.COUNT_DELTAS)
        self.assertIn("formula_header_mismatch", audit["reasons"])

    def test_count_corruption_deltas_vary_in_sign_and_magnitude_across_fixed_ids(self):
        formula_deltas, sites_deltas, wyckoff_deltas = [], [], []
        for number in range(100, 132):
            material_id = f"mp-{number}"
            _, formula_edit, formula_audit = prepare.corrupt_structure(
                TARGET, "formula_count", material_id, 42, simple_audit)
            _, sites_edit, sites_audit = prepare.corrupt_structure(
                TARGET, "sites_count", material_id, 42, simple_audit)
            _, wyckoff_edit, wyckoff_audit = prepare.corrupt_structure(
                TARGET, "wyckoff_multiplicity", material_id, 42, simple_audit)
            self.assertIn("formula_header_mismatch", formula_audit["reasons"])
            self.assertIn("sites_header_mismatch", sites_audit["reasons"])
            self.assertIn("Wyckoff multiplicity does not match label", wyckoff_audit["reasons"][0])
            formula_deltas.append(formula_edit["delta"])
            sites_deltas.append(sites_edit["delta"])
            wyckoff_deltas.append(wyckoff_edit["delta"])

        for deltas in (formula_deltas, sites_deltas, wyckoff_deltas):
            self.assertTrue(any(delta < 0 for delta in deltas))
            self.assertTrue(any(delta > 0 for delta in deltas))
            self.assertGreaterEqual(len(set(deltas)), 4)

    def test_sites_and_multiplicity_corruptions_are_single_field_and_audited(self):
        def audit(text):
            lines = text.splitlines()
            reasons = []
            if lines[4] != "Sites (4)":
                reasons.append("sites_header_mismatch")
            if lines[-1].split()[-1] != "2b":
                reasons.append("ValueError: Wyckoff multiplicity does not match label")
            return {"valid": not reasons, "reasons": reasons}

        for kind, reason in (("sites_count", "sites_header_mismatch"),
                             ("wyckoff_multiplicity", "Wyckoff multiplicity does not match label")):
            corrupted, edit, checked = prepare.corrupt_structure(TARGET, kind, "mp-1", 42, audit)
            self.assertFalse(checked["valid"])
            self.assertTrue(any(reason in item for item in checked["reasons"]))
            self.assertNotEqual(corrupted, TARGET)
            self.assertEqual(sum(left != right for left, right in
                                 zip(corrupted.splitlines(), TARGET.splitlines())), 1)
            if kind == "sites_count":
                self.assertEqual(edit["corrupted_count"], edit["target_count"] + edit["delta"])
            if kind == "wyckoff_multiplicity":
                self.assertEqual(edit["corrupted_label"][-1], edit["target_label"][-1])
                self.assertEqual(int(edit["corrupted_label"][:-1]),
                                 int(edit["target_label"][:-1]) + edit["delta"])

    def test_checker_feedback_uses_request_condition_and_corrupted_audit_only(self):
        audit = {"valid": False, "reasons": ["formula_header_mismatch"],
                 "facts": {"formula": "Li3O2", "sg": 221, "atoms": 4, "orbits": 2}}
        feedback = prepare.checker_feedback(
            "Generate a crystal with space group number 221 and energy above hull no greater than 0.2 eV/atom",
            audit)
        self.assertIn("0.2 eV/atom", feedback)
        self.assertNotIn("0.08 eV/atom", feedback)
        self.assertIn("formula=Li3O2", feedback)
        self.assertNotIn("Li2O2", feedback)
        self.assertIn("decoded space group is undetermined", feedback)

    def test_repair_prompt_contains_corrupted_input_without_target_leakage(self):
        corrupted, edit, _ = prepare.corrupt_structure(
            TARGET, "formula_count", "mp-1", 42, simple_audit)
        generation = {
            "prefix": "User: Generate a crystal with space group number 221 and energy above hull no greater than 0.2 eV/atom.\nAssistant:",
            "lead": " source description\n<CRYSTAL_START>",
            "crystal": TARGET,
        }
        task = prepare.build_task_r(
            generation, corrupted,
            "Checker feedback: formula_header_mismatch; ehull not evaluated.",
            "The formula header count for Li is corrected from 3 to 2.")

        self.assertIn(corrupted, task["prefix"])
        self.assertIn("Previous failed crystal (input data):", task["prefix"])
        self.assertNotIn(TARGET, task["prefix"])
        self.assertNotIn(TARGET, task["lead"])
        self.assertTrue(task["lead"].endswith(prepare.START))
        self.assertNotIn("<CRYSTAL_START>", task["prefix"])
        self.assertNotIn("<CRYSTAL_END>", task["prefix"])
        self.assertEqual(task["crystal"], TARGET)
        self.assertIn("0.2 eV/atom", task["prefix"])
        self.assertEqual(task["prefix"].count("User:"), 1)
        self.assertEqual(task["prefix"].count("Assistant:"), 1)

    def test_requested_ineligible_type_falls_back_and_records_both_types(self):
        class GeneralPosition:
            def get_dof(self):
                return 3

        class Group:
            def get_wyckoff_position(self, _label):
                return GeneralPosition()

        def symmetry_group(_sg):
            return Group()

        result = prepare.build_corruption(
            TARGET, "coordinate_constraint", "mp-1", 42, simple_audit,
            symmetry_group=symmetry_group)

        self.assertEqual(result["requested_type"], "coordinate_constraint")
        self.assertEqual(result["type"], "formula_count")
        self.assertFalse(result["audit"]["valid"])
        self.assertEqual(result["structure"].splitlines()[1:], TARGET.splitlines()[1:])

    def test_coordinate_corruption_changes_one_representative_and_is_audited(self):
        class ConstrainedPosition:
            def get_dof(self):
                return 2

        class Group:
            def get_wyckoff_position(self, _label):
                return ConstrainedPosition()

        def audit(text):
            reasons = [] if text == TARGET else ["wyckoff_coordinate_constraint"]
            return {"valid": not reasons, "reasons": reasons}

        offsets = []
        for number in range(100, 132):
            corrupted, edit, checked = prepare.corrupt_structure(
                TARGET, "coordinate_constraint", f"mp-{number}", 42, audit,
                symmetry_group=lambda _symbol: Group())
            before, after = TARGET.splitlines(), corrupted.splitlines()
            changed_lines = [index for index, values in enumerate(zip(before, after))
                             if values[0] != values[1]]
            self.assertEqual(len(changed_lines), 1)
            changed_fields = [index for index, values in enumerate(zip(
                before[changed_lines[0]].split(), after[changed_lines[0]].split()))
                              if values[0] != values[1]]
            self.assertEqual(len(changed_fields), 1)
            self.assertIn(changed_fields[0], (1, 2, 3))
            self.assertFalse(checked["valid"])
            self.assertEqual(edit["target_coordinate"], before[changed_lines[0]].split()[changed_fields[0]])
            self.assertEqual(TARGET, "\n".join(before))
            offsets.append(edit["fractional_delta"])
        self.assertTrue(any(offset < 0 for offset in offsets))
        self.assertTrue(any(offset > 0 for offset in offsets))
        self.assertGreaterEqual(len(set(offsets)), 4)


if __name__ == "__main__":
    unittest.main()
