import unittest

from experiments.crystal_tasks_v1.evaluation.understanding import (
    grade_four_facts,
    summarize_understanding,
)


class UnderstandingTests(unittest.TestCase):
    response = (
        "The composition is K2 Cu2 S2. The space group number is 119. "
        "The unit cell contains 6 atoms and 3 independent orbits."
    )

    def grade(self, response=None, **reference):
        return grade_four_facts(
            self.response if response is None else response,
            reference.get("formula", "K2Cu2S2"),
            reference.get("space_group", 119),
            reference.get("atom_count", 6),
            reference.get("orbit_count", 3),
        )

    def test_gold_response_gets_all_four_facts(self):
        result = self.grade()
        self.assertEqual(
            result["fields"],
            {
                "formula": True,
                "space_group": True,
                "cell_atom_count": True,
                "independent_orbit_count": True,
            },
        )
        self.assertTrue(result["all_four_correct"])

    def test_each_single_wrong_fact_only_loses_its_field(self):
        cases = {
            "formula": "The composition is K2 Cu2 O2.",
            "space_group": "The space group number is 120.",
            "cell_atom_count": "The unit cell contains 7 atoms",
            "independent_orbit_count": "4 independent orbits",
        }
        for field, replacement in cases.items():
            with self.subTest(field=field):
                response = self.response.replace(
                    {
                        "formula": "The composition is K2 Cu2 S2.",
                        "space_group": "The space group number is 119.",
                        "cell_atom_count": "The unit cell contains 6 atoms",
                        "independent_orbit_count": "3 independent orbits",
                    }[field],
                    replacement,
                )
                result = self.grade(response)
                self.assertFalse(result["fields"][field])
                self.assertEqual(
                    [name for name, correct in result["fields"].items() if not correct],
                    [field],
                )
                self.assertFalse(result["all_four_correct"])

    def test_formula_order_and_reduced_composition_are_equivalent(self):
        response = self.response.replace("K2 Cu2 S2", "S2 K2 Cu2")
        self.assertTrue(self.grade(response)["fields"]["formula"])
        self.assertTrue(self.grade(formula="KCuS")["fields"]["formula"])

    def test_malformed_or_contradictory_facts_are_false(self):
        self.assertFalse(self.grade("The formula is not-a-formula.")["fields"]["formula"])
        contradictory = self.response + " The formula is K2 Cu2 O2."
        self.assertFalse(self.grade(contradictory)["fields"]["formula"])
        contradictory = self.response + " The unit cell contains 7 atoms."
        self.assertFalse(self.grade(contradictory)["fields"]["cell_atom_count"])

    def test_decimal_and_negative_numbers_are_not_truncated_to_integers(self):
        cases = (
            ("The space group number is 119.", "The space group number is 99.5.", "space_group", 99),
            ("The space group number is 119.", "The space group number is -3.", "space_group", 3),
            ("The unit cell contains 6 atoms", "The unit cell contains 99.5 atoms", "cell_atom_count", 99),
            ("The unit cell contains 6 atoms", "The unit cell contains -3 atoms", "cell_atom_count", 3),
            ("3 independent orbits", "99.5 independent orbits", "independent_orbit_count", 99),
            ("3 independent orbits", "-3 independent orbits", "independent_orbit_count", 3),
        )
        for original, replacement, field, target in cases:
            with self.subTest(replacement=replacement):
                response = self.response.replace(original, replacement)
                self.assertFalse(self.grade(response, **{
                    "space_group": target if field == "space_group" else 119,
                    "atom_count": target if field == "cell_atom_count" else 6,
                    "orbit_count": target if field == "independent_orbit_count" else 3,
                })["fields"][field])

    def test_summary_reports_per_field_and_all_four_rates_with_row_grades(self):
        rows = [
            {
                "response": self.response,
                "reference": {
                    "formula": "K2Cu2S2",
                    "space_group": 119,
                    "cell_atom_count": 6,
                    "independent_orbit_count": 3,
                },
            },
            {
                "response": self.response.replace("119", "120"),
                "reference": {
                    "formula": "K2Cu2S2",
                    "space_group": 119,
                    "cell_atom_count": 6,
                    "independent_orbit_count": 3,
                },
            },
        ]
        summary = summarize_understanding(rows)
        self.assertEqual(summary["fields"]["formula"], {"k": 2, "n": 2, "rate": 1.0})
        self.assertEqual(summary["fields"]["space_group"], {"k": 1, "n": 2, "rate": 0.5})
        self.assertEqual(summary["all_four"], {"k": 1, "n": 2, "rate": 0.5})
        self.assertEqual(len(summary["grades"]), 2)
        self.assertTrue(summary["grades"][0]["all_four_correct"])
        self.assertFalse(summary["grades"][1]["fields"]["space_group"])


if __name__ == "__main__":
    unittest.main()
