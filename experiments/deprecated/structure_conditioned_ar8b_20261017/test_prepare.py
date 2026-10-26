import copy
import importlib.util
from pathlib import Path
import unittest


class StructuralPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "structural_prepare", Path(__file__).with_name("prepare.py"))
        cls.prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.prepare)

    def row(self, energy):
        return {"property_labels": {"energy_above_hull": energy, "unit": "eV/atom"},
                "structure_description": "A hexagonal crystal.", "plaid_text": "reference crystal",
                "core_facts": {"space_group_number": 194, "cell_formula": "Mg2",
                               "cell_atom_count": 2, "independent_orbit_count": 1}}

    def test_bound_covers_label_without_mislabeling_high_energy(self):
        for energy, bound in [(0, .08), (.08, .08), (.08001, .1), (.1, .1),
                              (.10001, .2), (.2, .2), (.21, .3), (.3, .3)]:
            self.assertEqual(self.prepare.ehull_upper_bound(energy), bound)

    def test_both_tasks_use_new_condition_and_preserve_targets_and_source(self):
        row = self.row(.045)
        original = copy.deepcopy(row)
        old = self.prepare.SOURCE.tasks(row)
        new = self.prepare.tasks(row)
        prefix = ("User: Generate a crystal with space group number 194 and "
                  "energy above hull no greater than 0.08 eV/atom.\nAssistant:")
        self.assertEqual(new[0]["prefix"], prefix)
        self.assertEqual(new[1]["prefix"], prefix + old[1]["prefix"][len(old[0]["prefix"]):])
        self.assertEqual(new[0]["crystal"], old[0]["crystal"])
        self.assertEqual(new[0]["lead"], old[0]["lead"])
        self.assertEqual(new[1]["target"], old[1]["target"])
        self.assertEqual(row, original)


if __name__ == "__main__":
    unittest.main()
