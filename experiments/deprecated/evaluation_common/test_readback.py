import unittest
from readback import grade_four_facts


class FourFacts(unittest.TestCase):
    def test_spaced_formula_and_full_cell_counts(self):
        text='The formula is K2 Cu2 S2. The space group number is 119. The unit cell contains 6 atoms and 3 independent orbits.<|im_end|>'
        self.assertTrue(grade_four_facts(text,'K2Cu2S2',119,6,3)['all_four_correct'])
        self.assertFalse(grade_four_facts(text,'K2Cu2O2',119,6,3)['fields']['formula'])

    def test_complete_formula_not_prefix_or_conflicting_statement(self):
        self.assertFalse(grade_four_facts('The formula is K2.','K2Cu2S2',119,6,3)['fields']['formula'])
        self.assertFalse(grade_four_facts('The formula is K2 Cu2 S2. The formula is K2 Cu2 O2.','K2Cu2S2',119,6,3)['fields']['formula'])
        self.assertTrue(grade_four_facts('The formula is Mg(OH)2.','MgO2H2',1,5,3)['fields']['formula'])

    def test_actual_space_group_and_count_errors_are_not_hidden(self):
        text='The formula is Ag18 Cu6. The space group number is 143. The unit cell contains 23 atoms and 2 independent orbits.'
        r=grade_four_facts(text,'Ag18Cu6',148,24,2)
        self.assertEqual(r['fields'],dict(formula=True,space_group=False,cell_atom_count=False,independent_orbit_count=True))


if __name__=='__main__':unittest.main()
