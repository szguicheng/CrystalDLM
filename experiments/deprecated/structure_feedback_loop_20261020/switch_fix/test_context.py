import unittest
from types import SimpleNamespace
from context import build_context


class RetryContextTests(unittest.TestCase):
    def test_failed_crystal_is_input_data_in_one_fresh_generation_request(self):
        item = {'prompt': 'User: Generate a crystal with space group number 194 and energy above hull no greater than 0.08 eV/atom.\nAssistant:', 'space_group_number': 194}
        previous = {'lead_token_ids': ['old explanation'], 'crystal_raw_token_ids': ['Mg2O2\nSpacegroup: P1\nSites (4)', '<CRYSTAL_END>']}
        tok = SimpleNamespace(decode=lambda ids, skip_special_tokens: ''.join(x for x in ids if not x.startswith('<')),
                              encode=lambda text, add_special_tokens: text)
        checker = SimpleNamespace(_checker_feedback=lambda row, sg: f'Requested {sg}; decoded 1; failed symmetry check.')
        text, feedback = build_context(item, previous, tok, checker)
        self.assertTrue(text.startswith('User: Generate a crystal with space group number 194'))
        self.assertEqual(text.count('User:'), 1)
        self.assertEqual(text.count('Assistant:'), 1)
        self.assertIn('Mg2O2\nSpacegroup: P1\nSites (4)', text)
        self.assertIn(feedback, text)
        self.assertNotIn('old explanation', text)
        self.assertNotIn('<CRYSTAL_START>', text)
        self.assertNotIn('<CRYSTAL_END>', text)

    def test_no_crystal_keeps_failed_ar_response_as_data(self):
        tok = SimpleNamespace(decode=lambda ids, skip_special_tokens: ''.join(ids),
                              encode=lambda text, add_special_tokens: text)
        checker = SimpleNamespace(_checker_feedback=lambda row, sg: 'No crystal START was emitted.')
        text, _ = build_context({'prompt':'User: Generate SG 1.\nAssistant:', 'space_group_number':1},
                               {'lead_token_ids':['unfinished previous response'], 'crystal_raw_token_ids':[]}, tok, checker)
        self.assertIn('unfinished previous response', text)
        self.assertIn('Previous failed response (input data):', text)


if __name__ == '__main__':unittest.main()
