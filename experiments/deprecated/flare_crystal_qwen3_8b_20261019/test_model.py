import unittest

from model import layout


class VisibilityTests(unittest.TestCase):
    def test_noisy_targets_are_masked_and_cannot_read_clean_answers(self):
        clean = [10, 11, 12, 13, 14, 15, 16]
        plan = layout(clean, 2, [True, False, True, False, True], 99, 4)
        self.assertEqual(plan['labels'], [12, 14, 16])
        for query, target in zip(plan['queries'], plan['target_positions']):
            block_start = 2 + (target - 2) // 4 * 4
            reachable = {query}
            for _ in range(len(plan['ids'])):
                reachable |= {k for q in list(reachable) for k, ok in enumerate(plan['allowed'][q]) if ok}
            self.assertNotIn(target, reachable, 'clean target leaks through attention')
            self.assertTrue(all(k < block_start for k in reachable if k < len(clean)))
            self.assertEqual(plan['ids'][query + 1], 99)
            self.assertEqual(plan['positions'][query], target - 1)
        self.assertEqual(plan['allowed'][0], [True] + [False] * (len(plan['ids']) - 1))

    def test_complementary_views_supervise_every_crystal_token_once(self):
        clean = [10, 11, 12, 13, 14, 15, 16]
        mask = [True, False, True, False, True]
        a = layout(clean, 2, mask, 99, 4)
        b = layout(clean, 2, [not v for v in mask], 99, 4)
        self.assertEqual(sorted(a['target_positions'] + b['target_positions']), list(range(2, 7)))
        self.assertTrue(set(a['target_positions']).isdisjoint(b['target_positions']))


if __name__ == '__main__':
    unittest.main()
