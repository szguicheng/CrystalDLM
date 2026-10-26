import unittest
from sampling import remask_low, commit_best


class RevisionTest(unittest.TestCase):
    def test_wrong_early_prediction_can_be_replaced_but_trusted_one_stays(self):
        current = [11, 22, 0, 0]
        confidence = [.01, .99, -1., -1.]
        revised, scores, changed = remask_low(current, confidence, 0, .1)
        self.assertEqual(changed, [0])
        self.assertEqual(revised, [0, 22, 0, 0])
        result, _, committed = commit_best(revised, scores, [0, 2, 3], [33, 44, 55], [.98, .97, .96], 4)
        self.assertEqual(result, [33, 22, 44, 55])
        self.assertEqual(len(committed), 3)
        self.assertEqual(current, [11, 22, 0, 0])

    def test_commit_only_cannot_revise_existing_token_and_respects_budget(self):
        result, _, committed = commit_best([11, 0, 0], [.2, -1, -1], [1, 2], [22, 33], [.4, .9], 2)
        self.assertEqual(result, [11, 0, 33])
        self.assertEqual(committed, [2])


if __name__ == '__main__':
    unittest.main()
