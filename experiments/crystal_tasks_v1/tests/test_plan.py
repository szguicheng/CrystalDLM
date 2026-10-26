import unittest
from experiments.crystal_tasks_v1.evaluation.plan import make_plan


class PlanTests(unittest.TestCase):
    def test_requests_are_deduplicated_and_attempt_ids_are_complete(self):
        a={'instruction':'Generate a crystal with space group number 1 and energy above hull no greater than 0.08 eV/atom.','input':''}
        b={'instruction':'Generate a crystal with space group number 15 and energy above hull no greater than 0.1 eV/atom.','input':''}
        plan=make_plan([b,a,a],[42,43,44],10)
        self.assertEqual(len(plan),60)
        self.assertEqual(len({r['attempt_id'] for r in plan}),60)
        self.assertEqual(plan,make_plan([a,b],[42,43,44],10))
        self.assertEqual(plan[0]['request'],{'space_group':1,'ehull_upper':.08})


if __name__=='__main__':
    unittest.main()
