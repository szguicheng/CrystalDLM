import unittest
from experiments.crystal_tasks_v1.run_design_evaluation import merge_by_id, require_full_plan
from experiments.crystal_tasks_v1.evaluation.plan import make_plan


class DesignPipelineTests(unittest.TestCase):
    def test_formal_evaluation_rejects_a_truncated_attempt_plan(self):
        rows=[{'instruction':'Generate a crystal with space group number 1 and energy above hull no greater than 0.08 eV/atom.','input':''}]
        plan=make_plan(rows,[42,43,44],10)
        require_full_plan(plan,rows,[42,43,44],10)
        with self.assertRaises(ValueError):require_full_plan(plan[:-1],rows,[42,43,44],10)

    def test_merge_preserves_plan_order_and_rejects_dropped_or_duplicated_attempts(self):
        plan=[{'attempt_id':s} for s in ['a','b','c']]
        shards=[[{'attempt_id':'c'},{'attempt_id':'a'}],[{'attempt_id':'b'}]]
        self.assertEqual([r['attempt_id'] for r in merge_by_id(plan,shards)],['a','b','c'])
        with self.assertRaises(ValueError):merge_by_id(plan,shards[:1])
        with self.assertRaises(ValueError):merge_by_id(plan,shards+[shards[1]])


if __name__=='__main__':unittest.main()
