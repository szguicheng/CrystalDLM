import unittest
from pathlib import Path
from experiments.crystal_tasks_v1.train_ar import overfit_indices,run_config,design_overfit_indices


class TrainingEntryTests(unittest.TestCase):
    def test_longer_diagnostic_does_not_change_full_training_epochs(self):
        path=Path('/tmp/unit-only')
        self.assertEqual(run_config('ar_design','overfit',path,120)['num_train_epochs'],120)
        self.assertEqual(run_config('ar_design','full',path,120)['num_train_epochs'],3)

    def test_design_overfit_uses32distinct_requests_without_changing_rows(self):
        rows=[{'instruction':f'request-{i//2}','input':'','output':str(i)} for i in range(100)]
        selected=design_overfit_indices(rows)
        self.assertEqual(len(selected),32)
        self.assertEqual(len({rows[i]['instruction'] for i in selected}),32)
        self.assertEqual(selected,design_overfit_indices(rows))
        self.assertEqual(len(rows),100)

    def test_overfit_selection_is_fixed32_and_full_epochs_stay_three(self):
        ids=overfit_indices(36362)
        self.assertEqual(len(ids),32)
        self.assertEqual(len(set(ids)),32)
        self.assertEqual(ids,overfit_indices(36362))
        small=run_config('ar_understanding','overfit',Path('/tmp/unit-only'))
        full=run_config('ar_understanding','full',Path('/tmp/unit-only'))
        self.assertEqual(small['num_train_epochs'],60)
        self.assertEqual(full['num_train_epochs'],3)
        for k in ['learning_rate','lora_rank','lora_alpha','gradient_accumulation_steps','seed']:
            self.assertEqual(small[k],full[k])
        self.assertEqual(full['metric_for_best_model'],'eval_token_nll')
        self.assertNotIn('resume_from_checkpoint',full)


if __name__=='__main__':unittest.main()
