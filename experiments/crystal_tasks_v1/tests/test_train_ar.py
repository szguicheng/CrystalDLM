import unittest
from pathlib import Path
from experiments.crystal_tasks_v1.train_ar import overfit_indices,run_config


class TrainingEntryTests(unittest.TestCase):
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
