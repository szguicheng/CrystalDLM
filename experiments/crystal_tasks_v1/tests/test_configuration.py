import unittest
from unittest.mock import patch
from experiments.crystal_tasks_v1.configuration import protocol, training_config, task_data


class ConfigurationTests(unittest.TestCase):
    def test_three_gpu_effective_batch_and_design_matching(self):
        p = protocol()
        t = p['training']
        self.assertEqual((t['world_size'], t['global_batch_size']), (3, 12))
        self.assertEqual(t['world_size'] * t['per_device_train_batch_size'] * t['gradient_accumulation_steps'], 12)
        a, h = training_config('ar_design'), training_config('hybrid_design')
        a.pop('output_dir'); h.pop('output_dir')
        self.assertEqual(a, h)
        self.assertEqual(a['num_train_epochs'], 3)
        self.assertFalse(a['train_on_prompt'])

    def test_completed_design_release_does_not_change_understanding_source(self):
        p=protocol()
        original=task_data('ar_understanding')
        for name in ('ar_design','hybrid_design'):
            p['tasks'][name].update(dataset_dir='/project/design-release',manifest_sha256='release-sha')
        with patch('experiments.crystal_tasks_v1.configuration.protocol',return_value=p):
            self.assertEqual(task_data('ar_understanding'),original)
            self.assertEqual(task_data('ar_design'),task_data('hybrid_design'))
            self.assertEqual(training_config('ar_design')['dataset_dir'],'/project/design-release')

    def test_answer_only_and_original_base(self):
        config = training_config('ar_understanding')
        self.assertNotIn('adapter_name_or_path', config)
        self.assertNotIn('max_samples', config)
        self.assertEqual(config['trainable_token_indices'], [151669,151670,151671,151672])
        self.assertEqual(config['cutoff_len'], 1024)
        self.assertFalse(config['enable_thinking'])


if __name__ == '__main__':
    unittest.main()
