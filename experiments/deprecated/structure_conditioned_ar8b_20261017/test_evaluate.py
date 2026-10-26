import importlib.util
from pathlib import Path
import unittest


EXPERIMENT = Path(__file__).resolve().parent


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StructuralEvaluationPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluate = load_script("conditioned_evaluate", EXPERIMENT / "evaluate.py")
        cls.original = load_script(
            "original_structure_request_runner",
            EXPERIMENT / "reference_source_snapshots/structure_request_eval_run.py")

    def test_fixed_700_plan_changes_prompts_and_preserves_seed_schedule(self):
        plan = self.evaluate.RUNNER.attempt_plan()
        original_plan = self.original.attempt_plan()
        self.assertEqual(len(plan), 700)
        self.assertEqual([item["seed"] for item in plan],
                         [item["seed"] for item in original_plan])
        self.assertEqual([item["space_group_number"] for item in self.evaluate.RUNNER.REQUESTS],
                         [1, 15, 38, 119, 143, 194, 216])
        self.assertEqual(
            plan[0]["prompt"],
            "User: Generate a crystal with space group number 1 and energy above hull no "
            "greater than 0.08 eV/atom.\nAssistant:")
        self.assertEqual(
            plan[100]["prompt"],
            "User: Generate a crystal with space group number 15 and energy above hull no "
            "greater than 0.08 eV/atom.\nAssistant:")
        self.assertIn("space-group number", self.evaluate.RUNNER.TRAINING_PROMPT_SCOPE)
        self.assertIn("GRACE scoring", self.evaluate.RUNNER.EVALUATION_SCOPE)


if __name__ == "__main__":
    unittest.main()
