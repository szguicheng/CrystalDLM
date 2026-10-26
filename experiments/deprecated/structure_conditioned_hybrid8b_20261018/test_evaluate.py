import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


EXPERIMENT = Path(__file__).resolve().parent


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HybridEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluate = load_script("conditioned_hybrid_evaluate", EXPERIMENT / "evaluate.py")
        cls.ar_evaluate = load_script(
            "matched_ar_evaluate_for_hybrid_test",
            EXPERIMENT.parent / "structure_conditioned_ar8b_20261017/evaluate.py")

    def test_requests_and_per_attempt_seed_plan_match_the_ar_run(self):
        plan = self.evaluate.RUNNER.attempt_plan()
        ar_plan = self.ar_evaluate.RUNNER.attempt_plan()
        self.assertEqual(len(plan), 700)
        self.assertEqual(self.evaluate.REQUESTS, self.ar_evaluate.RUNNER.REQUESTS)
        self.assertEqual([item["seed"] for item in plan], [item["seed"] for item in ar_plan])
        self.assertEqual(self.evaluate.DIFFUSION_STEPS, 32)
        self.assertEqual(self.evaluate.CRYSTAL_CAPACITY, 544)
        self.assertIn("hybrid", self.evaluate.TRAINING_PROMPT_SCOPE)

    def test_full_mask_schedule_releases_tokens_once_and_keeps_known_positions_fixed(self):
        mask_id = 99_999
        current = [mask_id] * self.evaluate.CRYSTAL_CAPACITY
        predictions = [100 + index for index in range(len(current))]
        schedule = list(range(self.evaluate.DIFFUSION_STEPS, 0, -1))

        for step_index, k in enumerate(schedule):
            masked = [index for index, token_id in enumerate(current) if token_id == mask_id]
            step_predictions = [predictions[index] for index in masked]
            draws = [0.0] * len(masked)
            if step_index == 0:
                draws[0] = 1.0 / k
            previous = current
            current = self.evaluate._apply_releases(
                current, masked, step_predictions, draws, mask_id, k)
            for index, token_id in enumerate(previous):
                if token_id != mask_id:
                    self.assertEqual(current[index], token_id)
            if step_index == 0:
                self.assertEqual(current[0], mask_id)
                self.assertEqual(current[1], predictions[1])

        self.assertEqual(schedule, list(range(32, 0, -1)))
        self.assertNotIn(mask_id, current)
        self.assertEqual(current[0], predictions[0])

    @unittest.skipIf(torch is None, "PyTorch is available in the project training environment")
    def test_denoiser_sees_full_bidirectional_span_and_predicts_masked_positions(self):
        class TinyHybrid:
            mask_id = 15

            def __init__(self, capacity):
                self.capacity = capacity
                self.time = SimpleNamespace(weight=SimpleNamespace(device=torch.device("cpu")))
                self.calls = []

            def logits(self, ids, mode, span, k, positions):
                current = ids[span[0]:span[1]].tolist()
                self.calls.append({"mode": mode, "span": span, "k": k,
                                   "positions": list(positions), "prefix": ids[:span[0]].tolist(),
                                   "current": current})
                self.assertions.assertEqual(len(current), self.capacity)
                self.assertions.assertEqual(mode, "hybrid")
                logits = torch.full((len(positions), 16), -100.0)
                for row, position in enumerate(positions):
                    logits[row, (position - span[0]) % 4 + 1] = 100.0
                return logits

        capacity = self.evaluate.CRYSTAL_CAPACITY
        prefix = [3, 4, 5]
        model = TinyHybrid(capacity)
        model.assertions = self
        current, forwards = self.evaluate._diffuse_crystal(
            model, prefix, capacity, self.evaluate.DIFFUSION_STEPS,
            torch.Generator(device="cpu").manual_seed(42), {0, model.mask_id}, .8, .95)

        self.assertEqual(forwards, self.evaluate.DIFFUSION_STEPS)
        self.assertEqual([call["k"] for call in model.calls],
                         list(range(self.evaluate.DIFFUSION_STEPS, 0, -1)))
        for call in model.calls:
            expected = [index + len(prefix) for index, token_id in enumerate(call["current"])
                        if token_id == model.mask_id]
            self.assertEqual(call["positions"], expected)
            self.assertEqual(call["span"], (len(prefix), len(prefix) + capacity))
            self.assertEqual(call["prefix"], prefix)
        for previous, following in zip(model.calls, model.calls[1:]):
            for before, after in zip(previous["current"], following["current"]):
                if before != model.mask_id:
                    self.assertEqual(after, before)
        self.assertNotIn(model.mask_id, current)
        self.assertEqual(current, [(index % 4) + 1 for index in range(capacity)])

    def test_invalid_end_pad_boundary_is_preserved_without_repair(self):
        class Model:
            end_id = 8
            pad_id = 9

        class Tokenizer:
            @staticmethod
            def decode(token_ids, skip_special_tokens=False):
                return str(token_ids)

        raw = [4, Model.end_id, 5, Model.pad_id]
        original = list(raw)
        with self.assertRaises(self.evaluate.RUNNER.GenerationFailure) as raised:
            self.evaluate.RUNNER._boundary_content(raw, Model(), Tokenizer())
        self.assertEqual(raw, original)
        self.assertEqual(raised.exception.token_ids, original)


if __name__ == "__main__":
    unittest.main()
