import hashlib
import random
import tempfile
import types
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from continuation import draw_noise_step, load_continuation, select_noise_step


class ContinuationTests(unittest.TestCase):
    def test_control_and_highmask_use_paired_uniform_draws(self):
        if torch is None:
            rng_control, rng_high = random.Random(42), random.Random(42)
            draws = [(rng_control.randint(1, 32), rng_control.random())
                     for _ in range(4096)]
            self.assertEqual(draws, [(rng_high.randint(1, 32), rng_high.random())
                                     for _ in range(4096)])
            control = [select_noise_step("control", k, u) for k, u in draws]
            highmask = [select_noise_step("highmask", k, u) for k, u in draws]
        else:
            control_generator = torch.Generator(device="cpu").manual_seed(42)
            high_generator = torch.Generator(device="cpu").manual_seed(42)
            control_draws = [draw_noise_step("control", control_generator)
                             for _ in range(4096)]
            high_draws = [draw_noise_step("highmask", high_generator)
                          for _ in range(4096)]
            self.assertEqual(
                [(row["uniform_k"], row["branch_u"]) for row in control_draws],
                [(row["uniform_k"], row["branch_u"]) for row in high_draws])
            control = [row["k"] for row in control_draws]
            highmask = [row["k"] for row in high_draws]
            draws = [(row["uniform_k"], row["branch_u"]) for row in control_draws]

        self.assertEqual(control, [k for k, _ in draws])
        self.assertEqual(highmask, [32 if u < 0.5 else k for k, u in draws])
        self.assertAlmostEqual(sum(k == 32 for k in control) / len(control), 1 / 32, delta=0.01)
        self.assertAlmostEqual(sum(k == 32 for k in highmask) / len(highmask),
                               0.515625, delta=0.02)
        self.assertEqual(select_noise_step("highmask", 7, 0.5), 7)
        self.assertEqual(select_noise_step("highmask", 7, 0.499999), 32)

    @unittest.skipUnless(torch is not None, "PyTorch is unavailable in this local environment")
    def test_loader_checks_source_then_restores_the_complete_trainable_state(self):
        from torch import nn

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([1.0]))

        def state_sha256(state):
            digest = hashlib.sha256()
            for name, tensor in sorted(state.items()):
                digest.update(name.encode() + b"\0")
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        train = types.SimpleNamespace(
            torch=torch,
            trainable_state=lambda model: {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters() if param.requires_grad
            },
            state_sha256=state_sha256,
            trainable_sha256=lambda model: state_sha256({
                name: param.detach().cpu() for name, param in model.named_parameters()
                if param.requires_grad
            }),
        )
        model = TinyModel()
        source_hash = train.trainable_sha256(model)
        final_state = {"weight": torch.tensor([2.0])}
        checkpoint = {
            "schema": 1,
            "completed_updates": 128,
            "source_trainable_sha256": source_hash,
            "trainable_parameter_names": ["weight"],
            "trainable_state": final_state,
            "trainable_state_sha256": state_sha256(final_state),
            "metadata": {"arm": "control", "test": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            metadata = load_continuation(model, path, train)

        self.assertEqual(float(model.weight), 2.0)
        self.assertEqual(metadata, checkpoint["metadata"])


if __name__ == "__main__":
    unittest.main()
