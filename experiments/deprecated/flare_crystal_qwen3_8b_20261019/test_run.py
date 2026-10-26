import importlib.util
from pathlib import Path
import random
import tempfile
import unittest


EXPERIMENT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("flare_pilot_run", EXPERIMENT / "run.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class PilotRunnerTests(unittest.TestCase):
    def test_selected_materials_are_seeded_and_updates_cycle_the_fixed_order(self):
        train_rows = [{"material_id": f"train-{index:02d}"} for index in range(12, 0, -1)]
        val_rows = [{"material_id": f"val-{index:02d}"} for index in range(12, 0, -1)]

        train, val = RUN.select_pilot_rows(train_rows, val_rows)

        self.assertEqual(train, random.Random(42).sample(
            sorted(train_rows, key=lambda row: row["material_id"]), 8))
        self.assertEqual(val, random.Random(42).sample(
            sorted(val_rows, key=lambda row: row["material_id"]), 8))
        self.assertEqual(RUN.materials_for_update(train, 1), train[:2])
        self.assertEqual(RUN.materials_for_update(train, 5), train[:2])
        self.assertEqual(RUN.materials_for_update(train, 4), train[6:8])

    def test_checkpoint_rejects_a_different_source_before_restoring_parameters(self):
        class FakeTorch:
            def load(self, path, map_location, weights_only):
                return {"schema": 1, "arm": "flare", "completed_updates": 64,
                        "source_trainable_sha256": "different-source"}

        class FakeTrain:
            torch = FakeTorch()

            @staticmethod
            def trainable_sha256(model):
                return "loaded-source"

        model = object()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            path.touch()
            with self.assertRaisesRegex(ValueError, "source trainable state differs"):
                RUN.load_checkpoint(model, path, FakeTrain(), expected_arm="flare")


if __name__ == "__main__":
    unittest.main()
