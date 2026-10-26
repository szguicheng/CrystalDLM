import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().with_name("run.py")
SPEC = importlib.util.spec_from_file_location("flare_crystal_full_run", PATH)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class FullTrainerTests(unittest.TestCase):
    @staticmethod
    def metadata(arm="flare", world_size=2):
        config = {"arm": arm, "training_world_size": world_size}
        return {
            "config": config,
            "config_sha256": RUN._json_sha256(config),
            "source_identity": {},
            "source_trainable_sha256": "checkpoint-source-hash",
            "source_checkpoint": {},
            "runtime_versions": {},
            "dataset_hashes": {},
            "epoch_order_sha256": "order-hash",
            "validation_ids_sha256": "val-hash",
            "source_sha256": {name: "code-hash" for name in
                               ("run.py", "model.py", "pilot_run.py", "pilot_model.py")},
            "arm": arm,
            "training_world_size": world_size,
        }

    def test_two_rank_partition_preserves_each_global_eight_material_update(self):
        rows = [{"material_id": f"m{index}"} for index in range(RUN.TRAIN_MATERIALS)]
        rank0 = RUN.partition_update(rows, update=1, rank=0, world_size=2)
        rank1 = RUN.partition_update(rows, update=1, rank=1, world_size=2)
        self.assertEqual([row["material_id"] for row in rank0], [f"m{i}" for i in range(4)])
        self.assertEqual([row["material_id"] for row in rank1], [f"m{i}" for i in range(4, 8)])
        self.assertEqual(rank0 + rank1, rows[:8])
        self.assertEqual(RUN.partition_update(rows, 2, 0, 1), rows[8:16])

    def test_checkpoint_rejects_wrong_arm_and_source_checkpoint(self):
        metadata = self.metadata()
        saved = {
            "schema": 1, "arm": "ar", "metadata": metadata,
            "metadata_sha256": RUN._json_sha256(metadata),
            "source_trainable_sha256": "checkpoint-source-hash",
            "completed_updates": 909, "world_size": 2,
            "rng_states": [{}, {}],
        }
        with self.assertRaisesRegex(ValueError, "schema or arm"):
            RUN.validate_checkpoint_identity(saved, "flare")

        saved["arm"] = "flare"

        class FakeTorch:
            @staticmethod
            def load(*_args, **_kwargs):
                return saved

            @staticmethod
            def no_grad():
                raise AssertionError("must reject before restoring parameters")

        class FakeTrain:
            torch = FakeTorch()

            @staticmethod
            def trainable_sha256(_model):
                return "fresh-source-hash"

        with self.assertRaisesRegex(ValueError, "source trainable state differs"):
            RUN.load_checkpoint(object(), "unused.pt", FakeTrain(), "flare")

    def test_resume_identity_requires_matching_code_data_and_world_size(self):
        metadata = {**self.metadata(), "data": "frozen-hash"}
        metadata["config"] = {**metadata["config"], "data": "frozen-hash"}
        metadata["config_sha256"] = RUN._json_sha256(metadata["config"])
        saved = {
            "schema": 1, "arm": "flare", "metadata": metadata,
            "metadata_sha256": RUN._json_sha256(metadata),
            "source_trainable_sha256": "checkpoint-source-hash",
            "completed_updates": 50, "world_size": 2,
            "rng_states": [{}, {}],
        }
        RUN.validate_checkpoint_identity(saved, "flare", metadata, 2, allow_partial=True)
        with self.assertRaisesRegex(ValueError, "world size differs"):
            RUN.validate_checkpoint_identity(saved, "flare", metadata, 1, allow_partial=True)
        changed = {**metadata, "data": "other-hash"}
        changed["config"] = {**metadata["config"], "data": "other-hash"}
        changed["config_sha256"] = RUN._json_sha256(changed["config"])
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            RUN.validate_checkpoint_identity(saved, "flare", changed, 2, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
