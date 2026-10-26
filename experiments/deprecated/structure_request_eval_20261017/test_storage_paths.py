import importlib.util
from pathlib import Path
import tempfile
import unittest


RUNNER = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("structure_request_eval_storage_paths", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class StoragePathIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "data1" / "crystalDLM"
        self.alias = Path(self.tmp.name) / "home" / "crystalDLM"
        for name in ("data", "model", "experiment", "sources"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.alias.parent.mkdir(parents=True)
        self.alias.symlink_to(self.root, target_is_directory=True)
        for name in ("train.py", "model.py"):
            (self.root / "sources" / name).write_text(name, encoding="utf-8")

    def identities(self):
        actual_paths = {
            "data_path": self.root / "data",
            "model_path": self.root / "model",
            "experiment_path": self.root / "experiment",
        }
        recorded_paths = {
            key: self.alias / path.relative_to(self.root)
            for key, path in actual_paths.items()
        }
        actual_sources = {
            name: self.root / "sources" / name for name in ("train.py", "model.py")
        }
        recorded_sources = {
            name: self.alias / path.relative_to(self.root)
            for name, path in actual_sources.items()
        }

        def identity(paths, sources):
            value = {
                "model": "Qwen3-8B",
                "model_config": {"_name_or_path": str(recorded_paths["model_path"])},
                "model_config_sha256": "same-config",
                "data_hashes": {"train_sha256": "train-a", "val_sha256": "val-a"},
                **{key: str(path) for key, path in paths.items()},
                "source_paths": {key: str(path) for key, path in sources.items()},
            }
            value["fingerprint_sha256"] = runner.json_sha256(
                {key: item for key, item in value.items() if key != "fingerprint_sha256"})
            return value

        return identity(actual_paths, actual_sources), identity(recorded_paths, recorded_sources)

    def test_accepts_samefile_aliases_and_recomputes_the_archived_fingerprint(self):
        actual, recorded = self.identities()

        model_path = runner._samefile_alias(
            self.root / "model", recorded["model_path"], "model_path")
        normalized = runner._normalize_identity_paths(actual, recorded)

        self.assertEqual(model_path, recorded["model_path"])
        self.assertEqual(normalized, recorded)

    def test_rejects_a_different_model_directory(self):
        actual, recorded = self.identities()
        other_model = self.root / "other-model"
        other_model.mkdir()

        with self.assertRaisesRegex(ValueError, "model_path"):
            runner._samefile_alias(other_model, recorded["model_path"], "model_path")

    def test_rejects_changed_model_config(self):
        actual, recorded = self.identities()
        actual["model_config_sha256"] = "changed-config"

        with self.assertRaisesRegex(ValueError, "does not match"):
            runner._normalize_identity_paths(actual, recorded)

    def test_rejects_changed_data_hashes(self):
        actual, recorded = self.identities()
        actual["data_hashes"]["train_sha256"] = "changed-data"

        with self.assertRaisesRegex(ValueError, "does not match"):
            runner._normalize_identity_paths(actual, recorded)


if __name__ == "__main__":
    unittest.main()
