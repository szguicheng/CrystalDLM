import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


EXPERIMENT = Path(__file__).resolve().parent


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 4

    def __init__(self):
        self.ids = {token: index for index, token in enumerate(
            ("<CRYSTAL_START>", "<CRYSTAL_END>", "<CRYSTAL_MASK>",
             "<CRYSTAL_PAD>", self.eos_token))}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def encode(self, text, **kwargs):
        return list(text)


def row():
    return {
        "material_id": "mp-test",
        "split": "train",
        "plaid_text": "NaCl\nSpacegroup: P63/mmc",
        "structure_description": "NaCl is hexagonal in space group P63/mmc.",
        "property_labels": {"energy_above_hull": 0.09, "unit": "eV/atom"},
        "core_facts": {
            "cell_formula": "Na4 Cl4", "space_group_number": 194,
            "cell_atom_count": 8, "independent_orbit_count": 2,
        },
    }


class ConditionedTrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_script("conditioned_train", EXPERIMENT / "train.py")
        cls.library = cls.train.load_training_module()
        cls.prepare = load_script("conditioned_prepare", EXPERIMENT / "prepare.py")

    def test_training_and_length_validation_use_new_conditioned_tasks(self):
        self.assertEqual(self.library.tasks(row()), self.prepare.tasks(row()))
        self.assertIs(self.library.BASELINE.tasks, self.library.tasks)
        self.assertIs(self.library.BASELINE.validate_lengths.__globals__["tasks"],
                      self.library.tasks)

        generation, reading = self.library.tasks(row())
        self.assertEqual(
            generation["prefix"],
            "User: Generate a crystal with space group number 194 and energy above hull "
            "no greater than 0.1 eV/atom.\nAssistant:")
        self.assertEqual(generation["lead"],
                         " NaCl is hexagonal in space group P63/mmc.\n<CRYSTAL_START>")
        self.assertEqual(generation["crystal"], "NaCl\nSpacegroup: P63/mmc")
        self.assertEqual(reading["target"],
                         "The formula is Na4 Cl4. The space group number is 194. "
                         "The unit cell contains 8 atoms and 2 independent orbits.<|im_end|>")
        lengths = self.library.BASELINE.validate_lengths([row()], CharacterTokenizer())
        self.assertEqual(lengths["generation_max_tokens"],
                         len(generation["prefix"] + generation["lead"]) + self.prepare.K)

    def test_checkpoint_identity_binds_new_sources_data_and_canonical_paths(self):
        with tempfile.TemporaryDirectory(dir=EXPERIMENT) as temporary:
            root = Path(temporary)
            model_path, data_path = root / "model", root / "data"
            model_path.mkdir()
            data_path.mkdir()
            (model_path / "config.json").write_text(json.dumps({"hidden_size": 4}))
            (model_path / "weights.bin").write_bytes(b"fresh base fixture")
            base_config = {"hidden_size": 4}
            order_ids, eval_ids = ["mp-a", "mp-b"], ["mp-val"]
            data_hashes = {"manifest_sha256": "manifest-a", "train_sha256": "train-a",
                           "val_sha256": "val-a"}
            first = self.library.finalize_identity(
                self.library.experiment_identity(
                    model_path, self.library.REVISION, base_config, data_hashes,
                    order_ids, eval_ids),
                data_path, model_path, CharacterTokenizer())
            changed = dict(data_hashes, train_sha256="train-b")
            second = self.library.finalize_identity(
                self.library.experiment_identity(
                    model_path, self.library.REVISION, base_config, changed,
                    order_ids, eval_ids),
                data_path, model_path, CharacterTokenizer())

            self.assertNotEqual(first["fingerprint_sha256"],
                                second["fingerprint_sha256"])
            self.assertEqual(first["experiment"], EXPERIMENT.name)
            self.assertEqual(first["experiment_path"], str(EXPERIMENT.resolve()))
            self.assertEqual(first["data_path"], str(data_path.resolve()))
            self.assertEqual(first["model_path"], str(model_path.resolve()))
            self.assertIn("prepare.py", first["source_sha256"])
            self.assertIn("reference_train.py", first["source_sha256"])
            self.assertIn("reference_model.py", first["source_sha256"])
            self.assertIn("config.json", first["source_sha256"])


if __name__ == "__main__":
    unittest.main()
