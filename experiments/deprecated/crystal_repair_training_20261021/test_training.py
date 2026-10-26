import importlib.util
from contextlib import nullcontext
import hashlib
import json
import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = load("run")
MODEL = load("model")


class Number:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return Number(self.value + float(other))

    __radd__ = __add__

    def __mul__(self, other):
        return Number(self.value * float(other))

    __rmul__ = __mul__


class FakeTensor:
    shape = (1,)
    dtype = "float32"
    device = "cpu"
    requires_grad = True

    def __init__(self, value):
        self.value = float(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def clone(self):
        return FakeTensor(self.value)

    def to(self, **_kwargs):
        return self

    def copy_(self, other):
        self.value = other.value


class FakeModel:
    def __init__(self):
        self.weight = FakeTensor(1)

    def named_parameters(self):
        return [("weight", self.weight)]


class FakeTrain:
    class Torch:
        @staticmethod
        def load(*_args, **_kwargs):
            return FakeTrain.saved

        @staticmethod
        def no_grad():
            return nullcontext()

    torch = Torch()

    @staticmethod
    def state_sha256(state):
        raw = json.dumps({name: value.value for name, value in sorted(state.items())},
                         sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def trainable_sha256(self, model):
        return self.state_sha256({name: value for name, value in model.named_parameters()
                                  if value.requires_grad})


class RepairTrainingTests(unittest.TestCase):
    def test_data_manifest_hashes_and_audit_contract_are_checked(self):
        def row(material_id):
            return {
                "material_id": material_id,
                "taskR": {"prefix": "User: request + corrupted plain-text crystal\nAssistant:",
                          "lead": " corrected fact\n<CRYSTAL_START>", "crystal": "valid target"},
                "original_tasks": [
                    {"task": "G", "prefix": "User: request\nAssistant:",
                     "lead": " verified facts\n<CRYSTAL_START>", "crystal": "valid target"},
                    {"task": "Q", "prefix": "User: read\nAssistant:", "target": "facts"},
                ],
                "targetaudit": {"valid": True},
                "corrupted_audit": {"valid": False},
                "corruption": {"source_row_sha256": "unused",
                               "target_crystal_sha256": "unused"},
            }

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            contents = {
                "train.jsonl": json.dumps(row("m1")) + "\n",
                "val.jsonl": json.dumps(row("m2")) + "\n",
            }
            hashes = {}
            for name, content in contents.items():
                raw = content.encode()
                (data / name).write_bytes(raw)
                hashes[name] = hashlib.sha256(raw).hexdigest()
            manifest = {"schema_version": 1,
                        "artifacts": {name: {"sha256": digest} for name, digest in hashes.items()},
                        "counts": {"output": {"train": 1, "val": 1}},
                        "source": {"immutable_manifest_sha256": "frozen-source-manifest"}}
            manifest_path = data / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            checked, train_rows, val_rows, _manifest = RUN.read_data(data)
            self.assertEqual([row["material_id"] for row in train_rows], ["m1"])
            self.assertEqual([row["material_id"] for row in val_rows], ["m2"])
            self.assertIn("manifest_sha256", checked)
            manifest["artifacts"]["train.jsonl"]["sha256"] = "changed"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "artifact hash differs"):
                RUN.read_data(data)
            bad_row = row("m1")
            bad_row["corrupted_audit"]["valid"] = True
            raw = (json.dumps(bad_row) + "\n").encode()
            (data / "train.jsonl").write_bytes(raw)
            manifest["artifacts"]["train.jsonl"]["sha256"] = hashlib.sha256(raw).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "valid target and failed-corruption audits"):
                RUN.read_data(data)

    def test_repair_rows_are_bound_to_their_frozen_split_source_tasks_and_crystal(self):
        source_train = [{"material_id": "m1", "plaid_text": "source crystal"}]
        source_val = [{"material_id": "m2", "plaid_text": "validation crystal"}]

        def source_tasks(source):
            prefix = f"User: Design {source['material_id']}\nAssistant:"
            return [
                {"task": "G", "prefix": prefix, "lead": " verified\n<CRYSTAL_START>",
                 "crystal": source["plaid_text"]},
                {"task": "Q", "prefix": f"{prefix} Read\nAssistant:", "target": "facts"},
            ]

        def repair_row(source):
            tasks = source_tasks(source)
            return {
                "material_id": source["material_id"],
                "taskR": {"prefix": (f"User: Design {source['material_id']}\n"
                                      "Previous failed crystal (input data):\ncorrupt\n"
                                      "Checker feedback:\nfailed\nAssistant:"),
                          "lead": " corrected\n<CRYSTAL_START>",
                          "crystal": source["plaid_text"]},
                "original_tasks": tasks,
                "targetaudit": {"valid": True},
                "corrupted_audit": {"valid": False},
                "corruption": {
                    "source_row_sha256": RUN._json_sha256(source),
                    "target_crystal_sha256": hashlib.sha256(
                        source["plaid_text"].encode()).hexdigest(),
                },
            }

        train_row, val_row = repair_row(source_train[0]), repair_row(source_val[0])
        RUN.validate_source_alignment([train_row], [val_row], source_train, source_val, source_tasks)
        with self.assertRaisesRegex(ValueError, "outside the frozen train split"):
            RUN.validate_source_alignment([repair_row(source_val[0])], [], source_train,
                                          source_val, source_tasks)
        train_row["taskR"]["crystal"] = "wrong target"
        with self.assertRaisesRegex(ValueError, "target differs from the original"):
            RUN.validate_source_alignment([train_row], [val_row], source_train,
                                          source_val, source_tasks)

    def test_raw_source_reader_checks_frozen_gzip_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val"):
                with gzip.open(root / f"{split}-triples.jsonl.gz", "wt", encoding="utf-8") as stream:
                    stream.write(json.dumps({"material_id": split, "split": split}) + "\n")
            hashes = {f"{split}_sha256": RUN._sha256_file(
                root / f"{split}-triples.jsonl.gz") for split in ("train", "val")}
            train_rows, val_rows = RUN.read_raw_source_splits(root, hashes)
            self.assertEqual(train_rows[0]["material_id"], "train")
            self.assertEqual(val_rows[0]["material_id"], "val")
            hashes["train_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "file changed"):
                RUN.read_raw_source_splits(root, hashes)

    def test_resume_at_the_completed_epoch_finalizes_the_checkpoint(self):
        self.assertTrue(RUN.needs_final_checkpoint(target=3, completed=3, total=3))
        self.assertFalse(RUN.needs_final_checkpoint(target=3, completed=2, total=3))
        with self.assertRaisesRegex(ValueError, "precedes"):
            RUN.needs_final_checkpoint(target=2, completed=3, total=3)
        with self.assertRaisesRegex(ValueError, "must exceed"):
            RUN.needs_final_checkpoint(target=2, completed=2, total=3)

    def test_repair_and_original_generation_reuse_flare_objective_and_keep_q_causal(self):
        repair = {"name": "R", "prefix": "fresh request + corrupted structure"}
        generation = {"name": "G", "task": "G", "prefix": "original generation"}
        reading = {"name": "Q", "task": "Q", "target": "verified facts"}
        calls = []
        values = {"R": Number(8), "G": Number(4)}

        def objective(_model, _tok, example, _generator, arm):
            calls.append((example, arm))
            return values[example["name"]], {"clean_ar": 2.0, "denoise": 1.0}

        class Reader:
            def loss(self, examples, _tok, mode, _generator):
                self.call = (examples, mode)
                return Number(12), {}

        reader = Reader()
        with patch.object(MODEL.PILOT, "objective", objective):
            loss, stats = MODEL.objective_tasks(
                reader, None, repair, [generation, reading], None)

        self.assertEqual(float(loss), 8.0)
        self.assertEqual(calls, [(repair, "flare"), (generation, "flare")])
        self.assertEqual(reader.call, ([reading], "ar"))
        self.assertEqual(stats["repair_joint"], 8.0)
        self.assertEqual(stats["generation_joint"], 4.0)
        self.assertEqual(stats["reading_ar"], 12.0)

    def test_three_rank_partition_covers_uneven_final_batch_once_with_equal_slots(self):
        rows = [{"material_id": f"m{i}"} for i in range(7)]
        slots = [RUN.partition_update(rows, 2, rank, 3) for rank in range(3)]

        self.assertEqual([len(rank) for rank in slots], [2, 2, 2])
        real = [index for rank in slots for index, _row in rank if index < len(rows)]
        self.assertEqual(real, [6])
        self.assertTrue(all(row is rows[0] for rank in slots
                            for index, row in rank if index >= len(rows)))
        self.assertEqual(RUN.actual_count(rows, 2), 1)
        self.assertEqual(RUN.gradient_scale(world_size=3, actual_count=1), 3.0)

    def test_rank_batches_cover_the_frozen_order_without_duplicates(self):
        rows = [{"material_id": f"m{i}"} for i in range(13)]
        observed = []
        for update in range(1, RUN.total_updates(len(rows)) + 1):
            for rank in range(3):
                observed.extend(index for index, _row in RUN.partition_update(
                    rows, update, rank, 3) if index < len(rows))
        self.assertEqual(observed, list(range(len(rows))))

    def test_rank_gpu_isolation_selects_only_its_physical_device(self):
        env = {"CUDA_VISIBLE_DEVICES": "0,1,2", "RANK": "2",
               "WORLD_SIZE": "3", "LOCAL_RANK": "2"}
        self.assertEqual(RUN.isolate_rank_gpu(env), (2, 3, 2))
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2")

    def test_repair_checkpoint_restores_only_after_parent_and_metadata_checks(self):
        metadata = {
            "config": {"total_updates": 2, "materials_per_update": 6,
                       "train_materials": 7},
            "arm": "repair",
            "training_world_size": 1,
            "source_identity": {},
            "pure_ar_trainable_sha256": "pure-ar-state",
            "parent_checkpoint_sha256": "parent-file",
            "parent_trainable_sha256": "parent-state",
            "parent_checkpoint_metadata_sha256": "parent-metadata",
            "parent_checkpoint_path": "parent.pt",
            "runtime_versions": {},
            "dataset_hashes": {"train_sha256": "train", "val_sha256": "val"},
            "train_material_ids": ["m0"],
            "validation_material_ids": ["v0"],
            "epoch_order_sha256": "order",
            "validation_ids_sha256": "validation",
            "source_sha256": {"run.py": "run", "model.py": "model"},
        }
        metadata["config_sha256"] = MODEL.json_sha256(metadata["config"])
        state = {"weight": FakeTensor(2)}
        FakeTrain.saved = {
            "schema": 1, "arm": "repair", "metadata": metadata,
            "metadata_sha256": MODEL.json_sha256(metadata),
            "parent_checkpoint_sha256": "parent-file",
            "parent_trainable_sha256": "parent-state",
            "completed_updates": 1, "total_updates": 2,
            "next_material_index": 6, "world_size": 1, "rng_states": [{}],
            "trainable_parameter_names": ["weight"], "trainable_state": state,
            "trainable_state_sha256": FakeTrain.state_sha256(state),
        }
        model, train = FakeModel(), FakeTrain()

        loaded = MODEL.load_checkpoint(
            model, "unused.pt", train, parent_checkpoint_sha256="parent-file",
            parent_trainable_sha256="parent-state", metadata=metadata,
            world_size=1, allow_partial=True)

        self.assertIs(loaded, FakeTrain.saved)
        self.assertEqual(model.weight.value, 2.0)
        with self.assertRaisesRegex(ValueError, "parent file hash differs"):
            MODEL.load_checkpoint(model, "unused.pt", train,
                                  parent_checkpoint_sha256="changed-file",
                                  parent_trainable_sha256="parent-state",
                                  metadata=metadata, world_size=1, allow_partial=True)


if __name__ == "__main__":
    unittest.main()
