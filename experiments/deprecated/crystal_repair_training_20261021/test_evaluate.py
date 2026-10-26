"""CPU contract checks for the frozen repair evaluation plan."""
import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).with_name("evaluate.py")
SPEC = importlib.util.spec_from_file_location("crystal_repair_evaluate_test", PATH)
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


def _row(material_id, kind):
    corrupted = f"corrupted input for {material_id}\n"
    target = f"reference target for {material_id}"
    prefix = ("User: Repair this structure.\n" + EVALUATE.INPUT_MARKER + corrupted +
              EVALUATE.FEEDBACK_MARKER + "The strict audit failed.\nAssistant:")
    return {
        "material_id": material_id,
        "taskR": {"prefix": prefix, "lead": "REFERENCE LEAD MUST NOT ENTER CONTEXT",
                  "crystal": target},
        "original_tasks": [
            {"task": "G", "lead": "REFERENCE G LEAD MUST NOT ENTER CONTEXT",
             "crystal": target},
            {"task": "Q", "target": "REFERENCE Q TARGET MUST NOT ENTER CONTEXT"},
        ],
        "corruption": {
            "type": kind,
            "source_row_sha256": EVALUATE._sha256_text(f"source-{material_id}"),
            "corrupted_crystal_sha256": EVALUATE._sha256_text(corrupted),
            "target_crystal_sha256": EVALUATE._sha256_text(target),
        },
        "targetaudit": {"valid": True, "facts": {"sg": 14}},
    }


class RepairEvaluationPlanTests(unittest.TestCase):
    def setUp(self):
        self.rows = [_row(f"mat-{kind}-{index:02d}", kind)
                     for kind in EVALUATE.CORRUPTION_TYPES for index in range(10)]

    def test_selection_is_deterministic_balanced_and_train_disjoint(self):
        first = EVALUATE._build_selection_manifest(self.rows, ["train-only"], {"val_sha256": "v"})
        second = EVALUATE._build_selection_manifest(
            list(reversed(self.rows)), ["train-only"], {"val_sha256": "v"})
        self.assertEqual(first, second)
        self.assertEqual(len(first["items"]), 32)
        self.assertEqual(
            {kind: sum(item["corruption_type"] == kind for item in first["items"])
             for kind in EVALUATE.CORRUPTION_TYPES},
            {kind: 8 for kind in EVALUATE.CORRUPTION_TYPES},
        )
        with self.assertRaisesRegex(ValueError, "need eight actual validation examples"):
            EVALUATE._build_selection_manifest(self.rows[:-3], ["train-only"], {})
        with self.assertRaisesRegex(ValueError, "overlap"):
            EVALUATE._build_selection_manifest(self.rows, [first["items"][0]["material_id"]], {})

    def test_model_item_uses_only_repair_prefix(self):
        selection = EVALUATE._build_selection_manifest(self.rows, [], {})
        selected = selection["items"][0]
        row = next(row for row in self.rows if row["material_id"] == selected["material_id"])
        item = EVALUATE._evaluation_item(row, selected)
        self.assertEqual(item["prompt"], row["taskR"]["prefix"])
        self.assertIn("corrupted input", item["prompt"])
        self.assertNotIn("REFERENCE LEAD", item["prompt"])
        self.assertNotIn("reference target", item["prompt"])
        self.assertNotIn("REFERENCE Q TARGET", item["prompt"])
        self.assertEqual(item["seed"], selected["seed"])

    def test_copy_diagnostic_ignores_outer_newline_but_detects_changed_body(self):
        self.assertTrue(EVALUATE._outer_whitespace_normalized_body_match(
            "copied crystal\n", "copied crystal"))
        self.assertFalse(EVALUATE._outer_whitespace_normalized_body_match(
            "different crystal\n", "copied crystal"))

    def test_checkpoint_dataset_identity_uses_shared_hashes_and_split_ids(self):
        hashes = {key: key for key in EVALUATE.DATASET_HASH_KEYS}
        metadata = {
            "dataset_hashes": {**hashes, "frozen_source_hashes": {"train_sha256": "extra"}},
            "train_material_ids": ["train-a", "train-b"],
            "validation_material_ids": ["val-a", "val-b"],
        }
        EVALUATE._validate_checkpoint_dataset(
            metadata, hashes, ["train-a", "train-b"], ["val-a", "val-b"], ["val-a"])
        with self.assertRaisesRegex(ValueError, "dataset hashes"):
            EVALUATE._validate_checkpoint_dataset(
                metadata, {**hashes, "val_sha256": "changed"},
                ["train-a", "train-b"], ["val-a", "val-b"], ["val-a"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            EVALUATE._validate_checkpoint_dataset(
                metadata, hashes, ["train-a", "train-b"], ["val-a", "val-b"], ["train-a"])


if __name__ == "__main__":
    unittest.main()
