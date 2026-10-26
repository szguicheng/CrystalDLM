import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from experiments.crystal_tasks_v1.evaluation.cli import (
    _attempt_hashes,
    _join_attempt_plan,
    _join_predictions,
    _join_property_receipts,
    _verify_inline_property_receipts,
)
from experiments.crystal_tasks_v1.evaluation.grace import (
    _append_receipt,
    _finish_run_meta,
    _receipt,
)
from experiments.crystal_tasks_v1.evaluation.references import (
    decode_reference_rows,
    load_task_rows,
    load_split_rows,
)


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(root):
    data = [
        {"instruction": "read", "input": "<CRYSTAL_START>Na<CRYSTAL_END>", "output": "answer a"},
        {"instruction": "read", "input": "<CRYSTAL_START>K<CRYSTAL_END>", "output": "answer b"},
    ]
    metadata = [
        {"row_index": 0, "source_split": "test", "material_id": "mp-a",
         "core_facts": {"cell_formula": "Na", "space_group_number": 221,
                        "cell_atom_count": 1, "independent_orbit_count": 1}},
        {"row_index": 1, "source_split": "test", "material_id": "mp-b",
         "core_facts": {"cell_formula": "K", "space_group_number": 221,
                        "cell_atom_count": 1, "independent_orbit_count": 1}},
    ]
    artifacts = {
        "data/understanding/test.jsonl": {
            "sha256": _jsonl(root / "understanding/test.jsonl", data),
        },
        "data/metadata/test.jsonl": {
            "sha256": _jsonl(root / "metadata/test.jsonl", metadata),
        },
    }
    (root / "manifest.json").write_text(json.dumps({
        "counts_per_task": {"train": 2, "val": 2, "test": 2}, "artifacts": artifacts,
    }), encoding="utf-8")
    return data, metadata


class EvaluationIOTests(TestCase):
    def test_task_row_count_must_match_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _dataset(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["counts_per_task"]["test"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count mismatch"):
                load_task_rows(root, task="understanding", split="test")

    def test_manifest_and_order_are_checked_for_both_split_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _ = _dataset(root)

            rows = load_split_rows(root, "test")

            self.assertEqual([row["metadata"]["material_id"] for row in rows], ["mp-a", "mp-b"])
            self.assertEqual(rows[0]["input"], "<CRYSTAL_START>Na<CRYSTAL_END>")
            self.assertEqual(load_task_rows(root, task="understanding", split="test"), data)
            rows[1]["metadata"]["row_index"] = 0
            meta_path = root / "metadata/test.jsonl"
            _jsonl(meta_path, [row["metadata"] for row in rows])
            with self.assertRaisesRegex(ValueError, "sha256"):
                load_split_rows(root, "test")
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["data/metadata/test.jsonl"]["sha256"] = _jsonl(
                meta_path, [row["metadata"] for row in rows]
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "order mismatch"):
                load_split_rows(root, "test")

    def test_predictions_join_by_material_id_and_reject_missing_or_duplicate_ids(self):
        rows = [
            {"metadata": {"material_id": "mp-a", "core_facts": {
                "cell_formula": "Na", "space_group_number": 221,
                "cell_atom_count": 1, "independent_orbit_count": 1,
            }}},
            {"metadata": {"material_id": "mp-b", "core_facts": {
                "cell_formula": "K", "space_group_number": 221,
                "cell_atom_count": 1, "independent_orbit_count": 1,
            }}},
        ]
        joined = _join_predictions(
            rows,
            [{"material_id": "mp-b", "response": "B"}, {"material_id": "mp-a", "response": "A"}],
        )
        self.assertEqual([row["response"] for row in joined], ["A", "B"])
        with self.assertRaisesRegex(ValueError, "missing"):
            _join_predictions(rows, [{"material_id": "mp-a", "response": "A"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _join_predictions(rows, [
                {"material_id": "mp-a", "response": "A"},
                {"material_id": "mp-a", "response": "A2"},
            ])

    def test_property_receipts_join_by_attempt_id_not_position(self):
        attempts = [
            {"attempt_id": "a", "seed": 11, "output": "Na",
             "request": {"space_group": 1, "ehull_upper": 0.08}},
            {"attempt_id": "b", "seed": 12, "output": "K",
             "request": {"space_group": 2, "ehull_upper": 0.1}},
        ]
        joined = _join_property_receipts(attempts, [
            {"attempt_id": "b", "seed": 12, **_attempt_hashes(attempts[1]),
             "property_receipt": {"score_status": "scored"}},
            {"attempt_id": "a", "seed": 11, **_attempt_hashes(attempts[0]),
             "property_receipt": {"score_status": "error"}},
        ])
        self.assertEqual(
            [attempt["property_receipt"]["score_status"] for attempt in joined],
            ["error", "scored"],
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            _join_property_receipts(attempts, [{
                "attempt_id": "a", "seed": 11, **_attempt_hashes(attempts[0]),
                "property_receipt": {"score_status": "error"},
            }])
        wrong_output = {"attempt_id": "a", "seed": 11, **_attempt_hashes(attempts[0]),
                        "property_receipt": {"score_status": "error"}}
        wrong_output["output_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "output_sha256 mismatch"):
            _join_property_receipts([attempts[0]], [wrong_output])
        wrong_request = {"attempt_id": "a", "seed": 11, **_attempt_hashes(attempts[0]),
                         "property_receipt": {"score_status": "error"}}
        wrong_request["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "request_sha256 mismatch"):
            _join_property_receipts([attempts[0]], [wrong_request])
        inline = {
            **attempts[0], **_attempt_hashes(attempts[0]),
            "property_receipt": {"score_status": "error"},
        }
        _verify_inline_property_receipts([inline])
        inline["request_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "request_sha256 mismatch"):
            _verify_inline_property_receipts([inline])

    def test_grace_decode_failure_still_produces_hash_bound_receipt(self):
        attempt = {
            "attempt_id": "failed", "seed": 17, "output": "not a crystal",
            "request": {"space_group": 221, "ehull_upper": 0.08},
        }
        parser = SimpleNamespace(parse_fn_wyckoff=Mock(side_effect=ValueError("bad structure")))
        scorer = SimpleNamespace(_score_structure=Mock())

        receipt = _receipt(attempt, parser, scorer, calculator=None, hull=None)

        self.assertEqual(receipt["attempt_id"], "failed")
        self.assertEqual(receipt["seed"], 17)
        self.assertEqual(receipt["output_sha256"], _attempt_hashes(attempt)["output_sha256"])
        self.assertEqual(receipt["request_sha256"], _attempt_hashes(attempt)["request_sha256"])
        self.assertEqual(receipt["property_receipt"]["score_status"], "decode_error")
        scorer._score_structure.assert_not_called()

    def test_grace_receipt_append_fsyncs_and_preserves_attempt_identity(self):
        row = {
            "attempt_id": "a", "seed": 4,
            "output_sha256": "1" * 64, "request_sha256": "2" * 64,
            "property_receipt": {"score_status": "scored"},
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipts.jsonl"
            with patch("experiments.crystal_tasks_v1.evaluation.grace.os.fsync") as fsync:
                with path.open("w", encoding="utf-8") as stream:
                    _append_receipt(stream, row)
                fsync.assert_called_once()
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), row)

    def test_grace_setup_failure_stays_unavailable_and_failed(self):
        run_meta = {
            "status": "running", "property_evaluation_available": False,
            "completed_receipts": 0, "receipt_status_counts": {},
        }

        _finish_run_meta(run_meta, Counter(setup_error=1), RuntimeError("no GPU"))

        self.assertEqual(run_meta["status"], "failed")
        self.assertFalse(run_meta["property_evaluation_available"])
        self.assertEqual(run_meta["receipt_status_counts"], {"setup_error": 1})
        self.assertEqual(run_meta["completed_receipts"], 1)
        self.assertIn("setup_error", run_meta)

    def test_attempt_plan_is_complete_and_binds_seed_and_request(self):
        attempts = [
            {"attempt_id": "failed", "seed": 8, "request": {
                "space_group": 221, "ehull_upper": 0.08,
            }, "output": ""},
            {"attempt_id": "ok", "seed": 9, "request": {
                "space_group": 62, "ehull_upper": 0.1,
            }, "output": "Na"},
        ]
        plan = [
            {"attempt_id": "ok", "seed": 9, "request": {
                "space_group": 62, "ehull_upper": 0.1,
            }, "attempt_index": 1},
            {"attempt_id": "failed", "seed": 8, "request": {
                "space_group": 221, "ehull_upper": 0.08,
            }, "attempt_index": 0},
        ]
        self.assertEqual([row["attempt_id"] for row in _join_attempt_plan(attempts, plan)], ["ok", "failed"])
        with self.assertRaisesRegex(ValueError, "missing"):
            _join_attempt_plan(attempts[:1], plan)
        wrong_seed = [dict(plan[0], seed=10), plan[1]]
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            _join_attempt_plan(attempts, wrong_seed)
        wrong_request = [dict(plan[0], request={"space_group": 62, "ehull_upper": 0.08}), plan[1]]
        with self.assertRaisesRegex(ValueError, "request mismatch"):
            _join_attempt_plan(attempts, wrong_request)

    def test_full_reference_decode_fails_with_identity_instead_of_dropping_row(self):
        parser = Mock(side_effect=[object(), ValueError("bad structure")])
        legacy = SimpleNamespace(parse_fn_wyckoff=parser)
        rows = [
            {"input": "<CRYSTAL_START>Na<CRYSTAL_END>", "metadata": {"material_id": "mp-a", "row_index": 0}},
            {"input": "<CRYSTAL_START>K<CRYSTAL_END>", "metadata": {"material_id": "mp-b", "row_index": 1}},
        ]
        with patch("experiments.crystal_tasks_v1.evaluation.references.load_legacy_audit", return_value=legacy):
            with self.assertRaisesRegex(ValueError, "mp-b.*row_index=1"):
                decode_reference_rows(rows)
        self.assertEqual(parser.call_count, 2)

