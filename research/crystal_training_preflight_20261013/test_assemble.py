import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble


class AssembleBindingTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "material_id": "mp-1",
            "split": "train",
            "status": "ok",
            "plaid": {
                "encoded_text": "NaCl\nSpacegroup: Fm-3m",
                "consistency": {
                    "source_structure_matches_decoded": True,
                    "all_scientific_consistency_checks_pass": True,
                },
                "encoder_fallback_used": False,
            },
            "core_facts": {"reduced_formula": "NaCl"},
            "property_labels": {
                "energy_above_hull": 0.0,
                "unit": "eV/atom",
                "source_field": "energy_above_hull",
                "provenance": {"source_document_id": "doc-1"},
            },
            "source": {"input_sha256": "a" * 64, "structure_sha256": "b" * 64},
        }
        raw = "The decoded structure is cubic."
        self.receipt = {
            "material_id": "mp-1",
            "split": "train",
            "status": "ok",
            "facts_sha256": assemble.helpers.sha(self.row),
            "request_sha256": "c" * 64,
            "structure_description": raw,
            "description": assemble.helpers.render_description(raw, self.row),
            "property_labels": copy.deepcopy(self.row["property_labels"]),
        }

    def test_preserves_triple_and_removes_only_the_rendered_label_suffix(self):
        triple = assemble.build_triple(self.row, self.receipt, "train")
        self.assertEqual(triple["plaid_text"], self.row["plaid"]["encoded_text"])
        self.assertEqual(triple["structure_description_raw"], self.receipt["structure_description"])
        self.assertEqual(triple["structure_description"], "The decoded structure is cubic.")
        self.assertEqual(triple["property_labels"], self.row["property_labels"])
        self.assertEqual(triple["core_facts"], self.row["core_facts"])
        self.assertEqual(triple["facts_sha256"], self.receipt["facts_sha256"])

    def test_rejects_receipt_for_another_material(self):
        receipt = dict(self.receipt, material_id="mp-2")
        with self.assertRaisesRegex(ValueError, "material_id mismatch"):
            assemble.build_triple(self.row, receipt, "train")

    def test_rejects_changed_property_label(self):
        receipt = dict(self.receipt, property_labels=dict(self.receipt["property_labels"], energy_above_hull=0.01))
        with self.assertRaisesRegex(ValueError, "property_labels changed"):
            assemble.build_triple(self.row, receipt, "train")

    def test_cleans_only_material_references_and_keeps_the_original_caption(self):
        for raw in ["The structure (material mp-1) is cubic.",
                    "This structure, material mp-1, is cubic.",
                    "The structure of mp-1 is cubic."]:
            receipt = dict(self.receipt, structure_description=raw,
                           description=assemble.helpers.render_description(raw, self.row))
            triple = assemble.build_triple(self.row, receipt, "train")
            self.assertNotIn("mp-1", triple["structure_description"])
            self.assertIn("is cubic.", triple["structure_description"])
            self.assertEqual(triple["structure_description_raw"], raw)


if __name__ == "__main__":
    unittest.main()
