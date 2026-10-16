import io
import json
import tempfile
from pathlib import Path
import unittest

from annotation import description_facts, payload_for, read_stream, render_description, save_receipts


class AnnotationTest(unittest.TestCase):
    def test_payload_separates_property_and_preserves_max(self):
        row = {"status": "ok", "material_id": "mp-test", "core_facts": {
                   "cell_formula": "Si8", "reduced_formula": "Si", "elements": ["Si"], "element_count": 1,
                   "cell_atom_count": 8, "space_group_number": 227, "space_group_symbol": "Fd-3m",
                   "crystal_system": "cubic", "independent_orbit_count": 1},
               "local_structure_facts": {"coordination_environments": [], "connectivity": {}, "polyhedra_sharing": {"status": "not_computed"}},
               "property_labels": {"energy_above_hull": 0.0123456789, "unit": "eV/atom", "provenance": "archive"}}
        config = {"model": "deepseek-v4-pro", "thinking": {"type": "enabled"}, "reasoning_effort": "max", "max_tokens": 16384}
        template = "{structure_version}\n{core_facts}\n{local_structure_facts}\n{property_labels}"
        payload = payload_for(row, config, template)
        self.assertEqual(payload["reasoning_effort"], "max")
        self.assertNotIn("0.0123456789", payload["messages"][0]["content"])
        self.assertIn("0.0123456789 eV/atom", render_description("The crystal is cubic.", row))
        self.assertIn("original archived MP", render_description("The crystal is cubic.", row))

    def test_repeated_orbits_distinguish_counts_from_coordination(self):
        env = {"center_element": "Cl", "coordination_number": 4,
               "neighbor_element_counts": {"In": 2, "Tl": 2}, "recognized_motif": None,
               "orbit_multiplicity": 1, "neighbor_distance_range_A": [2.5, 3.5]}
        row = {"local_structure_facts": {"coordination_environments": [env] * 6,
                                        "connectivity": {}, "polyhedra_sharing": {"status": "not_computed"}}}
        group = description_facts(row)["coordination_environment_groups"][0]
        self.assertEqual(group["geometric_neighbors_per_center_atom"], 4)
        self.assertEqual(group["number_of_independent_orbits_with_this_environment"], 6)
        self.assertEqual(group["number_of_unit_cell_atoms_with_this_environment"], 6)

    def test_render_omits_pipeline_ids_and_unavailable_metadata(self):
        row = {"material_id": "mp-test", "property_labels": {"energy_above_hull": 0.1, "unit": "eV/atom"}}
        text = "The PLaID-decoded structure of Si (mp-test) is cubic. Theoretical density was not supplied. Polyhedra-sharing information was not computed."
        self.assertEqual(render_description(text, row), "The structure of Si is cubic. The original archived MP structure has a supplied energy above hull label of 0.1 eV/atom.")
        for absent in ["No property labels were supplied for this structure.",
                       "No polyhedra-sharing analysis or property labels were supplied for this structure."]:
            self.assertNotIn("were supplied", render_description("The crystal is cubic. " + absent, row))
        self.assertIn("The graph is three-dimensional.", render_description("The graph is three-dimensional; polyhedra-sharing analysis was not computed, and no property labels are supplied here.", row))

    def test_render_combines_structural_sentences_without_losing_facts(self):
        row = {"material_id": "mp-test", "property_labels": {"energy_above_hull": 0.1, "unit": "eV/atom"}}
        raw = "The structure of material mp-test is cubic. Its cell has eight atoms. Si has four neighbors. Tetrahedra share corners. The graph is three-dimensional."
        rendered = render_description(raw, row)
        self.assertIn("The structure is cubic; its cell has eight atoms.", rendered)
        self.assertIn("Tetrahedra share corners.", rendered)
        self.assertIn("0.1 eV/atom", rendered)

    def test_stream_keeps_usage_without_reasoning(self):
        events = [{"model": "deepseek-v4-pro", "choices": [{"delta": {"content": "An English description."}, "finish_reason": None}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 42}}]
        stream = io.BytesIO(("\n".join("data: " + json.dumps(e) for e in events) + "\ndata: [DONE]\n").encode())
        result = read_stream(stream)
        self.assertEqual(result["description"], "An English description.")
        self.assertEqual(result["usage"]["total_tokens"], 42)
        self.assertEqual(result["reasoning_content_characters"], 0)
        self.assertEqual(result["content_chunks"], 1)
        self.assertGreaterEqual(result["time_to_first_content_seconds"], 0)
        self.assertEqual(result["content_stream_seconds"], 0)
        truncated = read_stream(io.BytesIO(b'data: {"choices": [{"delta": {"content": "cut"}, "finish_reason": "length"}], "usage": {"total_tokens": 42}}\n'))
        self.assertEqual(truncated["finish_reason"], "length")
        self.assertEqual(truncated["usage"]["total_tokens"], 42)

    def test_receipts_replacement_preserves_previous_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipts.jsonl"
            first = {"material_id": "a", "status": "ok"}
            second = {"material_id": "b", "status": "error"}
            save_receipts(path, [first])
            save_receipts(path, [first, second])
            self.assertEqual([json.loads(line) for line in path.read_text().splitlines()], [first, second])


if __name__ == "__main__":
    unittest.main()
