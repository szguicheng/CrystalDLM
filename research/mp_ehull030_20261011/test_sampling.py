import hashlib
import random
import unittest

from research.mp_ehull030_20261011.sampling import select_records, split_records


def row(material_id, energy, chemsys, structure_key):
    return {
        "material_id": material_id,
        "energy_above_hull": energy,
        "chemsys": chemsys,
        "structure_key": structure_key,
    }


class SamplingTests(unittest.TestCase):
    def test_energy_bin_edges_and_out_of_range_rows(self):
        energies = [0, 0.0001, 0.05, 0.0501, 0.1, 0.1001, 0.2, 0.2001, 0.3, -0.01, 0.3001]
        rows = [row(f"mp-{i}", energy, "Fe-O", f"s-{i}") for i, energy in enumerate(energies)]

        selected, report = select_records(rows, per_bin=100)

        self.assertEqual(len(selected), 9)
        self.assertEqual(
            [report["bins"][name]["candidate_count"] for name in ("0", "(0,0.05]", "(0.05,0.1]", "(0.1,0.2]", "(0.2,0.3]")],
            [1, 2, 2, 2, 2],
        )
        self.assertEqual(set(report["out_of_range_ids"]), {"mp-9", "mp-10"})

    def test_structure_duplicate_keeps_lowest_hash_representative(self):
        rows = [row("mp-a", 0.02, "Fe-O", "same"), row("mp-b", 0.25, "Fe-O", "same")]
        expected = min(
            rows,
            key=lambda item: (hashlib.sha256(f"17|{item['material_id']}".encode()).hexdigest(), item["material_id"]),
        )

        selected, report = select_records(rows)

        self.assertEqual([item["material_id"] for item in selected], [expected["material_id"]])
        self.assertEqual(report["deduplicated_ids"], [next(item["material_id"] for item in rows if item is not expected)])
        self.assertEqual(len({item["structure_key"] for item in selected}), len(selected))

    def test_sparse_bin_is_not_filled_from_other_bins(self):
        rows = [row("mp-1", 0.01, "Fe-O", "s1"), row("mp-2", 0.02, "Fe-O", "s2"), row("mp-3", 0.2, "Li-O", "s3")]

        selected, report = select_records(rows, per_bin=5)

        self.assertEqual(len(selected), 3)
        self.assertEqual(report["bins"]["(0,0.05]"]["selected_count"], 2)
        self.assertEqual(report["bins"]["(0.05,0.1]"]["selected_count"], 0)

    def test_capped_chemical_system_quota_is_redistributed(self):
        rows = [row("a-0", 0.01, "A", "a-0")]
        rows.extend(row(f"b-{i}", 0.01, "B", f"b-{i}") for i in range(100))

        selected, report = select_records(rows, per_bin=20)

        quotas = report["bins"]["(0,0.05]"]["chemical_systems"]
        self.assertEqual(len(selected), 20)
        self.assertEqual(quotas["A"]["quota"], 1)
        self.assertEqual(quotas["B"]["quota"], 19)

    def test_equal_sqrt_weights_use_sorted_system_order_for_remainder(self):
        rows = [row(f"{system}-{i}", 0.01, system, f"{system}-{i}") for system in "CBA" for i in range(4)]

        _, report = select_records(rows, per_bin=5)

        quotas = report["bins"]["(0,0.05]"]["chemical_systems"]
        self.assertEqual([quotas[system]["quota"] for system in "ABC"], [2, 2, 1])

    def test_selection_and_report_are_input_order_independent(self):
        rows = [row(f"mp-{i}", 0.02, "Fe-O" if i < 4 else "Li-O", f"s-{i}") for i in range(10)]
        rows.extend([row("mp-duplicate", 0.03, "Fe-O", "s-2")])
        shuffled = list(rows)
        random.Random(9).shuffle(shuffled)

        selected_a, report_a = select_records(rows, per_bin=5)
        selected_b, report_b = select_records(shuffled, per_bin=5)

        self.assertEqual(selected_a, selected_b)
        self.assertEqual(report_a, report_b)
        self.assertEqual(len({item["structure_key"] for item in selected_a}), len(selected_a))

    def test_split_is_deterministic_and_keeps_structure_groups_together(self):
        rows = [row("mp-a", 0.01, "Fe-O", "shared"), row("mp-b", 0.02, "Fe-O", "shared")]
        rows.extend(row(f"mp-{i}", 0.03, "Li-O", f"s-{i}") for i in range(10))

        split_a, report_a = split_records(rows)
        split_b, report_b = split_records(list(reversed(rows)))
        split_by_id = {item["material_id"]: item["split"] for item in split_a}

        self.assertEqual(split_by_id, {item["material_id"]: item["split"] for item in split_b})
        self.assertEqual(report_a, report_b)
        self.assertEqual(split_by_id["mp-a"], split_by_id["mp-b"])
        self.assertTrue(set(split_by_id.values()) <= {"train", "val", "test"})
        self.assertEqual(report_a["structure_counts"]["train"] + report_a["structure_counts"]["val"] + report_a["structure_counts"]["test"], 11)


if __name__ == "__main__":
    unittest.main()
