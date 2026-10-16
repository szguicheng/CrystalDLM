import gzip
import json
import subprocess
import sys
from pathlib import Path

import preprocess
from preprocess import classify_motif


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "research/crystal_preannotation_20261012/preprocess.py"
SOURCE = ROOT / "research/mp_ehull030_20261011/collection_001/data_001/records.jsonl.gz"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_three_source_fixtures_keep_structure_and_label_consistency(tmp_path):
    output = tmp_path / "first-three.jsonl"
    result = run_cli("--input", SOURCE, "--output", output, "--limit", 3)
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["material_id"] for row in rows] == ["mp-624928", "mp-639347", "mp-605060"]
    assert all(row["status"] == "ok" for row in rows)
    assert rows[0]["property_labels"]["energy_above_hull"] == 0.017097575234372
    assert [row["source"]["source_atom_count"] for row in rows] == [8, 9, 8]
    assert [row["core_facts"]["cell_atom_count"] for row in rows] == [8, 9, 8]

    required_flags = {
        "encoded_header_sites_matches_decoded_atom_count",
        "encoded_orbit_multiplicity_sum_matches_header",
        "source_reduced_composition_matches_decoded",
        "source_element_set_matches_decoded",
        "decoded_cell_volume_positive",
        "decoded_sites_unique",
        "encoded_space_group_matches_decoded_analysis",
        "encoded_orbit_count_matches_decoded_analysis",
    }
    for row in rows:
        consistency = row["plaid"]["consistency"]
        assert required_flags <= consistency.keys()
        assert consistency["all_scientific_consistency_checks_pass"] is True
        assert row["core_facts"]["independent_orbit_count"] == len(row["core_facts"]["wyckoff_labels"])
        assert row["local_structure_facts"]["polyhedra_sharing"]["status"] == "not_computed"
        assert all(
            "oxidation_state" not in species
            for site in row["plaid"]["decoded_structure"]["sites"]
            for species in site["species"]
        )
    copper = next(
        item for item in rows[0]["local_structure_facts"]["coordination_environments"]
        if item["center_element"] == "Cu"
    )
    assert copper["coordination_number"] == 4
    assert copper["recognized_motif"] == "tetrahedral"
    assert copper["recognized_motif_score"] > copper["recognized_motif_threshold"]


def test_malformed_structure_is_a_separate_error_record_with_label(tmp_path):
    source = tmp_path / "bad.jsonl.gz"
    output = tmp_path / "bad-output.jsonl"
    with gzip.open(source, "wt") as stream:
        stream.write(json.dumps({
            "material_id": "mp-malformed",
            "split": "train",
            "structure": None,
            "energy_above_hull": 0.125,
            "energy_above_hull_unit": "eV/atom",
            "provenance": {"source": "fixture"},
        }) + "\n")

    result = run_cli("--input", source, "--output", output)
    assert result.returncode == 0, result.stderr
    row = json.loads(output.read_text())
    assert row["status"] == "error"
    assert row["material_id"] == "mp-malformed"
    assert row["property_labels"]["energy_above_hull"] == 0.125
    assert row["error"]["stage"] == "source_structure"


def test_shard_and_limit_select_source_rows_deterministically(tmp_path):
    output = tmp_path / "shard-1.jsonl"
    result = run_cli(
        "--input", SOURCE, "--output", output,
        "--shard-index", 1, "--shards", 2, "--limit", 1,
    )
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["material_id"] == "mp-639347"
    assert rows[0]["source"]["row_index"] == 1
    original = output.read_bytes()
    duplicate = run_cli(
        "--input", SOURCE, "--output", output,
        "--shard-index", 1, "--shards", 2, "--limit", 1,
    )
    assert duplicate.returncode != 0
    assert output.read_bytes() == original


def test_undefined_order_parameters_do_not_hide_supported_motif():
    result = classify_motif(5, {
        "cn": 5.0,
        "qtet": None,
        "qoct": None,
        "qbcc": None,
        "q6": None,
        "qsqpyr": 0.91,
        "qtribipyr": None,
    })
    assert result["label"] == "square pyramidal"
    assert result["score"] == 0.91
    assert result["threshold"] == 0.8

    unsupported = classify_motif(4, {"cn": 4.0, "qtet": None})
    assert unsupported["classification"] == "unrecognized"
    assert unsupported["label"] is None


def test_pyxtal_spacegroup_aliases_compare_by_canonical_number():
    compare = getattr(preprocess, "space_group_consistency", None)
    assert callable(compare)
    cases = [
        ("P21/m", "P2_1/m", 11, "P12_1/m1"),
        ("P21", "P2_1", 4, "P12_11"),
        ("P21/c", "P2_1/c", 14, "P12_1/c1"),
    ]
    for encoded, decoded, number, canonical in cases:
        result = compare(encoded, decoded, number)
        assert result["matches"] is True
        assert result["encoded_space_group_number"] == number
        assert result["encoded_space_group_canonical_symbol"] == canonical
        assert result["decoded_space_group_canonical_symbol"] == canonical

    mismatch = compare("P21/c", "P1", 1)
    assert mismatch["matches"] is False
