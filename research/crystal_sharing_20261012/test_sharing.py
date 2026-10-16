import gzip
import json

import numpy as np
import pytest

from augment import augment_file, augment_record
from sharing import (
    _candidate_polyhedron_pairs,
    _validated_motif_metadata,
    build_polyhedron,
    candidate_translations,
    classify_pair,
)


TOL = 1e-5
IDENTITY = np.eye(3)


def polyhedron(center, points, site_indices=None, images=None):
    site_indices = site_indices or list(range(10, 10 + len(points)))
    images = images or [[0, 0, 0]] * len(points)
    rows = [
        {
            "site_index": site_index,
            "image": image,
            "element": "X",
            "coords": coords,
        }
        for site_index, image, coords in zip(site_indices, images, points, strict=True)
    ]
    result = build_polyhedron(center, rows, TOL)
    assert result["status"] == "valid", result
    return result


def test_tetrahedra_meet_at_one_separated_corner():
    shared = (10, (0, 0, 0))
    left = polyhedron(
        [-0.75, 0, 0.25],
        [[0, 0, 0], [-1, 1, 0], [-1, -1, 0], [-1, 0, 1]],
        [10, 11, 12, 13],
    )
    right = polyhedron(
        [0.75, 0, -0.25],
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0, -1]],
        [10, 21, 22, 23],
    )

    relation = classify_pair(left, right, (0, 0, 0), IDENTITY, TOL)

    assert shared in left["points"]
    assert relation["status"] == "shared"
    assert relation["sharing_type"] == "corner"
    assert relation["shared_vertices"] == [
        {"site_index": 10, "image": [0, 0, 0], "element": "X"}
    ]


def test_tetrahedra_meet_along_a_hull_edge():
    left = polyhedron(
        [-0.5, 0, 0],
        [[0, -1, 0], [0, 1, 0], [-1, 0, 1], [-1, 0, -1]],
        [10, 11, 12, 13],
    )
    right = polyhedron(
        [0.5, 0, 0],
        [[0, -1, 0], [0, 1, 0], [1, 0, 1], [1, 0, -1]],
        [10, 11, 22, 23],
    )

    relation = classify_pair(left, right, (0, 0, 0), IDENTITY, TOL)

    assert relation["status"] == "shared"
    assert relation["sharing_type"] == "edge"
    assert len(relation["shared_vertices"]) == 2


def test_tetrahedra_meet_across_the_same_face():
    left = polyhedron(
        [0, -0.25, -0.25],
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, -1]],
        [10, 11, 12, 13],
    )
    right = polyhedron(
        [0, -0.25, 0.25],
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, 1]],
        [10, 11, 12, 23],
    )

    relation = classify_pair(left, right, (0, 0, 0), IDENTITY, TOL)

    assert relation["status"] == "shared"
    assert relation["sharing_type"] == "face"
    assert len(relation["shared_vertices"]) == 3


def test_coplanar_triangles_merge_into_one_shared_square_face():
    base = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    left = polyhedron([0, 0, -0.25], [*base, [0, 0, -1]], [10, 11, 12, 13, 14])
    right = polyhedron([0, 0, 0.25], [*base, [0, 0, 1]], [10, 11, 12, 13, 24])

    relation = classify_pair(left, right, (0, 0, 0), IDENTITY, TOL)

    assert relation["status"] == "shared"
    assert relation["sharing_type"] == "face"
    assert len(relation["shared_vertices"]) == 4


def test_two_common_vertices_that_are_not_hull_edge_are_ambiguous():
    cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    common_points = {tuple(point): index for index, point in enumerate(cube)}
    common_sites = {common_points[(-1, -1, -1)]: 10, common_points[(1, 1, 1)]: 11}
    left_ids = [common_sites.get(i, 100 + i) for i in range(8)]
    right_ids = [common_sites.get(i, 200 + i) for i in range(8)]
    left = polyhedron([0, 0, 0], cube, left_ids)
    right = polyhedron([0, 0, 0], cube, right_ids)

    relation = classify_pair(left, right, (0, 0, 0), IDENTITY, TOL)

    assert relation["status"] == "ambiguous"
    assert relation["reason"] == "common_vertices_do_not_form_hull_edge"


def test_three_common_vertices_without_a_common_facet_are_not_face_sharing():
    cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    shared_coordinates = {(-1, -1, -1), (1, 1, 1), (1, -1, 1)}
    shared_sites = {point: 10 + index for index, point in enumerate(shared_coordinates)}
    left_ids = [shared_sites.get(tuple(point), 100 + i) for i, point in enumerate(cube)]
    right_ids = [shared_sites.get(tuple(point), 200 + i) for i, point in enumerate(cube)]
    left = polyhedron([0, 0, 0], cube, left_ids)
    right = polyhedron([0, 0, 0], cube, right_ids)

    relation = classify_pair(left, right, (0, 0, 0), IDENTITY, TOL)

    assert relation["status"] == "ambiguous"
    assert relation["reason"] == "common_vertices_do_not_form_same_hull_facet"


def test_periodic_vertex_identity_finds_the_center_translation():
    left = polyhedron(
        [0.75, 0, -0.25],
        [[0, 0, 0], [1, 1, 0], [1, -1, 0], [1, 0, -1]],
        [10, 11, 12, 13],
    )
    right = polyhedron(
        [3.25, 0, 0.25],
        [[4, 0, 0], [3, 1, 0], [3, -1, 0], [3, 0, 1]],
        [10, 21, 22, 23],
        [[1, 0, 0]] * 4,
    )
    lattice = np.diag([4.0, 4.0, 4.0])

    shifts = candidate_translations(left, right)

    assert shifts == [(-1, 0, 0)]
    relation = classify_pair(left, right, shifts[0], lattice, TOL)
    assert relation["status"] == "shared"
    assert relation["sharing_type"] == "corner"
    assert relation["center_b_image"] == [-1, 0, 0]
    assert relation["shared_vertices"][0]["image"] == [0, 0, 0]


def test_self_orbit_pair_keeps_one_periodic_translation_orientation():
    cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    site_indices = [5 if index in {0, 7} else 100 + index for index in range(8)]
    images = [[1, 0, 0] if index == 7 else [0, 0, 0] for index in range(8)]
    poly = polyhedron([0, 0, 0], cube, site_indices, images)

    assert _candidate_polyhedron_pairs([poly]) == [(0, 0, (1, 0, 0))]


def test_motif_label_is_dropped_when_recomputed_neighbor_data_disagrees():
    result = _validated_motif_metadata(
        {
            "center_element": "O",
            "coordination_number": 4,
            "neighbor_element_counts": {"Si": 4},
            "motif_classification": "tetrahedral",
            "recognized_motif": "tetrahedral",
        },
        "O",
        5,
        {"Si": 5},
    )

    assert result["motif_metadata_validation"] == "mismatch_with_stored_orbit"
    assert result["motif_classification"] == "unsupported_by_recomputed_neighbors"
    assert result["recognized_motif"] is None


def test_sharing_failure_preserves_preexisting_core_facts_and_labels():
    row = {
        "status": "ok",
        "property_labels": {"energy_above_hull": 0.125},
        "core_facts": {"cell_atom_count": 4},
        "local_structure_facts": {"coordination_environments": []},
    }

    augmented, status = augment_record(row, {})

    assert status == "error"
    assert augmented["status"] == "ok"
    assert augmented["property_labels"] == {"energy_above_hull": 0.125}
    assert augmented["core_facts"] == {"cell_atom_count": 4}
    assert augmented["local_structure_facts"]["polyhedra_sharing"]["status"] == "error"


def test_non_ok_fact_rows_are_left_unchanged():
    row = {"status": "error", "error": {"stage": "core_facts"}}

    augmented, status = augment_record(row, {})

    assert status == "preserved_non_ok"
    assert augmented == row


def test_gzip_stream_limit_and_existing_output_protection(tmp_path):
    source = tmp_path / "input.jsonl.gz"
    output = tmp_path / "output.jsonl.gz"
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for index in range(2):
            stream.write(json.dumps({"status": "error", "row": index}) + "\n")

    summary = augment_file(source, output, limit=1)
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    original = output.read_bytes()

    assert summary["input_rows"] == 1
    assert summary["preserved_non_ok"] == 1
    assert rows == [{"status": "error", "row": 0}]
    with pytest.raises(FileExistsError):
        augment_file(source, output, limit=1)
    assert output.read_bytes() == original
