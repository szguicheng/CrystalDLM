import numpy as np
import pytest

from geometry import (
    execute_geometry_plan,
    expand_orbit_sites,
    to_input_frame_stable,
    verify_public_table,
)


def test_p1_origin_in_double_a_input_cell_keeps_two_stable_sites():
    table = verify_public_table()
    plan = {
        "space_group_number": 1,
        "orbits": [{"id": "origin", "element": "Si", "wyckoff": "1a", "parameters": [0, 0, 0]}],
        # P=2 doubles the input a axis; the input cell has a=8 A.
        "frame_transform": {"matrix": np.diag([2, 1, 1]), "origin_shift": [0, 0, 0]},
        "composition": {"Si": 2},
        "lattice": [8, 5, 6, 90, 90, 90],
    }

    structure = execute_geometry_plan(plan, table=table)
    np.testing.assert_allclose(structure.lattice.abc, [8, 5, 6], atol=1e-12)
    np.testing.assert_allclose(structure.frac_coords, [[0, 0, 0], [0.5, 0, 0]], atol=1e-12)

    std_sites = expand_orbit_sites(1, plan["orbits"], table=table)
    positions, species, site_ids, site_id_to_index = to_input_frame_stable(
        std_sites,
        plan["frame_transform"]["matrix"],
        plan["frame_transform"]["origin_shift"],
    )
    np.testing.assert_allclose(positions, structure.frac_coords, atol=1e-12)
    assert species == [14, 14]
    assert site_ids == ["origin_1", "origin_2"]
    assert site_id_to_index == {"origin_1": 0, "origin_2": 1}


def test_input_frame_dedup_keeps_distinct_species_at_the_same_position():
    mixed_species = to_input_frame_stable(
        [(np.zeros(3), 14, "silicon"), (np.zeros(3), 8, "oxygen")], np.eye(3), np.zeros(3)
    )
    assert mixed_species[1] == [14, 8]
    assert mixed_species[2] == ["silicon_1", "oxygen_1"]


def test_orbit_and_correction_order_preserve_site_ids_and_full_fractional_deltas():
    table = verify_public_table()
    orbits = [
        {"id": "alpha", "element": "Li", "wyckoff": "1a", "parameters": [0.126, 0.374, 0.626]},
        {"id": "beta", "element": "O", "wyckoff": "1a", "parameters": [0.376, 0.126, 0.876]},
    ]
    corrections = [
        {"site_id": "alpha_1", "element": "Li", "delta": [0.9, 0.8, -0.7]},
        {"site_id": "beta_1", "element": "O", "delta": [-0.5, 0.95, 0.2]},
    ]
    plan = {
        "space_group_number": 1,
        "orbits": orbits,
        "frame_transform": {"matrix": np.eye(3), "origin_shift": [0, 0, 0]},
        "round_digits": 2,
        "precision_corrections": corrections,
        "composition": {"Li": 1, "O": 1},
        "lattice": [4, 5, 6, 90, 90, 90],
    }
    reordered = {
        **plan,
        "orbits": list(reversed(orbits)),
        "precision_corrections": list(reversed(corrections)),
    }

    def ids_and_indices(candidate):
        std_sites = expand_orbit_sites(1, candidate["orbits"], table=table)
        _, _, site_ids, site_id_to_index = to_input_frame_stable(
            std_sites,
            candidate["frame_transform"]["matrix"],
            candidate["frame_transform"]["origin_shift"],
        )
        return site_ids, site_id_to_index

    site_ids, site_id_to_index = ids_and_indices(plan)
    reordered_ids, reordered_indices = ids_and_indices(reordered)
    assert site_ids == reordered_ids == ["alpha_1", "beta_1"]
    assert site_id_to_index == reordered_indices == {"alpha_1": 0, "beta_1": 1}

    structure = execute_geometry_plan(plan, table=table)
    reordered_structure = execute_geometry_plan(reordered, table=table)
    np.testing.assert_allclose(structure.frac_coords, reordered_structure.frac_coords, atol=1e-12)
    np.testing.assert_allclose(
        structure.frac_coords[site_id_to_index["alpha_1"]], [0.03, 0.17, 0.93], atol=1e-12
    )
    np.testing.assert_allclose(
        structure.frac_coords[site_id_to_index["beta_1"]], [0.88, 0.08, 0.08], atol=1e-12
    )
    np.testing.assert_allclose(
        reordered_structure.frac_coords[reordered_indices["alpha_1"]], [0.03, 0.17, 0.93], atol=1e-12
    )
    np.testing.assert_allclose(
        reordered_structure.frac_coords[reordered_indices["beta_1"]], [0.88, 0.08, 0.08], atol=1e-12
    )


@pytest.mark.parametrize(
    ("corrections", "message"),
    [
        ([{"site_id": "missing_1", "element": "Li", "delta": [0, 0, 0]}], "not found"),
        (
            [
                {"site_id": "alpha_1", "element": "Li", "delta": [0, 0, 0]},
                {"site_id": "alpha_1", "element": "Li", "delta": [0, 0, 0]},
            ],
            "Duplicate precision correction target",
        ),
        ([{"site_id": "alpha_1", "element": "O", "delta": [0, 0, 0]}], "element mismatch"),
    ],
)
def test_precision_corrections_reject_invalid_site_references(corrections, message):
    plan = {
        "space_group_number": 1,
        "orbits": [{"id": "alpha", "element": "Li", "wyckoff": "1a", "parameters": [0.1, 0.2, 0.3]}],
        "frame_transform": {"matrix": np.eye(3), "origin_shift": [0, 0, 0]},
        "precision_corrections": corrections,
        "composition": {"Li": 1},
        "lattice": [4, 5, 6, 90, 90, 90],
    }

    with pytest.raises(ValueError, match=message):
        execute_geometry_plan(plan)
