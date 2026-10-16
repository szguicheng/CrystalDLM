import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
INLINE = ROOT / "experiments/naturecry_inline_20261003"
sys.path.insert(0, str(INLINE))

import geometry
import spglib

_old_spec = importlib.util.spec_from_file_location("oldcodec", INLINE / "codec.py")
oldcodec = importlib.util.module_from_spec(_old_spec)
_old_spec.loader.exec_module(oldcodec)

from relational_codec import GateError, decode, encode, normalize


@pytest.fixture(scope="module")
def table():
    return geometry.verify_public_table()


def _facts(table, sg, label, parameters, element="Ti", matrix=None, shift=None,
           round_digits=None, corrections=None):
    definition = table[sg]["definitions"][label]
    multiplicity = definition["multiplicity"]
    matrix = matrix if matrix is not None else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    shift = shift if shift is not None else [0.0, 0.0, 0.0]
    hall_symbol = spglib.get_spacegroup_type(table[sg]["hall"]).hall_symbol
    return {
        "input_lattice_parameters": [4.1, 5.2, 6.3, 90.0, 90.0, 90.0],
        "input_composition": {element: multiplicity},
        "input_atoms": multiplicity,
        "space_group_number": sg,
        "hall_symbol": hall_symbol,
        "frame_transform": {"matrix": matrix, "origin_shift": shift},
        "round_digits": round_digits,
        "orbits": [{"id": "o000", "element": element, "wyckoff": label, "parameters": parameters}],
        "precision_corrections": corrections or [],
    }


def _validated_plan(facts, table):
    return oldcodec.validate_plan(oldcodec.plan_from_facts(facts), table)


def test_rutile_relation_recovers_free_parameter_without_a_wyckoff_label(table):
    facts = _facts(table, 136, "4f", [0.137])
    text = encode(facts, table=table)

    assert "Titanium forms a nominal 4-site symmetry orbit named o000 in the standard cell." in text
    assert "The input cell edges a, b, and c measure 4.1, 5.2, and 6.3 angstrom; the angles alpha, beta, and gamma measure 90.0, 90.0, and 90.0 degrees." in text
    assert "Along a, its representative has the free fractional offset 0.137." in text
    assert "Along b, its representative shares the offset along a." in text
    assert "Its representative lies in the plane at fractional c offset 0.0." in text
    assert "Wyckoff" not in text and "4f" not in text and "parameters" not in text
    assert decode(text, table=table) == _validated_plan(facts, table)


def test_share_synonym_is_finite_and_changed_or_duplicate_relation_is_not_ignored(table):
    text = encode(_facts(table, 136, "4f", [0.137]), table=table)
    synonym = text.replace(
        "shares the offset along a", "matches the offset along a"
    )
    assert synonym != text
    assert decode(synonym, table=table)["orbits"][0]["parameters"] == [0.137]

    changed = text.replace(
        "Along b, its representative shares the offset along a.",
        "Its representative lies in the plane at fractional b offset 0.0.",
    )
    with pytest.raises(GateError):
        decode(changed, table=table)

    duplicate = text.replace(
        "Its representative lies in the plane at fractional c offset 0.0.",
        "Along b, its representative shares the offset along a.\n"
        "Its representative lies in the plane at fractional c offset 0.0.",
    )
    assert duplicate != text
    with pytest.raises(GateError):
        decode(duplicate, table=table)

    missing_correction_state = "\n".join(text.splitlines()[:-1])
    with pytest.raises(GateError):
        decode(missing_correction_state, table=table)


def test_frame_rounding_and_residual_facts_round_trip_and_normalize(table):
    facts = _facts(
        table,
        1,
        "1a",
        [0.137, 0.231, 0.367],
        element="Li",
        matrix=[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        shift=[0.125, 0.25, 0.375],
        round_digits=6,
        corrections=[{"site_id": "o000_1", "element": "Li", "delta": [0.003141592653589793, -0.000001, 0.125]}],
    )
    facts["input_composition"] = {"Li": 2}
    facts["input_atoms"] = 2
    text = encode(facts, table=table)

    assert "The standard fractional coordinate a equals 2.0 times input a plus 0.0 times input b plus 0.0 times input c, then shift 0.125." in text
    assert "The input-frame fractional coordinates are rounded to 6 decimal places before any listed corrections." in text
    assert "The input-frame fractional correction for retained site o000_1 of Lithium is (0.003141592653589793, -1e-06, 0.125)." in text
    decoded = decode(text, table=table)
    assert decoded == _validated_plan(facts, table)
    canonical_facts = oldcodec.source_facts_from_plan(decoded)
    assert "space_group_symbol" not in canonical_facts
    assert normalize(text, table=table) == encode(canonical_facts, table=table) == text


def test_signed_zero_in_identity_frame_is_not_compressed_away(table):
    facts = _facts(
        table,
        1,
        "1a",
        [0.137, 0.231, 0.367],
        matrix=[[1.0, -0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        shift=[-0.0, 0.0, 0.0],
    )
    text = encode(facts, table=table)

    assert "The standard fractional coordinate a equals 1.0 times input a plus -0.0 times input b plus 0.0 times input c, then shift -0.0." in text
    decoded = decode(text, table=table)
    assert repr(decoded["frame_transform"]["matrix"][0][1]) == "-0.0"
    assert repr(decoded["frame_transform"]["origin_shift"][0]) == "-0.0"
    assert normalize(text, table=table) == text


def test_every_public_definition_has_an_exact_relational_encoding(table):
    parameter_values = [0.137, 0.231, 0.367]
    definition_count = 0
    for sg, group in table.items():
        hall_symbol = spglib.get_spacegroup_type(group["hall"]).hall_symbol
        for label, definition in group["definitions"].items():
            multiplicity = definition["multiplicity"]
            facts = {
                "input_lattice_parameters": [3.1, 4.2, 5.3, 80.0, 90.0, 100.0],
                "input_composition": {"Si": multiplicity},
                "input_atoms": multiplicity,
                "space_group_number": sg,
                "hall_symbol": hall_symbol,
                "frame_transform": {
                    "matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "origin_shift": [0.0, 0.0, 0.0],
                },
                "round_digits": None,
                "orbits": [{
                    "id": "o000",
                    "element": "Si",
                    "wyckoff": label,
                    "parameters": [parameter_values[axis] for axis in definition["axes"]],
                }],
                "precision_corrections": [],
            }
            text = encode(facts, table=table)
            expected = _validated_plan(facts, table)
            assert decode(text, table=table) == expected, (sg, label)
            assert normalize(text, table=table) == text, (sg, label)
            definition_count += 1

    assert definition_count == 1731
