import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Composition

from geometry import compare_structures


def test_compare_structures_rejects_nan_errors():
    class StructureView(SimpleNamespace):
        def __len__(self):
            return len(self.atomic_numbers)

    lattice = SimpleNamespace(
        matrix=np.eye(3),
        get_all_distances=lambda _a, _b: np.zeros((1, 1)),
    )
    expected = StructureView(
        composition=Composition("Si"),
        lattice=lattice,
        atomic_numbers=np.array([14]),
        frac_coords=np.zeros((1, 3)),
    )
    actual = StructureView(**vars(expected))
    actual.lattice = SimpleNamespace(matrix=np.full((3, 3), np.nan))

    with pytest.raises(ValueError, match="Lattice component error"):
        compare_structures(expected, actual)

    actual.lattice = lattice
    actual.frac_coords = np.full((1, 3), np.nan)
    with pytest.raises(ValueError, match="Coordinate error"):
        compare_structures(expected, actual)


def test_inline_binding_requires_each_reference_once():
    from codec import GateError, bind_inline, validate_encoder_document

    inventory = {
        "hall": {"type": "string", "unit": "identifier", "frame": "public", "value": "P & <1>"}
    }
    assert bind_inline("The setting is [[param:hall]].", inventory).count("<param ") == 1
    for prose in (
        "No parameter.",
        "[[param:hall]] and [[param:hall]]",
        "[[param:unknown]]",
        "<param name=\"hall\">injected</param> [[param:hall]]",
    ):
        with pytest.raises(GateError):
            bind_inline(prose, inventory)

    document = bind_inline("The setting is [[param:hall]].", inventory)
    assert "P &amp; &lt;1&gt;" in document
    assert validate_encoder_document(document, inventory)["hall"] == "P & <1>"


def test_free_prose_gate_rejects_unbound_numbers_and_structured_appendices():
    from codec import GateError, bind_inline

    inventory = {"input.atom_count": {"type": "int", "unit": "atom", "frame": "input", "value": 2}}
    narrative = "The original cell contains [[param:input.atom_count]] atoms in the stated composition."
    assert "<param name=\"input.atom_count\"" in bind_inline(narrative, inventory)
    invalid_tails = (
        "Si (0.125, 0.250, 0.375).",
        "a=4.0 angstrom.",
        "a is ٤.٠ angstrom.",
        "data_example\n_atom_site_type_symbol Si",
        '{"element": "Si"}',
        "element = Si",
        "| element | x | y | z |\n| --- | --- | --- | --- |",
    )
    for tail in invalid_tails:
        with pytest.raises(GateError):
            bind_inline(narrative + "\n\n" + tail, inventory)


def test_encoder_gate_checks_parameter_attributes_and_coverage():
    from codec import GateError, bind_inline, validate_encoder_document

    inventory = {
        "cell.a": {"type": "float64", "unit": "angstrom", "frame": "input", "value": 3.25},
        "frame.shift": {"type": "float64[3]", "unit": "fractional", "frame": "standard", "value": [0.0, 0.5, 0.0]},
    }
    prose = "The cell edge is [[param:cell.a]], and the frame shift is [[param:frame.shift]]."
    document = bind_inline(prose, inventory)
    assert validate_encoder_document(document, inventory)["cell.a"] == 3.25
    for invalid in (
        document.replace('unit="angstrom"', 'unit="bohr"'),
        document.replace('frame="standard"', 'frame="input"'),
        document.replace('type="float64[3]"', 'type="float64[2]"'),
        document.replace('<param name="cell.a"', '<param extra="x" name="cell.a"'),
        document.replace("0.5,0.0", "0.5"),
    ):
        with pytest.raises(GateError):
            validate_encoder_document(invalid, inventory)


def test_decoder_json_gate_rejects_non_strict_json():
    from codec import GateError, strict_json_loads

    for invalid in (
        '{"status":"ok","status":"ok"}',
        '{"value":NaN}',
        '```json\n{"status":"ok"}\n```',
        '{"value":1} trailing',
    ):
        with pytest.raises(GateError):
            strict_json_loads(invalid)


def _prepared_case_with_correction():
    from codec import fixture_narrative, make_inventory, bind_inline

    root = Path(__file__).resolve().parent / "prepared_final"
    manifest = json.loads((root / "manifest.json").read_text())
    item = next(case for case in manifest["cases"] if case["status"] == "prepared" and case["correction_atoms"])
    case_dir = root / "cases" / item["id"]
    facts = json.loads((case_dir / "encoder_facts.json").read_text())
    plan = json.loads((root / "preflight_only/oracle_plans" / f"{item['id']}.json").read_text())
    inventory = make_inventory(facts)
    document = bind_inline(fixture_narrative(facts), inventory)
    return plan, document


def test_decoder_gate_accepts_only_exact_plan_values_and_shapes():
    from codec import GateError, decode_plan_response
    from geometry import verify_public_table

    plan, document = _prepared_case_with_correction()
    table = verify_public_table()
    decoded = decode_plan_response(json.dumps(plan, allow_nan=False), document, table)
    assert decoded["input_atom_count"] == plan["input_atom_count"]
    assert decoded["precision_corrections"] == plan["precision_corrections"]
    wrong_value = copy.deepcopy(plan)
    wrong_value["lattice"][0] += 0.01
    with pytest.raises(GateError) as mismatch:
        decode_plan_response(json.dumps(wrong_value, allow_nan=False), document, table)
    assert mismatch.value.stage == "decoder_inline_mismatch"

    invalid_plans = []
    bad = copy.deepcopy(plan)
    bad["space_group_number"] = True
    invalid_plans.append(bad)
    bad = copy.deepcopy(plan)
    bad["lattice"][0] = str(bad["lattice"][0])
    invalid_plans.append(bad)
    bad = copy.deepcopy(plan)
    bad["input_atom_count"] += 1
    invalid_plans.append(bad)
    bad = copy.deepcopy(plan)
    del bad["round_digits"]
    invalid_plans.append(bad)
    bad = copy.deepcopy(plan)
    bad["source_cif"] = "must not be accepted"
    invalid_plans.append(bad)
    bad = copy.deepcopy(plan)
    bad["orbits"].append(copy.deepcopy(bad["orbits"][0]))
    invalid_plans.append(bad)
    bad = copy.deepcopy(plan)
    bad["precision_corrections"][0]["site_id"] = "o999_999"
    invalid_plans.append(bad)

    for invalid in invalid_plans:
        with pytest.raises(GateError):
            decode_plan_response(json.dumps(invalid, allow_nan=False), document, table)
