"""Strict CNL decoding, physical audit and task grading for the relational pilot."""
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import re

import numpy as np
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.optimize import linear_sum_assignment

import prepare


codec = prepare.relational_codec
geometry = codec.geometry
TABLE = prepare.TABLE


def _plan_from(value):
    if isinstance(value, Structure):
        return None
    if not isinstance(value, dict):
        return None
    if "decoded_plan" in value:
        return value["decoded_plan"]
    if "plan" in value and isinstance(value["plan"], dict):
        return value["plan"]
    if "plans" in value:
        return value["plans"].get("source")
    if "space_group_number" in value and "lattice" in value:
        return value
    return None


def _composition(structure):
    return dict(sorted((symbol, float(amount)) for symbol, amount in structure.composition.get_el_amt_dict().items()))


def _minimum_pair_distance(structure):
    radius = float(np.linalg.norm(structure.lattice.matrix, axis=1).sum() / 2 + 1e-8)
    neighbors = structure.get_all_neighbors(radius, include_index=True, include_image=True)
    distances = [
        float(neighbor.nn_distance)
        for center, group in enumerate(neighbors)
        for neighbor in group
        if not (neighbor.index == center and np.allclose(neighbor.image, (0, 0, 0), rtol=0, atol=0))
    ]
    return min(distances) if distances else None


def _expected_plan(requested):
    if requested is None:
        return None
    if isinstance(requested, dict) and "plans" in requested:
        return requested["plans"]["source"]
    if isinstance(requested, dict) and "decoded_plan" in requested:
        return requested["decoded_plan"]
    return requested


def audit(raw_text, requested=None):
    """Decode raw model text once, execute geometry, and keep every failure."""
    result = {
        "raw_text": raw_text,
        "valid": False,
        "failure_stage": None,
        "reasons": [],
        "requested_checks": {},
        "physical": {},
    }
    try:
        plan = codec.decode(raw_text, table=TABLE)
    except Exception as error:
        result["failure_stage"] = "decode"
        result["reasons"].append(f"{type(error).__name__}: {error}")
        return result

    result["decoded_plan"] = plan
    result["decoded_plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    try:
        structure = geometry.execute_geometry_plan(plan, table=TABLE)
    except Exception as error:
        result["failure_stage"] = "geometry"
        result["reasons"].append(f"{type(error).__name__}: {error}")
        return result

    try:
        matrix = np.asarray(structure.lattice.matrix, dtype=float)
        volume = float(structure.volume)
        minimum_distance = _minimum_pair_distance(structure)
        detected_sg = int(SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5).get_space_group_number())
        structure_summary = {
            "atoms": int(len(structure)),
            "composition": _composition(structure),
            "volume_A3": volume,
            "input_lattice_parameters": [float(value) for value in structure.lattice.parameters],
            "independent_orbits": int(len(plan["orbits"])),
            "space_group_from_plan": int(plan["space_group_number"]),
            "space_group_redetected": detected_sg,
            "minimum_pair_distance_A": minimum_distance,
        }
        checks = {
            "finite_lattice": bool(np.isfinite(matrix).all()),
            "finite_positive_volume": bool(np.isfinite(volume) and volume > 0),
            "minimum_pair_distance_defined": minimum_distance is not None and bool(np.isfinite(minimum_distance)),
            "minimum_pair_distance_at_least_0_5_A": bool(
                minimum_distance is not None and minimum_distance >= 0.5 - 1e-10
            ),
            "space_group_redetected": bool(detected_sg == int(plan["space_group_number"])),
        }
        expected = _expected_plan(requested)
        if expected is not None:
            expected_composition = dict(sorted((key, float(value)) for key, value in expected["composition"].items()))
            checks.update({
                "requested_input_composition": _composition(structure) == expected_composition,
                "requested_input_atoms": len(structure) == int(expected["input_atom_count"]),
                "requested_space_group": int(plan["space_group_number"]) == int(expected["space_group_number"]),
            })
        result["structure"] = structure_summary
        result["physical"] = checks
        result["requested_checks"] = {
            key: value for key, value in checks.items() if key.startswith("requested_")
        }
        if not checks["finite_lattice"]:
            result["reasons"].append("nonfinite_lattice")
        if not checks["finite_positive_volume"]:
            result["reasons"].append("invalid_volume")
        if not checks["minimum_pair_distance_defined"]:
            result["reasons"].append("minimum_pair_distance_undefined")
        elif not checks["minimum_pair_distance_at_least_0_5_A"]:
            result["reasons"].append("minimum_pair_distance_below_0.5_A")
        if not checks["space_group_redetected"]:
            result["reasons"].append("redetected_space_group_mismatch")
        for key in ("requested_input_composition", "requested_input_atoms", "requested_space_group"):
            if checks.get(key) is False:
                result["reasons"].append(key + "_mismatch")
        result["valid"] = not result["reasons"]
    except Exception as error:
        result["failure_stage"] = "physical_audit"
        result["reasons"].append(f"{type(error).__name__}: {error}")
    return result


def _expected_fields(value):
    if isinstance(value, Structure):
        detected_sg = int(SpacegroupAnalyzer(value, symprec=0.01, angle_tolerance=5).get_space_group_number())
        sym = SpacegroupAnalyzer(value, symprec=0.01, angle_tolerance=5).get_symmetrized_structure()
        return {
            "sg": detected_sg,
            "counts": _composition(value),
            "atoms": int(len(value)),
            "orbits": len(sym.equivalent_indices),
            "lengths": [float(x) for x in value.lattice.abc],
        }
    if isinstance(value, dict) and "structure" in value and "decoded_plan" in value:
        structure = value["structure"]
        plan = value["decoded_plan"]
        return {
            "sg": int(structure["space_group_redetected"]),
            "counts": structure["composition"],
            "atoms": int(structure["atoms"]),
            "orbits": int(structure["independent_orbits"]),
            "lengths": [float(x) for x in plan["lattice"][:3]],
        }
    plan = _plan_from(value)
    if plan is None:
        raise ValueError("Expected a decoded relational plan, audit result or executed structure")
    return {
        "sg": int(plan["space_group_number"]),
        "counts": dict(sorted((key, float(amount)) for key, amount in plan["composition"].items())),
        "atoms": int(plan["input_atom_count"]),
        "orbits": len(plan["orbits"]),
        "lengths": [float(x) for x in plan["lattice"][:3]],
    }


def answer(text_or_plan):
    """Construct the supervised response from an executed plan, never a PLaID parse."""
    if isinstance(text_or_plan, str):
        plan = codec.decode(text_or_plan, table=TABLE)
    else:
        plan = _plan_from(text_or_plan)
    if plan is None:
        raise ValueError("Cannot form an answer without a decoded plan")
    structure = geometry.execute_geometry_plan(plan, table=TABLE)
    fields = {
        "sg": int(SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5).get_space_group_number()),
        "counts": _composition(structure),
        "atoms": int(len(structure)),
        "orbits": len(plan["orbits"]),
        "lengths": [float(value) for value in plan["lattice"][:3]],
    }
    formula = "".join(
        element + ("" if count == 1 else str(int(count)) if float(count).is_integer() else format(count, ".15g"))
        for element, count in fields["counts"].items()
    )
    a, b, c = fields["lengths"]
    return (
        f"The space group is {fields['sg']}. The composition is {formula}. "
        f"The input cell contains {fields['atoms']} atoms in {fields['orbits']} independent orbits. "
        f"The lattice lengths are a={a:.2f}, b={b:.2f}, c={c:.2f} angstroms."
    )


def grade_answer(response, expected_plan_or_structure):
    try:
        expected = _expected_fields(expected_plan_or_structure)
    except Exception as error:
        return {"success": False, "fields": {key: False for key in ("sg", "formula", "atoms", "orbits", "a", "b", "c")},
                "error": f"{type(error).__name__}: {error}", "raw_text": response}
    patterns = {
        "sg": r"(?:space\s*group)\s*(?:number\s*)?(?:is\s*|[:=]\s*)?(\d+)",
        "formula": r"(?:composition|formula)\s*(?:is\s*|[:=]\s*)?((?:[A-Z][a-z]?\d*)+)",
        "atoms": r"(\d+)\s+atoms\b",
        "orbits": r"(\d+)\s+(?:independent\s+)?orbits\b",
    }
    fields = {}
    for name, pattern in patterns.items():
        values = re.findall(pattern, response, re.I)
        if name == "formula":
            try:
                fields[name] = bool(values) and all(
                    Composition(value).get_el_amt_dict() == expected["counts"] for value in values
                )
            except ValueError:
                fields[name] = False
        else:
            fields[name] = bool(values) and all(int(value) == expected[name] for value in values)
    triple = re.findall(
        r"lattice\s+lengths\s*(?:are\s*|[:=]\s*)([\d.]+)\s*,\s*([\d.]+)\s*,?\s*(?:and\s*)?([\d.]+)",
        response,
        re.I,
    )
    for index, name in enumerate("abc"):
        values = re.findall(rf"\b{name}\s*[:=]\s*(-?\d+(?:\.\d+)?)", response, re.I)
        values += [row[index] for row in triple]
        try:
            fields[name] = bool(values) and all(abs(float(value) - expected["lengths"][index]) <= 0.005 + 1e-10 for value in values)
        except ValueError:
            fields[name] = False
    return {"success": all(fields.values()), "fields": fields, "raw_text": response}


def _geometry_errors(expected, observed):
    lattice_error = float(np.max(np.abs(expected.lattice.matrix - observed.lattice.matrix)))
    if len(expected) != len(observed) or expected.composition != observed.composition:
        return lattice_error, None, "atom_count_or_composition_mismatch"
    same = np.asarray(expected.atomic_numbers)[:, None] == np.asarray(observed.atomic_numbers)[None, :]
    distances = expected.lattice.get_all_distances(expected.frac_coords, observed.frac_coords)
    left, right = linear_sum_assignment(np.where(same, distances, np.inf))
    if len(left) != len(expected) or not np.isfinite(distances[left, right]).all():
        return lattice_error, None, "species_assignment_unavailable"
    residual = expected.frac_coords[left] - observed.frac_coords[right]
    residual -= np.rint(residual)
    coordinate_error = float(np.linalg.norm(residual @ expected.lattice.matrix, axis=1).max())
    return lattice_error, coordinate_error, None


def grade_edit(output, actual_base_text, factor):
    result = {
        "success": False,
        "raw_text": output,
        "fields": {"lengths_exact": False, "other_plan_fields_exact": False, "whole_geometry_scale": False},
        "diagnostics": {},
    }
    base = audit(actual_base_text)
    if not base["valid"]:
        result["error"] = "actual_base_invalid"
        result["base_audit"] = {"valid": False, "failure_stage": base["failure_stage"], "reasons": base["reasons"]}
        return result
    edited = audit(output)
    result["output_audit"] = {key: edited[key] for key in ("valid", "failure_stage", "reasons", "physical", "structure") if key in edited}
    if edited.get("failure_stage") == "geometry":
        result["error"] = "edited_geometry_failed"
        return result
    if not edited.get("decoded_plan"):
        result["error"] = "edited_plan_decode_failed"
        return result

    base_plan, observed = base["decoded_plan"], edited["decoded_plan"]
    expected = deepcopy(base_plan)
    decimal_factor = Decimal(str(factor))
    target_lengths = [float(Decimal(repr(length)) * decimal_factor) for length in base_plan["lattice"][:3]]
    expected["lattice"][:3] = target_lengths
    observed_lengths = [float(value) for value in observed["lattice"][:3]]
    errors = [abs(actual - target) for actual, target in zip(observed_lengths, target_lengths)]
    result["diagnostics"]["length_max_abs_error_A"] = max(errors)
    relative_errors = [error / abs(target) if target else None for error, target in zip(errors, target_lengths)]
    result["diagnostics"]["length_max_relative_error"] = (
        max(relative_errors) if all(error is not None for error in relative_errors) else None
    )
    result["fields"]["lengths_exact"] = observed_lengths == target_lengths
    result["fields"]["other_plan_fields_exact"] = all(
        observed[key] == base_plan[key] for key in base_plan if key != "lattice"
    ) and observed["lattice"][3:] == base_plan["lattice"][3:]
    result["fields"]["exact_target_plan"] = observed == expected

    base_structure = geometry.execute_geometry_plan(base_plan, table=TABLE)
    edited_structure = geometry.execute_geometry_plan(observed, table=TABLE)
    scaled_reference = Structure(
        Lattice(base_structure.lattice.matrix * float(decimal_factor)),
        base_structure.species,
        base_structure.frac_coords,
        coords_are_cartesian=False,
    )
    lattice_error, coordinate_error, geometry_error = _geometry_errors(scaled_reference, edited_structure)
    geometry_defined = geometry_error is None
    result["diagnostics"]["whole_geometry_scale_errors_A"] = {
        "lattice_matrix_max_abs": lattice_error,
        "coordinate_max": coordinate_error,
        "defined": geometry_defined,
        "undefined_reason": geometry_error,
    }
    result["fields"]["whole_geometry_scale_defined"] = geometry_defined
    result["fields"]["whole_geometry_scale"] = (
        geometry_defined and lattice_error < 1e-8 and coordinate_error < 1e-6
    )
    result["fields"]["edited_physical_audit"] = bool(edited["valid"])
    result["success"] = all(result["fields"].values())
    return result
