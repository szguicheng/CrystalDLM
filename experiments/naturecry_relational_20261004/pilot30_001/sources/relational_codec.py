"""Controlled-English relational codec for the frozen NatureCry fact domain."""
import importlib.util
from pathlib import Path
import re
import sys


_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "naturecry_inline_20261003"
if str(_REFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(_REFERENCE_DIR))

import geometry

_OLD_SPEC = importlib.util.spec_from_file_location("oldcodec", _REFERENCE_DIR / "codec.py")
oldcodec = importlib.util.module_from_spec(_OLD_SPEC)
_OLD_SPEC.loader.exec_module(oldcodec)
GateError = oldcodec.GateError

_AXES = "abc"
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ELEMENT_NAMES = {
    geometry.Element.from_Z(z).long_name: geometry.Element.from_Z(z).symbol
    for z in range(1, 119)
}
_ELEMENT_SYMBOLS = {symbol: symbol for symbol in _ELEMENT_NAMES.values()}

_CELL = re.compile(
    rf"^The input cell edges a, b, and c measure ({_NUMBER}), ({_NUMBER}), and ({_NUMBER}) angstrom; "
    rf"the angles alpha, beta, and gamma measure ({_NUMBER}), ({_NUMBER}), and ({_NUMBER}) degrees\.$"
)
_SYMMETRY = re.compile(r"^The public setting is space group ([1-9]\d*), with literal Hall symbol (.+)\.$")
_ATOM_COUNT = re.compile(r"^The input cell contains ([1-9]\d*) atoms in total\.$")
_COMPOSITION = re.compile(r"^(.+) contributes ([1-9]\d*) atoms to the input composition\.$")
_FRAME_COORDINATE = re.compile(
    rf"^The standard fractional coordinate ([abc]) equals ({_NUMBER}) times input a plus "
    rf"({_NUMBER}) times input b plus ({_NUMBER}) times input c, then shift ({_NUMBER})\.$"
)
_ROUND = re.compile(r"^The input-frame fractional coordinates are rounded to (\d+) decimal places before any listed corrections\.$")
_ORBIT = re.compile(r"^(.+) forms a nominal ([1-9]\d*)-site symmetry orbit named (o\d{3,}) in the standard cell\.$")
_FREE = re.compile(rf"^Along ([abc]), its representative has the free fractional offset ({_NUMBER})\.$")
_FIXED = re.compile(rf"^Its representative lies in the plane at fractional ([abc]) offset ({_NUMBER})\.$")
_SHARED = re.compile(r"^Along ([abc]), its representative (?:shares|matches) the offset along ([abc])\.$")
_DOUBLED = re.compile(r"^Along ([abc]), its representative doubles the offset along ([abc])\.$")
_NEGATED = re.compile(r"^Along ([abc]), its representative is the negative of the offset along ([abc])\.$")
_SHIFTED = re.compile(rf"^Along ([abc]), its representative is shifted by ({_NUMBER}) from the offset along ([abc])\.$")
_AFFINE = re.compile(
    rf"^Along ([abc]), its representative has fractional offset equal to ({_NUMBER}) "
    rf"times the offset along ([abc]) plus ({_NUMBER})\.$"
)
_CORRECTION = re.compile(
    rf"^The input-frame fractional correction for retained site (o\d{{3,}}_\d+) "
    rf"of (.+) is \(({_NUMBER}), ({_NUMBER}), ({_NUMBER})\)\.$"
)
_NO_CORRECTIONS = "No input-frame fractional site corrections are applied."
_IDENTITY_FRAME = "The standard and input fractional frames coincide."


def _table(table):
    return geometry.verify_public_table() if table is None else table


def _float(value):
    return repr(float(value))


def _element_name(symbol):
    return geometry.Element(symbol).long_name


def _element_symbol(name):
    if name in _ELEMENT_NAMES:
        return _ELEMENT_NAMES[name]
    if name in _ELEMENT_SYMBOLS:
        return _ELEMENT_SYMBOLS[name]
    raise GateError(f"Unknown element name in relational document: {name}", "decoder_semantic")


def _facts_plan(facts, table):
    return oldcodec.validate_plan(oldcodec.plan_from_facts(facts), table)


def _representative_lines(definition, parameters, orbit_id):
    matrix = definition["a"]
    offset = definition["b"]
    axes = definition["axes"]
    values = dict(zip(axes, parameters))
    lines = []
    for row, axis in enumerate(_AXES):
        if row in values:
            if any(float(matrix[row][column]) != (1.0 if column == row else 0.0) for column in range(3)) or float(offset[row]) != 0.0:
                raise GateError("Public free axis is not a direct identity coordinate", "encoder_semantic")
            lines.append(f"Along {axis}, its representative has the free fractional offset {_float(values[row])}.")
            continue
        terms = [(column, float(matrix[row][column])) for column in range(3) if float(matrix[row][column]) != 0.0]
        shift = float(offset[row])
        if not terms:
            lines.append(f"Its representative lies in the plane at fractional {axis} offset {_float(shift)}.")
            continue
        if len(terms) != 1 or terms[0][0] not in axes:
            raise GateError(f"Unsupported public affine relation for orbit {orbit_id}, axis {axis}", "encoder_semantic")
        source_axis, coefficient = terms[0]
        if coefficient == 1.0 and shift == 0.0:
            lines.append(f"Along {axis}, its representative shares the offset along {_AXES[source_axis]}.")
        elif coefficient == 1.0:
            lines.append(f"Along {axis}, its representative is shifted by {_float(shift)} from the offset along {_AXES[source_axis]}.")
        elif coefficient == 2.0 and shift == 0.0:
            lines.append(f"Along {axis}, its representative doubles the offset along {_AXES[source_axis]}.")
        elif coefficient == -1.0 and shift == 0.0:
            lines.append(f"Along {axis}, its representative is the negative of the offset along {_AXES[source_axis]}.")
        else:
            lines.append(
                f"Along {axis}, its representative has fractional offset equal to {_float(coefficient)} "
                f"times the offset along {_AXES[source_axis]} plus {_float(shift)}."
            )
    return lines


def encode(facts, table=None):
    """Write facts as controlled English, with no hidden plan or label appendix."""
    table = _table(table)
    plan = _facts_plan(facts, table)
    lattice = plan["lattice"]
    lines = [
        "The input cell edges a, b, and c measure "
        + ", ".join(_float(value) for value in lattice[:2])
        + f", and {_float(lattice[2])} angstrom; the angles alpha, beta, and gamma measure "
        + ", ".join(_float(value) for value in lattice[3:5])
        + f", and {_float(lattice[5])} degrees.",
        f"The public setting is space group {plan['space_group_number']}, with literal Hall symbol {plan['hall_symbol']}.",
        f"The input cell contains {plan['input_atom_count']} atoms in total.",
    ]
    lines.extend(
        f"{_element_name(element)} contributes {count} atoms to the input composition."
        for element, count in sorted(plan["composition"].items())
    )

    frame = plan["frame_transform"]
    matrix, shift = frame["matrix"], frame["origin_shift"]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    same_matrix = all(_float(matrix[row][column]) == _float(identity[row][column]) for row in range(3) for column in range(3))
    zero_shift = all(_float(value) == _float(0.0) for value in shift)
    if same_matrix and zero_shift:
        lines.append(_IDENTITY_FRAME)
    else:
        for row, axis in enumerate(_AXES):
            terms = " plus ".join(f"{_float(matrix[row][column])} times input {_AXES[column]}" for column in range(3))
            lines.append(
                f"The standard fractional coordinate {axis} equals {terms}, then shift {_float(shift[row])}."
            )
    digits = plan["round_digits"]
    if digits is None:
        lines.append("The input-frame fractional coordinates are not rounded before any listed corrections.")
    else:
        lines.append(f"The input-frame fractional coordinates are rounded to {digits} decimal places before any listed corrections.")

    sg = plan["space_group_number"]
    for orbit in plan["orbits"]:
        definition = table[sg]["definitions"][orbit["wyckoff"]]
        multiplicity = definition["multiplicity"]
        lines.append(
            f"{_element_name(orbit['element'])} forms a nominal {multiplicity}-site symmetry orbit named {orbit['id']} in the standard cell."
        )
        lines.extend(_representative_lines(definition, orbit["parameters"], orbit["id"]))

    corrections = plan["precision_corrections"]
    if not corrections:
        lines.append(_NO_CORRECTIONS)
    else:
        for correction in corrections:
            delta = ", ".join(_float(value) for value in correction["delta"])
            lines.append(
                f"The input-frame fractional correction for retained site {correction['site_id']} "
                f"of {_element_name(correction['element'])} is ({delta})."
            )
    return "\n".join(lines)


def _parse_coordinate(line):
    for pattern, kind in ((_FREE, "free"), (_FIXED, "fixed")):
        match = pattern.fullmatch(line)
        if match:
            return match.group(1), kind, tuple(float(value) for value in match.groups()[1:])
    match = _SHARED.fullmatch(line)
    if match:
        return match.group(1), "relation", (1.0, match.group(2), 0.0)
    match = _DOUBLED.fullmatch(line)
    if match:
        return match.group(1), "relation", (2.0, match.group(2), 0.0)
    match = _NEGATED.fullmatch(line)
    if match:
        return match.group(1), "relation", (-1.0, match.group(2), 0.0)
    match = _SHIFTED.fullmatch(line)
    if match:
        return match.group(1), "relation", (1.0, match.group(3), float(match.group(2)))
    match = _AFFINE.fullmatch(line)
    if match:
        return match.group(1), "relation", (float(match.group(2)), match.group(3), float(match.group(4)))
    raise GateError(f"Unrecognized representative coordinate sentence: {line}", "decoder_semantic")


def _orbit_from_relations(element, multiplicity, orbit_id, lines, sg_table):
    clauses = {}
    for line in lines:
        axis, kind, values = _parse_coordinate(line)
        if axis in clauses:
            raise GateError(f"Repeated representative coordinate for axis {axis}", "decoder_semantic")
        clauses[axis] = (kind, values)
    if set(clauses) != set(_AXES):
        raise GateError("Every representative must describe exactly axes a, b, and c", "decoder_semantic")

    free_axes = {axis for axis, (kind, _) in clauses.items() if kind == "free"}
    free_values = {axis: values[0] for axis, (kind, values) in clauses.items() if kind == "free"}
    matrix = [[0.0, 0.0, 0.0] for _ in _AXES]
    shift = [0.0, 0.0, 0.0]
    for row, axis in enumerate(_AXES):
        kind, values = clauses[axis]
        if kind == "free":
            matrix[row][row] = 1.0
        elif kind == "fixed":
            shift[row] = values[0]
        else:
            coefficient, source_axis, constant = values
            if source_axis not in free_axes:
                raise GateError(f"Representative relation along {axis} refers to a non-free coordinate", "decoder_semantic")
            matrix[row][_AXES.index(source_axis)] = coefficient
            shift[row] = constant

    matches = []
    for label, definition in sg_table["definitions"].items():
        if definition["multiplicity"] != multiplicity:
            continue
        public_matrix = tuple(tuple(float(value) for value in row) for row in definition["a"])
        public_shift = tuple(float(value) for value in definition["b"])
        if public_matrix == tuple(tuple(row) for row in matrix) and public_shift == tuple(shift):
            matches.append((label, definition))
    if len(matches) != 1:
        raise GateError("Orbit multiplicity and representative relations do not select one public definition", "decoder_semantic")
    label, definition = matches[0]
    parameters = [free_values[_AXES[axis]] for axis in definition["axes"]]
    return {"id": orbit_id, "element": element, "wyckoff": label, "parameters": parameters}


def decode(text, table=None):
    """Decode only the finite relational grammar emitted by :func:`encode`."""
    table = _table(table)
    if not isinstance(text, str):
        raise GateError("Expected relational English text", "decoder_format")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cursor = 0

    def take():
        nonlocal cursor
        if cursor >= len(lines):
            raise GateError("Relational document ended before all facts were stated", "decoder_format")
        line = lines[cursor]
        cursor += 1
        return line

    match = _CELL.fullmatch(take())
    if not match:
        raise GateError("Input cell sentence is missing or malformed", "decoder_format")
    lattice = [float(value) for value in match.groups()]

    match = _SYMMETRY.fullmatch(take())
    if not match:
        raise GateError("Public symmetry sentence is missing or malformed", "decoder_format")
    sg, hall_symbol = int(match.group(1)), match.group(2)
    if sg not in table:
        raise GateError(f"Unknown space group number: {sg}", "decoder_semantic")

    match = _ATOM_COUNT.fullmatch(take())
    if not match:
        raise GateError("Input atom-count sentence is missing or malformed", "decoder_format")
    atom_count = int(match.group(1))

    composition = {}
    while cursor < len(lines):
        match = _COMPOSITION.fullmatch(lines[cursor])
        if not match:
            break
        symbol = _element_symbol(match.group(1))
        if symbol in composition:
            raise GateError(f"Repeated composition entry for {symbol}", "decoder_semantic")
        composition[symbol] = int(match.group(2))
        cursor += 1
    if not composition:
        raise GateError("Input composition is missing", "decoder_format")

    frame_line = take()
    if frame_line == _IDENTITY_FRAME:
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        frame_shift = [0.0, 0.0, 0.0]
    else:
        matrix, frame_shift = [], []
        for expected_axis in _AXES:
            match = _FRAME_COORDINATE.fullmatch(frame_line)
            if not match or match.group(1) != expected_axis:
                raise GateError("Standard-frame coordinate relation is missing or malformed", "decoder_format")
            matrix.append([float(value) for value in match.groups()[1:4]])
            frame_shift.append(float(match.group(5)))
            if expected_axis != _AXES[-1]:
                frame_line = take()

    rounding_line = take()
    if rounding_line == "The input-frame fractional coordinates are not rounded before any listed corrections.":
        round_digits = None
    else:
        match = _ROUND.fullmatch(rounding_line)
        if not match:
            raise GateError("Coordinate-rounding sentence is missing or malformed", "decoder_format")
        round_digits = int(match.group(1))

    orbits = []
    while cursor < len(lines):
        match = _ORBIT.fullmatch(lines[cursor])
        if not match:
            break
        element = _element_symbol(match.group(1))
        multiplicity, orbit_id = int(match.group(2)), match.group(3)
        cursor += 1
        coordinate_lines = [take(), take(), take()]
        orbit = _orbit_from_relations(element, multiplicity, orbit_id, coordinate_lines, table[sg])
        orbits.append(orbit)
    if not orbits:
        raise GateError("At least one relational orbit is required", "decoder_format")

    corrections = []
    if cursor < len(lines) and lines[cursor] == _NO_CORRECTIONS:
        cursor += 1
    else:
        if cursor >= len(lines):
            raise GateError("An explicit correction-state sentence is required", "decoder_format")
        while cursor < len(lines):
            match = _CORRECTION.fullmatch(take())
            if not match:
                raise GateError("Fractional correction sentence is missing or malformed", "decoder_format")
            site_id, element = match.group(1), _element_symbol(match.group(2))
            corrections.append({"site_id": site_id, "element": element, "delta": [float(value) for value in match.groups()[2:]]})
    if cursor != len(lines):
        raise GateError("Unexpected text after relational facts", "decoder_format")

    return oldcodec.validate_plan(
        {
            "status": "ok",
            "space_group_number": sg,
            "hall_symbol": hall_symbol,
            "lattice": lattice,
            "input_atom_count": atom_count,
            "composition": composition,
            "orbits": orbits,
            "frame_transform": {"matrix": matrix, "origin_shift": frame_shift},
            "round_digits": round_digits,
            "precision_corrections": corrections,
        },
        table,
    )


def normalize(text, table=None):
    """Return canonical controlled English without reading original source facts."""
    table = _table(table)
    plan = decode(text, table=table)
    return encode(oldcodec.source_facts_from_plan(plan), table=table)
