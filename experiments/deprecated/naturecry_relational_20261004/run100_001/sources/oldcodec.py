"""Typed inline facts and strict blind-decoder gates for NatureCry."""
from collections import Counter
from decimal import Decimal
import json
import math
import numbers
import re
import xml.etree.ElementTree as ET

import spglib


CODEC_VERSION = "naturecry-inline-v1"
ENCODER_PROMPT = """Write a natural scientific description of the supplied crystal facts. Choose the wording, paragraph order, and spatial organization yourself. Explain the symmetry, independent Wyckoff orbits, original input cell, coordinate frames, and any precision corrections as a coherent spatial narrative. State only spatial relationships supported by the supplied facts or public crystallographic rules. Do not invent bonding, coordination, stability, or other properties. Preserve small differences between values; do not call them exactly equal when they are not. Describe precision corrections as real deviations from the idealized backbone, not as part of the ideal structure.

Place every supplied reference exactly once at the point where its fact belongs. Preserve references verbatim in the form [[param:NAME]]. Do not state protected numeric or identifier values outside their references, omit references, add parameter tags, or append a table, CIF, atom list, or JSON. Describe relationships in words, while keeping the independent facts exact. Return prose only."""

DECODER_PROMPT = """You are a blind semantic decoder. The inline crystal document in the user message is your only material-specific source. Check that its natural-language spatial claims agree with the typed values and public crystallographic rules. If prose, tags, and public rules conflict, return status=\"conflict\" and identify the contradiction; do not silently privilege a false sentence or repair it. Do not use external retrieval, infer missing facts, repair values, or assume a default for any field. Return exactly one JSON object, with no Markdown fences.

For a complete, consistent description return status=\"ok\" and exactly these keys: status, space_group_number, hall_symbol, lattice (six input-cell values in a,b,c,alpha,beta,gamma order), input_atom_count, composition (element-to-positive-integer counts), orbits (objects with id, element, wyckoff, and parameters), frame_transform (matrix and origin_shift), round_digits (integer or null), and precision_corrections (objects with site_id, element, and a three-number input-frame fractional delta). Orbit parameters follow the public table's listed free-axis order; do not substitute a fitted equivalent representative. Preserve every number and identifier exactly as tagged. Keep distinct same-element orbits distinct.

If facts are absent or contradictory, return only {\"status\":\"insufficient_information\",\"reason\":\"...\"} or {\"status\":\"conflict\",\"reason\":\"...\"}."""

DECODER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "NatureCry inline decoder plan",
    "type": "object",
    "required": ["status", "space_group_number", "hall_symbol", "lattice", "input_atom_count", "composition", "orbits", "frame_transform", "round_digits", "precision_corrections"],
    "additionalProperties": False,
    "properties": {
        "status": {"const": "ok"},
        "space_group_number": {"type": "integer", "minimum": 1, "maximum": 230},
        "hall_symbol": {"type": "string"},
        "lattice": {"type": "array", "minItems": 6, "maxItems": 6, "items": {"type": "number"}},
        "input_atom_count": {"type": "integer", "minimum": 1},
        "composition": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "integer", "minimum": 1}},
        "orbits": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["id", "element", "wyckoff", "parameters"], "additionalProperties": False, "properties": {"id": {"type": "string"}, "element": {"type": "string"}, "wyckoff": {"type": "string"}, "parameters": {"type": "array", "items": {"type": "number"}}}}},
        "frame_transform": {"type": "object", "required": ["matrix", "origin_shift"], "additionalProperties": False, "properties": {"matrix": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}}, "origin_shift": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}}},
        "round_digits": {"type": ["integer", "null"], "minimum": 0, "maximum": 15},
        "precision_corrections": {"type": "array", "items": {"type": "object", "required": ["site_id", "element", "delta"], "additionalProperties": False, "properties": {"site_id": {"type": "string"}, "element": {"type": "string"}, "delta": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}}}},
    },
}

_REF = re.compile(r"\[\[param:([A-Za-z0-9_.-]+)\]\]")
_TAG = re.compile(r"<param\b[^>]*>.*?</param>", re.DOTALL)
_FLOAT_VECTOR = re.compile(r"float64\[(\d+(?:,\d+)*)\]\Z")
_ORBIT_ID = re.compile(r"o\d{3,}\Z")
_SITE_ID = re.compile(r"o\d{3,}_\d+\Z")
_FIELD_ATTRIBUTES = {"name", "type", "unit", "frame"}
_PLAN_KEYS = set(DECODER_SCHEMA["required"])
_CIF_MARKER = re.compile(r"(?im)^\s*(?:data_[\w.-]*|loop_|save_[\w.-]+|_[A-Za-z0-9_]+)")
_KEY_VALUE_LINE = re.compile(r"(?m)^\s*[A-Za-z_][\w.-]*\s*:\s*\S+")
_MARKDOWN_LINE = re.compile(r"(?m)^\s*(?:#{1,6}\s|[-*+]\s|>\s|\|)")


class GateError(ValueError):
    def __init__(self, message, stage="format"):
        super().__init__(message)
        self.stage = stage


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"Duplicate JSON key: {key}", "decoder_format")
        result[key] = value
    return result


def _bad_constant(value):
    raise GateError(f"Non-finite JSON number: {value}", "decoder_format")


def strict_json_loads(text):
    if not isinstance(text, str) or "```" in text:
        raise GateError("Expected unfenced JSON text", "decoder_format")
    try:
        value = json.loads(text, object_pairs_hook=_unique_object, parse_float=Decimal, parse_constant=_bad_constant)
    except GateError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GateError(f"Invalid JSON: {error}", "decoder_format") from error
    if not isinstance(value, dict):
        raise GateError("JSON root must be an object", "decoder_format")
    return value


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (numbers.Real, Decimal)):
        raise GateError(f"{label} must be a JSON number", "decoder_schema")
    result = float(value)
    if not math.isfinite(result):
        raise GateError(f"{label} must be finite", "decoder_schema")
    return result


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise GateError(f"{label} must be a positive JSON integer", "decoder_schema")
    return value


def _normalise_typed(value, kind, label):
    if kind == "string":
        if not isinstance(value, str):
            raise GateError(f"{label} must be a string", "parameter_schema")
        return value
    if kind == "int":
        if type(value) is not int:
            raise GateError(f"{label} must be an integer", "parameter_schema")
        return value
    if kind == "null|int":
        if value is None or type(value) is int:
            return value
        raise GateError(f"{label} must be an integer or null", "parameter_schema")
    if kind == "composition_count":
        if not isinstance(value, dict) or set(value) != {"element", "count"}:
            raise GateError(f"{label} must contain element and count", "parameter_schema")
        if not isinstance(value["element"], str) or type(value["count"]) is not int or value["count"] <= 0:
            raise GateError(f"{label} has an invalid element count", "parameter_schema")
        return value
    if kind == "float64":
        return _number(value, label)
    match = _FLOAT_VECTOR.fullmatch(kind)
    if match:
        shape = tuple(map(int, match.group(1).split(",")))
        def visit(node, depth, prefix):
            if depth == len(shape):
                return _number(node, prefix)
            if not isinstance(node, list) or len(node) != shape[depth]:
                raise GateError(f"{prefix} has the wrong shape", "parameter_schema")
            return [visit(item, depth + 1, f"{prefix}[{i}]") for i, item in enumerate(node)]
        return visit(value, 0, label)
    raise GateError(f"Unknown parameter type: {kind}", "parameter_schema")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _field(kind, unit, frame, value):
    return {"type": kind, "unit": unit, "frame": frame, "value": _normalise_typed(value, kind, "inventory value")}


def _site_order(site_id):
    orbit, site = site_id.rsplit("_", 1)
    return orbit, int(site)


def make_inventory(facts):
    """Build exact inline fields from prepared facts; no source atom table is needed."""
    fields = {}
    lattice = facts["input_lattice_parameters"]
    for name, value, unit in zip(("a", "b", "c", "alpha", "beta", "gamma"), lattice,
                                 ("angstrom", "angstrom", "angstrom", "degree", "degree", "degree")):
        fields[f"cell.{name}"] = _field("float64", unit, "input", value)
    fields["space_group.number"] = _field("int", "identifier", "public", facts["space_group_number"])
    fields["space_group.hall"] = _field("string", "Hall_symbol", "public", facts["hall_symbol"])
    fields["input.atom_count"] = _field("int", "atom", "input", facts["input_atoms"])
    for element, count in sorted(facts["input_composition"].items()):
        fields[f"composition.{element}"] = _field("composition_count", "atom", "input", {"element": element, "count": count})
    for orbit in sorted(facts["orbits"], key=lambda item: item["id"]):
        prefix = f"orbit.{orbit['id']}"
        fields[f"{prefix}.id"] = _field("string", "orbit_identifier", "public_expansion", orbit["id"])
        fields[f"{prefix}.element"] = _field("string", "element", "orbit", orbit["element"])
        fields[f"{prefix}.wyckoff"] = _field("string", "Wyckoff_label", "public_table", orbit["wyckoff"])
        if orbit["parameters"]:
            fields[f"{prefix}.parameters"] = _field(f"float64[{len(orbit['parameters'])}]", "fractional", "standard", orbit["parameters"])
    frame = facts["frame_transform"]
    fields["frame.matrix"] = _field("float64[3,3]", "1", "standard_from_input", frame["matrix"])
    fields["frame.origin_shift"] = _field("float64[3]", "fractional", "standard", frame["origin_shift"])
    fields["round_digits"] = _field("null|int", "decimal_places", "input_fractional", facts["round_digits"])
    for index, correction in enumerate(sorted(facts["precision_corrections"], key=lambda item: _site_order(item["site_id"]))):
        prefix = f"correction.c{index:03d}"
        fields[f"{prefix}.site_id"] = _field("string", "expanded_site_identifier", "input_cell", correction["site_id"])
        fields[f"{prefix}.element"] = _field("string", "element", "input_site", correction["element"])
        fields[f"{prefix}.delta"] = _field("float64[3]", "fractional", "input_delta", correction["delta"])
    return fields


def _param_markup(name, field):
    node = ET.Element("param", {"name": name, "type": field["type"], "unit": field["unit"], "frame": field["frame"]})
    node.text = _json(field["value"])
    return ET.tostring(node, encoding="unicode", short_empty_elements=False)


def _validate_plain_prose(text):
    if not text.strip() or not re.search(r"[A-Za-z]", text):
        raise GateError("Inline document must contain natural prose", "encoder_format")
    if any(char.isnumeric() for char in text):
        raise GateError("Exact numeric values must appear only inside typed parameters", "encoder_untagged_number")
    if any(char in "{}[]|=`" for char in text) or _CIF_MARKER.search(text) or _KEY_VALUE_LINE.search(text) or _MARKDOWN_LINE.search(text):
        raise GateError("Structured or tabular appendix is not allowed in inline prose", "encoder_structure")


def bind_inline(narrative, inventory):
    """Replace each encoder reference in place with one escaped typed value."""
    if not isinstance(narrative, str) or "<" in narrative or ">" in narrative:
        raise GateError("Encoder prose contains markup", "encoder_format")
    names = list(_REF.findall(narrative))
    counts = Counter(names)
    if set(counts) != set(inventory) or any(count != 1 for count in counts.values()) or "[[param:" in _REF.sub("", narrative):
        raise GateError("Encoder references must cover the inventory exactly once", "encoder_inventory")
    _validate_plain_prose(_REF.sub("", narrative))
    document = _REF.sub(lambda match: _param_markup(match.group(1), inventory[match.group(1)]), narrative)
    validate_encoder_document(document, inventory)
    return document


def parse_inline_parameters(document):
    if not isinstance(document, str) or "```" in document:
        raise GateError("Inline document must be plain text", "encoder_format")
    matches = list(_TAG.finditer(document))
    remainder = _TAG.sub("", document)
    if "<" in remainder or ">" in remainder:
        raise GateError("Unknown or injected markup in inline document", "encoder_format")
    _validate_plain_prose(remainder)
    fields = {}
    for match in matches:
        try:
            node = ET.fromstring(match.group(0))
        except ET.ParseError as error:
            raise GateError("Malformed parameter tag", "encoder_format") from error
        if node.tag != "param" or list(node) or set(node.attrib) != _FIELD_ATTRIBUTES:
            raise GateError("Parameter tag has nested content or invalid attributes", "encoder_schema")
        name = node.attrib["name"]
        if name in fields:
            raise GateError(f"Duplicate parameter: {name}", "encoder_inventory")
        try:
            raw = strict_json_loads(node.text or "null") if (node.text or "").lstrip().startswith("{") else json.loads(node.text or "null", parse_float=Decimal, parse_constant=_bad_constant)
        except (json.JSONDecodeError, GateError) as error:
            raise GateError(f"Invalid value for parameter {name}", "encoder_value") from error
        kind = node.attrib["type"]
        value = _normalise_typed(raw, kind, name)
        fields[name] = {"type": kind, "unit": node.attrib["unit"], "frame": node.attrib["frame"], "value": value}
    return fields


def validate_encoder_document(document, inventory):
    fields = parse_inline_parameters(document)
    if set(fields) != set(inventory):
        raise GateError("Inline document is missing or adds inventory fields", "encoder_inventory")
    for name, expected in inventory.items():
        actual = fields[name]
        if any(actual[key] != expected[key] for key in ("type", "unit", "frame")):
            raise GateError(f"Typed metadata mismatch for {name}", "encoder_metadata")
        if _json(actual["value"]) != _json(expected["value"]):
            raise GateError(f"Value mismatch for {name}", "encoder_value")
    return {name: field["value"] for name, field in fields.items()}


def source_facts_from_plan(plan):
    return {
        "input_lattice_parameters": plan["lattice"],
        "input_composition": plan["composition"],
        "input_atoms": plan["input_atom_count"],
        "space_group_number": plan["space_group_number"],
        "hall_symbol": plan["hall_symbol"],
        "frame_transform": plan["frame_transform"],
        "round_digits": plan["round_digits"],
        "orbits": plan["orbits"],
        "precision_corrections": plan["precision_corrections"],
    }


def plan_from_facts(facts):
    return {
        "status": "ok",
        "space_group_number": facts["space_group_number"],
        "hall_symbol": facts["hall_symbol"],
        "lattice": facts["input_lattice_parameters"],
        "input_atom_count": facts["input_atoms"],
        "composition": facts["input_composition"],
        "orbits": facts["orbits"],
        "frame_transform": facts["frame_transform"],
        "round_digits": facts["round_digits"],
        "precision_corrections": facts["precision_corrections"],
    }


def _element(value, label):
    if not isinstance(value, str):
        raise GateError(f"{label} must be an element symbol", "decoder_schema")
    from pymatgen.core import Element
    try:
        if Element(value).symbol != value:
            raise ValueError(value)
    except ValueError as error:
        raise GateError(f"Invalid element symbol in {label}: {value}", "decoder_schema") from error
    return value


def validate_plan(value, table):
    if set(value) != _PLAN_KEYS:
        missing, extra = _PLAN_KEYS - set(value), set(value) - _PLAN_KEYS
        raise GateError(f"Decoder plan keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}", "decoder_schema")
    if value["status"] != "ok":
        raise GateError("Decoder did not return a complete plan", "decoder_semantic")
    sg = value["space_group_number"]
    if type(sg) is not int or sg not in table:
        raise GateError("Invalid space-group number", "decoder_schema")
    hall_symbol = value["hall_symbol"]
    expected_hall = spglib.get_spacegroup_type(int(table[sg]["hall"])).hall_symbol
    if not isinstance(hall_symbol, str) or hall_symbol != expected_hall:
        raise GateError("Hall symbol does not match the public space-group setting", "decoder_semantic")
    lattice = value["lattice"]
    if not isinstance(lattice, list) or len(lattice) != 6:
        raise GateError("Lattice must contain six input-cell parameters", "decoder_schema")
    lattice = [_number(item, "lattice") for item in lattice]
    if min(lattice[:3]) <= 0 or not all(0 < angle < 180 for angle in lattice[3:]):
        raise GateError("Invalid lattice lengths or angles", "decoder_schema")
    atom_count = _positive_int(value["input_atom_count"], "input_atom_count")
    composition = value["composition"]
    if not isinstance(composition, dict) or not composition:
        raise GateError("Composition must be a nonempty element-count object", "decoder_schema")
    composition = {_element(element, "composition key"): _positive_int(count, f"composition.{element}") for element, count in composition.items()}
    if len(composition) != len(value["composition"]) or sum(composition.values()) != atom_count:
        raise GateError("Composition does not match input_atom_count", "decoder_semantic")
    if not isinstance(value["orbits"], list) or not value["orbits"]:
        raise GateError("At least one independent orbit is required", "decoder_schema")
    orbits = []
    orbit_ids = set()
    for index, item in enumerate(value["orbits"]):
        if not isinstance(item, dict) or set(item) != {"id", "element", "wyckoff", "parameters"}:
            raise GateError(f"Invalid orbit object at index {index}", "decoder_schema")
        orbit_id, element, label = item["id"], _element(item["element"], "orbit element"), item["wyckoff"]
        if not isinstance(orbit_id, str) or not _ORBIT_ID.fullmatch(orbit_id) or orbit_id in orbit_ids:
            raise GateError(f"Invalid or duplicate orbit id: {orbit_id}", "decoder_schema")
        orbit_ids.add(orbit_id)
        if element not in composition or not isinstance(label, str) or label not in table[sg]["definitions"]:
            raise GateError(f"Unknown orbit element or Wyckoff label: {label}", "decoder_semantic")
        parameters = item["parameters"]
        axes = table[sg]["definitions"][label]["axes"]
        if not isinstance(parameters, list) or len(parameters) != len(axes):
            raise GateError(f"Orbit {orbit_id} must give the exact public free-axis vector", "decoder_schema")
        orbits.append({"id": orbit_id, "element": element, "wyckoff": label,
                       "parameters": [_number(number, f"orbit {orbit_id} parameter") for number in parameters]})
    frame = value["frame_transform"]
    if not isinstance(frame, dict) or set(frame) != {"matrix", "origin_shift"}:
        raise GateError("Frame transform must state matrix and origin_shift", "decoder_schema")
    matrix = frame["matrix"]
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in matrix):
        raise GateError("Frame matrix must be 3 by 3", "decoder_schema")
    matrix = [[_number(number, "frame matrix") for number in row] for row in matrix]
    shift = frame["origin_shift"]
    if not isinstance(shift, list) or len(shift) != 3:
        raise GateError("Frame origin_shift must have three values", "decoder_schema")
    shift = [_number(number, "frame origin_shift") for number in shift]
    import numpy as np
    matrix_array = np.asarray(matrix, dtype=float)
    if abs(float(np.linalg.det(matrix_array))) < 1e-12 or np.max(np.abs(matrix_array)) > 4 or max(map(abs, shift)) > 4:
        raise GateError("Frame transform is singular or outside the frozen support", "decoder_schema")
    digits = value["round_digits"]
    if digits is not None and (type(digits) is not int or not 0 <= digits <= 15):
        raise GateError("round_digits must be explicitly stated as 0-15 or null", "decoder_schema")
    corrections = value["precision_corrections"]
    if not isinstance(corrections, list):
        raise GateError("precision_corrections must be an array", "decoder_schema")
    clean_corrections, targets = [], set()
    for item in corrections:
        if not isinstance(item, dict) or set(item) != {"site_id", "element", "delta"}:
            raise GateError("Invalid precision correction object", "decoder_schema")
        site_id, element = item["site_id"], _element(item["element"], "correction element")
        if not isinstance(site_id, str) or not _SITE_ID.fullmatch(site_id) or site_id in targets:
            raise GateError(f"Invalid or repeated correction site_id: {site_id}", "decoder_schema")
        targets.add(site_id)
        delta = item["delta"]
        if not isinstance(delta, list) or len(delta) != 3:
            raise GateError(f"Correction {site_id} requires a complete three-component vector", "decoder_schema")
        clean_corrections.append({"site_id": site_id, "element": element,
                                  "delta": [_number(number, f"correction {site_id} delta") for number in delta]})
    return {"status": "ok", "space_group_number": sg, "hall_symbol": hall_symbol, "lattice": lattice,
            "input_atom_count": atom_count, "composition": composition, "orbits": orbits,
            "frame_transform": {"matrix": matrix, "origin_shift": shift}, "round_digits": digits,
            "precision_corrections": clean_corrections}


def decode_plan_response(response, inline_document, table):
    value = strict_json_loads(response)
    status = value.get("status")
    if status == "insufficient_information" or status == "conflict":
        if set(value) != {"status", "reason"} or not isinstance(value["reason"], str):
            raise GateError("Invalid decoder refusal object", "decoder_schema")
        raise GateError(value["reason"], "decoder_semantic")
    plan = validate_plan(value, table)
    try:
        validate_encoder_document(inline_document, make_inventory(source_facts_from_plan(plan)))
    except GateError as error:
        raise GateError(str(error), "decoder_inline_mismatch") from error
    return plan


def fixture_narrative(facts):
    """Deterministic binding fixture; it is not an LLM-generated description."""
    inventory = make_inventory(facts)
    refs = lambda prefix: "[[param:" + prefix + "]]"
    lines = [
        "The input cell has edge lengths " + ", ".join(refs(f"cell.{axis}") for axis in ("a", "b", "c")) +
        " angstrom and angles " + ", ".join(refs(f"cell.{axis}") for axis in ("alpha", "beta", "gamma")) + " degrees.",
        "Its symmetry is space group " + refs("space_group.number") + " in Hall setting " + refs("space_group.hall") + ".",
        "The cell contains " + refs("input.atom_count") + " atoms in total.",
    ]
    for element in sorted(facts["input_composition"]):
        lines.append("The declared element count is " + refs(f"composition.{element}") + ".")
    for orbit in sorted(facts["orbits"], key=lambda item: item["id"]):
        prefix = f"orbit.{orbit['id']}"
        line = "Independent orbit " + refs(f"{prefix}.id") + " contains element " + refs(f"{prefix}.element") + " at Wyckoff position " + refs(f"{prefix}.wyckoff")
        if orbit["parameters"]:
            line += " with standard-frame free coordinates " + refs(f"{prefix}.parameters")
        lines.append(line + ".")
    lines.extend((
        "The standard fractional frame is obtained from the input frame by applying matrix " + refs("frame.matrix") +
        " and adding origin shift " + refs("frame.origin_shift") + ".",
        "Input fractional coordinates are rounded to " + refs("round_digits") + " decimal places before corrections.",
    ))
    corrections = sorted(facts["precision_corrections"], key=lambda item: _site_order(item["site_id"]))
    for index, _ in enumerate(corrections):
        prefix = f"correction.c{index:03d}"
        lines.append("Precision correction for retained site " + refs(f"{prefix}.site_id") + " of element " + refs(f"{prefix}.element") +
                     " is the input-frame fractional displacement " + refs(f"{prefix}.delta") + ".")
    if set(_REF.findall("\n".join(lines))) != set(inventory):
        raise GateError("Fixture narrative does not cover its inventory", "preflight_fixture")
    return "\n".join(lines)
