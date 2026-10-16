#!/usr/bin/env python3
"""Build provenance-bound, PLaID-decoded crystal facts as streaming JSONL."""

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from collections import Counter

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

from monty.json import MontyEncoder
from pymatgen.analysis.dimensionality import get_dimensionality_larsen, get_structure_components
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup
from pyxtal.symmetry import Group


ROOT = Path(__file__).resolve().parents[2]
METHOD_PATH = Path(__file__).with_name("feature-method.json")
METHOD = json.loads(METHOD_PATH.read_text(encoding="utf-8"))
METHOD_ID = METHOD["feature_method_id"]
SCHEMA_VERSION = METHOD["schema_version"]
METHOD_SHA256 = hashlib.sha256(METHOD_PATH.read_bytes()).hexdigest()
PLAID_UTILS_PATH = ROOT / METHOD["plaid"]["implementation"]
sys.path.insert(0, str(PLAID_UTILS_PATH.parent))
from utils import get_crystal_string_wyckoff_pyx, parse_fn_wyckoff  # noqa: E402


NN_SETTINGS = METHOD["local_structure_facts"]["neighbor_settings"]
MOTIF_SETTINGS = METHOD["local_structure_facts"]["recognized_motif"]
class RecordFailure(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_runtime():
    for package, expected in METHOD["software_versions"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} version {actual} does not match frozen version {expected}")
    actual_hash = sha256_file(PLAID_UTILS_PATH)
    expected_hash = METHOD["plaid"]["implementation_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(f"PLaID utils.py SHA-256 {actual_hash} does not match frozen version {expected_hash}")


def open_jsonl(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")


def row_property_labels(row):
    value = row.get("energy_above_hull")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RecordFailure("property_label", "energy_above_hull must be a finite numeric source value")
    unit = row.get("energy_above_hull_unit")
    if not isinstance(unit, str) or not unit:
        raise RecordFailure("property_label", "energy_above_hull_unit is missing")
    return {
        "energy_above_hull": value,
        "unit": unit,
        "source_field": "energy_above_hull",
        "provenance": row.get("provenance"),
    }


def base_record(row, input_path, input_sha256, row_index):
    structure = row.get("structure") if isinstance(row, dict) else None
    try:
        structure_hash = canonical_sha256(structure)
    except (TypeError, ValueError):
        structure_hash = None
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_method_id": METHOD_ID,
        "feature_method_sha256": METHOD_SHA256,
        "status": "error",
        "material_id": row.get("material_id") if isinstance(row, dict) else None,
        "split": row.get("split") if isinstance(row, dict) else None,
        "source": {
            "input_path": str(input_path),
            "input_sha256": input_sha256,
            "row_index": row_index,
            "structure_sha256": structure_hash,
        },
        "property_labels": None,
        "plaid": None,
        "core_facts": None,
        "local_structure_facts": None,
    }


def fail_record(record, stage, error):
    record["status"] = "error"
    record["error"] = {
        "stage": stage,
        "type": type(error).__name__,
        "message": str(error)[:1000],
    }
    return record


def parse_encoded_text(encoded_text):
    lines = [line.strip() for line in encoded_text.splitlines() if line.strip()]
    if len(lines) < 6 or not lines[1].startswith("Spacegroup:"):
        raise ValueError("PLaID encoder output is missing its expected header")
    match = re.fullmatch(r"Sites\s*\((\d+)\)", lines[4])
    if match is None:
        raise ValueError("PLaID encoder output has an invalid Sites header")
    rows = []
    for line in lines[5:]:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"PLaID orbit row has {len(parts)} fields, expected 5: {line}")
        multiplicity = re.fullmatch(r"(\d+)[A-Za-z]", parts[4])
        if multiplicity is None:
            raise ValueError(f"PLaID orbit row has an invalid Wyckoff label: {parts[4]}")
        rows.append({"element": parts[0], "wyckoff_label": parts[4], "multiplicity": int(multiplicity.group(1))})
    return {
        "formula": lines[0],
        "space_group_symbol": lines[1].split(":", 1)[1].strip(),
        "header_site_count": int(match.group(1)),
        "orbit_rows": rows,
    }


def element_set(structure):
    return sorted(element.symbol for element in structure.composition.elements)


def reduced_composition_matches(first, second, tolerance=1e-8):
    a = first.composition.reduced_composition.get_el_amt_dict()
    b = second.composition.reduced_composition.get_el_amt_dict()
    return set(a) == set(b) and all(math.isclose(float(a[key]), float(b[key]), rel_tol=tolerance, abs_tol=tolerance) for key in a)


def duplicate_site_pairs(structure, tolerance_angstrom):
    duplicates = []
    for left in range(len(structure)):
        for right in range(left + 1, len(structure)):
            distance, _ = structure.lattice.get_distance_and_image(
                structure[left].frac_coords, structure[right].frac_coords
            )
            if distance <= tolerance_angstrom:
                duplicates.append([left, right])
    return duplicates


def structure_matcher():
    settings = METHOD["consistency_checks"]["source_structure_matches_decoded"]
    return StructureMatcher(
        ltol=settings["ltol"],
        stol=settings["stol"],
        angle_tol=settings["angle_tolerance_degrees"],
        primitive_cell=settings["primitive_cell"],
        scale=settings["scale"],
    )


def space_group_consistency(encoded_symbol, decoded_symbol, decoded_number):
    encoded_number = Group(encoded_symbol).number
    decoded_number = int(decoded_number)
    encoded_canonical = SpaceGroup.from_int_number(encoded_number).symbol
    decoded_canonical = SpaceGroup.from_int_number(decoded_number).symbol
    return {
        "matches": encoded_number == decoded_number and encoded_canonical == decoded_canonical,
        "encoded_space_group_symbol": encoded_symbol,
        "decoded_space_group_symbol": decoded_symbol,
        "encoded_space_group_number": encoded_number,
        "decoded_space_group_number": decoded_number,
        "encoded_space_group_canonical_symbol": encoded_canonical,
        "decoded_space_group_canonical_symbol": decoded_canonical,
    }


def consistency_checks(source, decoded, encoded, parsed, symmetrized, analyzer):
    encoded_symbol = parsed["space_group_symbol"]
    decoded_symbol = analyzer.get_space_group_symbol()
    decoded_number = analyzer.get_space_group_number()
    group_check = space_group_consistency(encoded_symbol, decoded_symbol, decoded_number)
    duplicate_pairs = duplicate_site_pairs(
        decoded,
        METHOD["consistency_checks"]["decoded_sites_unique"]["minimum_periodic_site_distance_angstrom"],
    )
    flags = {
        "encoded_header_sites_matches_decoded_atom_count": parsed["header_site_count"] == len(decoded),
        "encoded_orbit_multiplicity_sum_matches_header": sum(row["multiplicity"] for row in parsed["orbit_rows"]) == parsed["header_site_count"],
        "source_reduced_composition_matches_decoded": reduced_composition_matches(source, decoded),
        "source_element_set_matches_decoded": element_set(source) == element_set(decoded),
        "decoded_cell_volume_positive": math.isfinite(float(decoded.volume)) and decoded.volume > 0,
        "decoded_sites_unique": not duplicate_pairs,
        "encoded_space_group_matches_decoded_analysis": group_check["matches"],
        "encoded_orbit_count_matches_decoded_analysis": len(parsed["orbit_rows"]) == len(symmetrized.equivalent_indices),
        "source_structure_matches_decoded": structure_matcher().fit(source, decoded),
    }
    return {
        **flags,
        "all_scientific_consistency_checks_pass": all(flags.values()),
        "encoded_header_site_count": parsed["header_site_count"],
        "encoded_orbit_multiplicity_sum": sum(row["multiplicity"] for row in parsed["orbit_rows"]),
        "decoded_atom_count": len(decoded),
        "source_atom_count": len(source),
        "encoded_orbit_count": len(parsed["orbit_rows"]),
        "decoded_orbit_count": len(symmetrized.equivalent_indices),
        "encoded_space_group_symbol": group_check["encoded_space_group_symbol"],
        "decoded_space_group_symbol": group_check["decoded_space_group_symbol"],
        "encoded_space_group_canonical_symbol": group_check["encoded_space_group_canonical_symbol"],
        "decoded_space_group_canonical_symbol": group_check["decoded_space_group_canonical_symbol"],
        "encoded_space_group_number": group_check["encoded_space_group_number"],
        "decoded_space_group_number": group_check["decoded_space_group_number"],
        "duplicate_site_pairs": duplicate_pairs,
    }


def core_facts(decoded, symmetrized, analyzer):
    if not decoded.is_ordered:
        raise ValueError("PLaID-decoded structure is not fully ordered")
    composition = decoded.composition
    wyckoff_labels = list(symmetrized.wyckoff_symbols)
    return {
        "cell_formula": decoded.formula,
        "reduced_formula": composition.reduced_formula,
        "elements": element_set(decoded),
        "element_count": len(composition.elements),
        "cell_atom_count": len(decoded),
        "a_A": float(decoded.lattice.a),
        "b_A": float(decoded.lattice.b),
        "c_A": float(decoded.lattice.c),
        "angles_deg": [float(x) for x in decoded.lattice.angles],
        "cell_volume_A3": float(decoded.volume),
        "theoretical_density_g_cm3": float(decoded.density),
        "space_group_number": int(analyzer.get_space_group_number()),
        "space_group_symbol": analyzer.get_space_group_symbol(),
        "crystal_system": analyzer.get_crystal_system(),
        "independent_orbit_count": len(symmetrized.equivalent_indices),
        "wyckoff_labels": wyckoff_labels,
        "wyckoff_multiplicities": [len(indices) for indices in symmetrized.equivalent_indices],
    }


def motif_order_parameters(site, neighbors):
    if not neighbors:
        return {}
    labels = ["cn", "qtet", "qoct", "qbcc", "q6", "qsqpyr", "qtribipyr"]
    params = LocalStructOrderParams(MOTIF_SETTINGS["order_parameter_types"])
    values = params.get_order_parameters(
        [site, *neighbors], 0, indices_neighs=list(range(1, len(neighbors) + 1))
    )
    return {name: None if value is None else float(value) for name, value in zip(labels, values, strict=True)}


def classify_motif(coordination_number, scores):
    if not scores:
        return {"classification": "unrecognized", "label": None, "score": None, "threshold": None, "candidates": []}
    cn_score = scores.get("cn")
    if cn_score is None or not math.isfinite(float(cn_score)):
        return {"classification": "unrecognized", "label": None, "score": None, "threshold": None, "candidates": []}
    motif_cn = int(float(cn_score) + 0.5)
    if motif_cn != coordination_number:
        return {"classification": "cn_mismatch", "label": None, "score": None, "threshold": None, "candidates": []}

    thresholds = MOTIF_SETTINGS["thresholds"]

    def above(name, threshold):
        value = scores.get(name)
        return value is not None and math.isfinite(float(value)) and float(value) > threshold

    def below(name, threshold):
        value = scores.get(name)
        return value is not None and math.isfinite(float(value)) and float(value) < threshold

    candidates = []
    rules = [
        ("tetrahedral", 4, "qtet", thresholds["qtet"], above("qtet", thresholds["qtet"])),
        ("square pyramidal", 5, "qsqpyr", thresholds["qsqpyr"], above("qsqpyr", thresholds["qsqpyr"])),
        ("trigonal bipyramidal", 5, "qtribipyr", thresholds["qtribipyr"], above("qtribipyr", thresholds["qtribipyr"])),
        ("octahedral", 6, "qoct", thresholds["qoct"], above("qoct", thresholds["qoct"])),
        (
            "bcc", 8, "qbcc", thresholds["qbcc"],
            above("qbcc", thresholds["qbcc"]) and below("qtet", thresholds["qtet"]),
        ),
        (
            "cp", 12, "q6", thresholds["q6"],
            above("q6", thresholds["q6"])
            and all(below(name, thresholds["q6"]) for name in ("qtet", "qoct", "qbcc")),
        ),
    ]
    for name, expected_cn, score_key, threshold, passes in rules:
        if coordination_number == expected_cn and passes:
            candidates.append({"label": name, "score": float(scores[score_key]), "threshold": threshold})
    if len(candidates) == 1:
        candidate = candidates[0]
        return {"classification": candidate["label"], "label": candidate["label"],
                "score": candidate["score"], "threshold": candidate["threshold"], "candidates": candidates}
    if len(candidates) > 1:
        return {"classification": "multiple assignments", "label": None, "score": None,
                "threshold": None, "candidates": candidates}
    return {"classification": "unrecognized", "label": None, "score": None, "threshold": None, "candidates": []}


def local_structure_facts(decoded, symmetrized):
    nn = CrystalNN(**NN_SETTINGS)
    environments = []
    for orbit_index, indices in enumerate(symmetrized.equivalent_indices):
        site_index = indices[0]
        neighbors = nn.get_nn_info(decoded, site_index)
        distances = [float(neighbor["site"].nn_distance) for neighbor in neighbors]
        neighbor_sites = [neighbor["site"] for neighbor in neighbors]
        motif_scores = motif_order_parameters(decoded[site_index], neighbor_sites)
        classification = classify_motif(len(neighbors), motif_scores)
        crystalnn_scores = nn.get_local_order_parameters(decoded, site_index) or {}
        environments.append({
            "orbit_index": orbit_index,
            "representative_site_index": site_index,
            "center_element": decoded[site_index].specie.symbol,
            "orbit_multiplicity": len(indices),
            "coordination_number": len(neighbors),
            "neighbor_element_counts": dict(sorted(Counter(item["site"].specie.symbol for item in neighbors).items())),
            "neighbor_distance_range_A": [round(min(distances), 4), round(max(distances), 4)] if distances else None,
            "recognized_motif": classification["label"],
            "recognized_motif_score": classification["score"],
            "recognized_motif_threshold": classification["threshold"],
            "motif_classification": classification["classification"],
            "motif_candidates": classification["candidates"],
            "motif_order_parameter_scores": motif_scores,
            "crystalnn_order_parameter_scores": {key: float(value) for key, value in crystalnn_scores.items()},
        })

    graph = StructureGraph.from_local_env_strategy(decoded, nn, weights=False, edge_properties=False)
    component_rows = get_structure_components(graph, inc_site_ids=True)
    component_dimensions = [int(component["dimensionality"]) for component in component_rows]
    return {
        "neighbor_method": "CrystalNN geometry-only",
        "symmetry_orbit_count": len(symmetrized.equivalent_indices),
        "coordination_environments": environments,
        "connectivity": {
            "maximum_periodic_dimension": int(get_dimensionality_larsen(graph)),
            "component_dimensions": component_dimensions,
            "component_site_counts": [len(component["site_ids"]) for component in component_rows],
        },
        "polyhedra_sharing": {"status": "not_computed"},
    }


def process_record(row, input_path, input_sha256, row_index):
    record = base_record(row, input_path, input_sha256, row_index)
    if not isinstance(row, dict):
        return fail_record(record, "json_record", ValueError("JSONL value must be an object"))
    try:
        record["property_labels"] = row_property_labels(row)
        stage = "source_structure"
        source = Structure.from_dict(row["structure"])
        if not source.is_ordered:
            raise RecordFailure(stage, "source structure is not fully ordered")
        record["source"].update({
            "source_atom_count": len(source),
            "source_formula": source.formula,
            "source_reduced_formula": source.composition.reduced_formula,
            "source_elements": element_set(source),
        })

        stage = "plaid_encode"
        encoder_output = io.StringIO()
        with contextlib.redirect_stdout(encoder_output):
            encoded = get_crystal_string_wyckoff_pyx(
                source.to(fmt="cif"),
                tol=METHOD["plaid"]["encode_arguments"]["tol"],
                translate=METHOD["plaid"]["encode_arguments"]["translate"],
            )
        record["plaid"] = {
            "encoded_text": encoded,
            "decoded_structure": None,
            "consistency": None,
            "encoder_fallback_used": "failed pyxtal 0.01 trying 0.0001 tol" in encoder_output.getvalue(),
        }
        if encoder_output.getvalue():
            record["plaid"]["encoder_stdout"] = encoder_output.getvalue().strip()[:1000]

        stage = "plaid_text_validation"
        parsed = parse_encoded_text(encoded)

        stage = "plaid_decode"
        decoded = parse_fn_wyckoff(encoded, to_cif=False)
        record["plaid"]["decoded_structure"] = decoded.as_dict()
        analyzer = SpacegroupAnalyzer(decoded, symprec=0.01, angle_tolerance=5)
        symmetrized = analyzer.get_symmetrized_structure()

        stage = "consistency"
        consistency = consistency_checks(source, decoded, encoded, parsed, symmetrized, analyzer)
        record["plaid"]["consistency"] = consistency
        if not consistency["all_scientific_consistency_checks_pass"]:
            failed = [key for key, value in consistency.items() if isinstance(value, bool) and not value]
            raise RecordFailure(stage, f"PLaID consistency checks failed: {', '.join(failed)}")

        stage = "core_facts"
        record["core_facts"] = core_facts(decoded, symmetrized, analyzer)
        if not all(math.isfinite(float(value)) for value in (
            record["core_facts"]["a_A"], record["core_facts"]["b_A"], record["core_facts"]["c_A"],
            record["core_facts"]["cell_volume_A3"], record["core_facts"]["theoretical_density_g_cm3"],
        )):
            raise ValueError("decoded core fact contains a non-finite numeric value")

        stage = "local_structure_facts"
        record["local_structure_facts"] = local_structure_facts(decoded, symmetrized)
        record["status"] = "ok"
        return record
    except RecordFailure as exc:
        return fail_record(record, exc.stage, exc)
    except Exception as exc:
        return fail_record(record, locals().get("stage", "record"), exc)


def write_json_line(stream, record):
    stream.write(json.dumps(record, cls=MontyEncoder, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def process_file(input_path, output_path, shard_index=0, shards=1, limit=None):
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("shard index must satisfy 0 <= shard-index < shards")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")

    validate_runtime()
    input_hash = sha256_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    processed = Counter()
    selected_rows = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output, open_jsonl(input_path) as source:
            for row_index, raw_line in enumerate(source):
                if row_index % shards != shard_index:
                    continue
                if limit is not None and selected_rows >= limit:
                    break
                selected_rows += 1
                row = None
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    record = base_record(None, input_path, input_hash, row_index)
                    record = fail_record(record, "json_decode", exc)
                else:
                    record = process_record(row, input_path, input_hash, row_index)
                try:
                    write_json_line(output, record)
                except (TypeError, ValueError) as exc:
                    serialization_error = base_record(row, input_path, input_hash, row_index)
                    if isinstance(row, dict):
                        try:
                            serialization_error["property_labels"] = row_property_labels(row)
                        except Exception:
                            pass
                    record = fail_record(serialization_error, "json_serialization", exc)
                    write_json_line(output, record)
                processed[record["status"]] += 1
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {"input_sha256": input_hash, "output": str(output_path), "shard_index": shard_index,
            "shards": shards, "selected_rows": selected_rows, "status_counts": dict(processed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="plain or gzip-compressed JSONL source")
    parser.add_argument("--output", required=True, help="new JSONL output path for this shard")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="maximum selected rows after sharding")
    args = parser.parse_args(argv)
    try:
        receipt = process_file(args.input, args.output, args.shard_index, args.shards, args.limit)
    except Exception as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
