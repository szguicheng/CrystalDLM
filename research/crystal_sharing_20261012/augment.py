#!/usr/bin/env python3
"""Stream shared-polyhedron facts onto existing decoded crystal JSONL rows."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

from pymatgen.core import Structure

from sharing import extract_polyhedra_sharing


ROOT = Path(__file__).resolve().parents[2]
METHOD_PATH = Path(__file__).with_name("sharing-method.json")
SOURCE_METHOD_PATH = ROOT / "research/crystal_preannotation_20261012/feature-method.json"
METHOD = json.loads(METHOD_PATH.read_text(encoding="utf-8"))
METHOD_SHA256 = hashlib.sha256(METHOD_PATH.read_bytes()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_method():
    source = json.loads(SOURCE_METHOD_PATH.read_text(encoding="utf-8"))
    expected_id = METHOD["source_feature_method"]["method_id"]
    expected_hash = METHOD["source_feature_method"]["sha256"]
    if source["feature_method_id"] != expected_id:
        raise RuntimeError("source feature method ID changed after sharing method was frozen")
    actual_hash = sha256_file(SOURCE_METHOD_PATH)
    if actual_hash != expected_hash:
        raise RuntimeError(f"source feature method SHA-256 {actual_hash} does not match {expected_hash}")
    return source


def validate_runtime():
    for package, expected in METHOD["software_versions"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} version {actual} does not match frozen version {expected}")
    return load_source_method()


def sharing_error(error):
    return {
        "status": "error",
        "method_id": METHOD["sharing_method_id"],
        "method_sha256": METHOD_SHA256,
        "error_type": type(error).__name__,
        "error_message": str(error)[:1000],
    }


def augment_record(row, source_method):
    if not isinstance(row, dict) or row.get("status") != "ok":
        return row, "preserved_non_ok"
    local = row.get("local_structure_facts")
    if not isinstance(local, dict):
        local = {}
        row["local_structure_facts"] = local

    existing = local.get("polyhedra_sharing")
    if isinstance(existing, dict) and existing.get("status") in {"computed", "not_applicable", "ambiguous"}:
        return row, "already_populated"
    try:
        if (
            row.get("feature_method_id") != source_method["feature_method_id"]
            or row.get("feature_method_sha256") != METHOD["source_feature_method"]["sha256"]
        ):
            raise ValueError("input facts do not match the frozen source feature method")
        structure = Structure.from_dict(row["plaid"]["decoded_structure"])
        local["polyhedra_sharing"] = extract_polyhedra_sharing(
            structure=structure,
            local_structure_facts=local,
            neighbor_settings=source_method["local_structure_facts"]["neighbor_settings"],
            symprec_angstrom=source_method["core_facts"]["symmetry_analysis"]["symprec_angstrom"],
            angle_tolerance_degrees=source_method["core_facts"]["symmetry_analysis"]["angle_tolerance_degrees"],
            tolerance_angstrom=METHOD["geometry"]["tolerance_angstrom"],
            method_id=METHOD["sharing_method_id"],
            method_sha256=METHOD_SHA256,
            source_method_sha256=METHOD["source_feature_method"]["sha256"],
        )
        return row, local["polyhedra_sharing"]["status"]
    except Exception as error:
        local["polyhedra_sharing"] = sharing_error(error)
        return row, "error"


def open_jsonl(path, mode, compressed=None):
    path = Path(path)
    if compressed is None:
        compressed = path.suffix == ".gz"
    if compressed:
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode + "t", encoding="utf-8")


def augment_file(input_path, output_path, limit=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    source_method = validate_runtime()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    summary = {
        "input_rows": 0,
        "preserved_non_ok": 0,
        "already_populated": 0,
        "computed": 0,
        "not_applicable": 0,
        "ambiguous": 0,
        "error": 0,
        "relation_count_sum_across_rows": 0,
        "sharing_type_counts": {"corner": 0, "edge": 0, "face": 0},
    }
    try:
        with open_jsonl(input_path, "r") as source, open_jsonl(
            temporary_name, "w", compressed=output_path.suffix == ".gz"
        ) as target:
            for line in source:
                if limit is not None and summary["input_rows"] >= limit:
                    break
                row = json.loads(line)
                row, status = augment_record(row, source_method)
                summary["input_rows"] += 1
                summary[status] += 1
                if status in {"computed", "not_applicable", "ambiguous"}:
                    counts = row["local_structure_facts"]["polyhedra_sharing"]["summary"]["sharing_type_counts"]
                    summary["relation_count_sum_across_rows"] += sum(counts.values())
                    for sharing_type, count in counts.items():
                        summary["sharing_type_counts"][sharing_type] += count
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        os.link(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(augment_file(args.input, args.output, args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
