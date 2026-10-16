"""Read-only MP archive export. Supply the MongoDB URI on stdin, never in argv."""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from bson import json_util
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "collection_001"
COLLECTION = "mp_api_all_2025_7"
PAGE = 500
META_FIELDS = [
    "material_id", "energy_above_hull", "nsites", "nelements", "elements", "chemsys",
    "formula_pretty", "deprecated", "theoretical", "symmetry", "builder_meta", "origins",
    "database_IDs", "entries.GGA.data", "entries.GGA_U.data", "entries.R2SCAN.data",
    "entries.GGA.parameters.run_type", "entries.GGA_U.parameters.run_type",
    "entries.R2SCAN.parameters.run_type",
]
RAW_FIELDS = META_FIELDS + ["structure", "delta-all_pred_GGA_ehull", "delta-all_pred_R2SCAN_ehull"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with gzip.open(path, "rt") as handle:
        for line in handle:
            yield json_util.loads(line)


def metadata_rejection(doc):
    value = doc.get("energy_above_hull")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "missing_or_nonfinite_ehull"
    if not 0 <= value <= 0.3:
        return "ehull_out_of_range"
    if doc.get("deprecated") is not False:
        return "deprecated_or_unspecified"
    if not 1 <= doc.get("nsites", 0) <= 20:
        return "nsites_out_of_range"
    if not 1 <= doc.get("nelements", 0) <= 4:
        return "nelements_out_of_range"
    if not doc.get("material_id"):
        return "missing_material_id"
    return None


def structure_key(structure):
    lattice = structure["lattice"]["matrix"]
    matrix = [[round(float(v), 8) for v in row] for row in lattice]
    sites = structure["sites"]
    typed = [(json.dumps(s["species"], sort_keys=True), s["abc"], s.get("properties", {})) for s in sites]
    first_species = min(s[0] for s in typed)
    anchors = [s[1] for s in typed if s[0] == first_species]
    images = []
    for anchor in anchors:
        image = sorted((species, tuple(round((float(v) - float(a)) % 1, 8) % 1 for v, a in zip(abc, anchor)), json.dumps(props, sort_keys=True)) for species, abc, props in typed)
        images.append(json.dumps(image, separators=(",", ":")))
    # shortcut: exact same-cell duplicates; cross-cell equivalence needs a separate grouping pass.
    payload = [matrix, min(images), structure.get("charge"), structure.get("properties", {})]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def structure_rejection(doc):
    structure = doc.get("structure")
    if not isinstance(structure, dict) or not structure.get("sites"):
        return "missing_structure"
    sites = structure["sites"]
    if len(sites) != doc["nsites"]:
        return "structure_nsites_mismatch"
    elements = []
    for site in sites:
        species = site.get("species", [])
        if len(species) != 1 or species[0].get("occu") != 1:
            return "disorder_or_partial_occupancy"
        elements.append(species[0]["element"])
        if len(site.get("abc", [])) != 3 or not all(math.isfinite(float(v)) for v in site["abc"]):
            return "invalid_fractional_coordinates"
    matrix = structure.get("lattice", {}).get("matrix", [])
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        return "missing_lattice"
    if not all(math.isfinite(float(v)) for row in matrix for v in row):
        return "nonfinite_lattice"
    a, b, c = matrix
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(det) < 1e-12:
        return "degenerate_lattice"
    if set(elements) != set(doc["elements"]) or len(set(elements)) != doc["nelements"]:
        return "structure_elements_mismatch"
    return None


def client_from_stdin():
    uri = sys.stdin.readline().strip()
    assert uri.startswith("mongodb://")
    return MongoClient(uri, appname="CrystalDLM-data-collection", serverSelectionTimeoutMS=15000,
                       connectTimeoutMS=10000, socketTimeoutMS=30000)


def inventory(client):
    OUT.mkdir(exist_ok=False)
    col = client["groupdb"][COLLECTION]
    projection = {field: 1 for field in META_FIELDS}
    total = 0
    reasons, versions = Counter(), Counter()
    after = None
    start = time.monotonic()
    with gzip.open(OUT / "metadata.jsonl.gz", "wt") as handle, (OUT / "metadata_exclusions.jsonl").open("w") as rejected:
        while True:
            query = {} if after is None else {"_id": {"$gt": after}}
            page = list(col.find(query, projection).sort("_id", 1).limit(PAGE).max_time_ms(10000))
            if not page:
                break
            for doc in page:
                handle.write(json_util.dumps(doc) + "\n")
                reason = metadata_rejection(doc)
                if reason:
                    reasons[reason] += 1
                    rejected.write(json.dumps({"material_id": doc.get("material_id"), "reason": reason}) + "\n")
                versions[str(doc.get("builder_meta", {}).get("database_version"))] += 1
            total += len(page)
            after = page[-1]["_id"]
            if total % 5000 == 0:
                print(json.dumps({"stage": "inventory", "processed": total}), flush=True)
    receipt = {"status": "metadata_export_complete", "database": "groupdb", "collection": COLLECTION,
               "records": total, "eligible_metadata": total-sum(reasons.values()), "first_exclusion_reasons": dict(reasons),
               "database_versions": dict(versions), "elapsed_seconds": time.monotonic()-start,
               "server_writes": 0, "read_method": "_id keyset pages, 500 records/page, maxTimeMS=10000"}
    write_json(OUT / "inventory_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def download(client):
    assert json.loads((OUT / "inventory_receipt.json").read_text())["status"] == "metadata_export_complete"
    candidates = [d for d in rows(OUT / "metadata.jsonl.gz") if metadata_rejection(d) is None]
    col = client["groupdb"][COLLECTION]
    projection = {field: 1 for field in RAW_FIELDS}
    reasons = Counter()
    processed = accepted = 0
    versions = Counter()
    start = time.monotonic()
    assert not (OUT / "raw_candidates.jsonl.gz").exists()
    with gzip.open(OUT / "raw_candidates.jsonl.gz", "wt") as raw, gzip.open(OUT / "valid_candidates.jsonl.gz", "wt") as valid, (OUT / "structure_exclusions.jsonl").open("w") as excluded:
        for offset in range(0, len(candidates), 200):
            batch = candidates[offset:offset+200]
            docs = {str(d["_id"]): d for d in col.find({"_id": {"$in": [d["_id"] for d in batch]}}, projection).max_time_ms(10000)}
            for meta in batch:
                doc = docs[str(meta["_id"])]
                assert doc["material_id"] == meta["material_id"]
                assert doc["energy_above_hull"] == meta["energy_above_hull"]
                assert doc.get("builder_meta", {}).get("database_version") == meta.get("builder_meta", {}).get("database_version")
                raw.write(json_util.dumps(doc) + "\n")
                reason = structure_rejection(doc)
                if reason:
                    reasons[reason] += 1
                    excluded.write(json.dumps({"material_id": doc["material_id"], "reason": reason}) + "\n")
                else:
                    doc["structure_key"] = structure_key(doc["structure"])
                    valid.write(json_util.dumps(doc) + "\n")
                    accepted += 1
                versions[str(doc.get("builder_meta", {}).get("database_version"))] += 1
                processed += 1
            if offset % 2000 == 0:
                print(json.dumps({"stage": "download", "processed": processed, "accepted": accepted}), flush=True)
    receipt = {"status": "candidate_structures_export_complete", "metadata_candidates": len(candidates),
               "processed": processed, "valid_candidates": accepted, "structure_exclusions": dict(reasons),
               "database_versions": dict(versions), "elapsed_seconds": time.monotonic()-start, "server_writes": 0}
    write_json(OUT / "download_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    client = client_from_stdin()
    try:
        {"inventory": inventory, "download": download}[sys.argv[1]](client)
    finally:
        client.close()
