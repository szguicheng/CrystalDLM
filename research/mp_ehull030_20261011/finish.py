"""Select the exported archive records and verify the published data contract."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

from bson import json_util
from collect import OUT, ROOT, metadata_rejection, rows, structure_rejection, write_json
from sampling import select_records, split_records


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    receipt = json.loads((OUT / "download_receipt.json").read_text())
    assert receipt["status"] == "candidate_structures_export_complete"
    light = []
    for doc in rows(OUT / "valid_candidates.jsonl.gz"):
        light.append({"material_id": doc["material_id"], "energy_above_hull": doc["energy_above_hull"],
                      "chemsys": "-".join(sorted(doc["elements"])), "structure_key": doc["structure_key"]})
    assert len(light) == receipt["valid_candidates"]
    assert len({d["material_id"] for d in light}) == len(light)
    chosen, selection = select_records(light, per_bin=10000, seed=17)
    chosen, split = split_records(chosen, seed=17)
    selected = {row["material_id"]: row for row in chosen}
    assert len(selected) == len(chosen)
    assert len({row["structure_key"] for row in chosen}) == len(chosen)
    assert all(r["selected_count"] <= 10000 for r in selection["bins"].values())

    output = OUT / "data_001"
    output.mkdir(exist_ok=False)
    handles = {name: gzip.open(output / f"{name}.jsonl.gz", "wt") for name in ["records", "train", "val", "test"]}
    emitted = Counter()
    snapshots, source_matches, theoretical = Counter(), Counter(), Counter()
    emitted_ids = set()
    try:
        for doc in rows(OUT / "valid_candidates.jsonl.gz"):
            mid = doc["material_id"]
            if mid not in selected:
                continue
            row = selected[mid]
            assert metadata_rejection(doc) is None and structure_rejection(doc) is None
            assert doc["energy_above_hull"] == row["energy_above_hull"]
            assert doc["structure_key"] == row["structure_key"]
            origin = doc.get("origins")
            task = origin.get("task_id") if isinstance(origin, dict) else None
            entry_matches = [kind for kind, entry in doc.get("entries", {}).items()
                             if isinstance(entry, dict) and task is not None and entry.get("data", {}).get("task_id") == task]
            record = {
                "material_id": mid, "structure": doc["structure"], "energy_above_hull": doc["energy_above_hull"],
                "energy_above_hull_unit": "eV/atom", "split": row["split"], "structure_key": row["structure_key"],
                "formula": doc.get("formula_pretty"), "elements": doc["elements"], "chemsys": row["chemsys"],
                "nsites": doc["nsites"], "nelements": doc["nelements"], "symmetry": doc.get("symmetry"),
                "theoretical": doc.get("theoretical"), "database_IDs": doc.get("database_IDs"),
                "provenance": {"source": "user MP MongoDB archive", "database": "groupdb",
                    "collection": "mp_api_all_2025_7", "source_document_id": str(doc["_id"]),
                    "builder_meta": doc.get("builder_meta"), "structure_origin": origin,
                    "matching_structure_entry_types": entry_matches, "entry_metadata": doc.get("entries"),
                    "label_source_field": "energy_above_hull", "ehull_thermo_type": None,
                    "thermo_type_note": "not explicitly stored in archive; preserved source label without inferred correction or substitution"},
            }
            text = json_util.dumps(record) + "\n"
            handles["records"].write(text)
            handles[row["split"]].write(text)
            emitted[row["split"]] += 1
            emitted_ids.add(mid)
            snapshots[str(doc.get("builder_meta", {}).get("database_version"))] += 1
            source_matches["|".join(sorted(entry_matches)) or "no_task_match"] += 1
            theoretical[str(doc.get("theoretical"))] += 1
    finally:
        for handle in handles.values():
            handle.close()
    assert emitted_ids == set(selected)
    assert dict(emitted) == split["row_counts"]
    assert len(snapshots) == 1
    with (output / "selection.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["material_id", "energy_above_hull", "chemsys", "structure_key", "split"])
        writer.writeheader()
        writer.writerows(chosen)
    write_json(output / "sampling_report.json", selection)
    write_json(output / "split_report.json", split)
    artifacts = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in output.iterdir() if p.is_file()}
    checks = json.loads((ROOT / "mp_official_three_check_001.json").read_text())
    manifest = {"status": "selected_archive_structure_label_data", "records": len(chosen), "split_counts": dict(emitted),
                "database_versions": dict(snapshots), "theoretical_counts": dict(theoretical),
                "filters": {"ehull_eV_per_atom": [0, 0.3], "nsites": [1, 20], "nelements": [1, 4],
                            "deprecated": False, "ordered_full_occupancy": True},
                "label": "original continuous energy_above_hull; no recalculation, no prediction-field substitution",
                "seed": 17, "per_bin_max": 10000, "bins": {k: {f:v for f,v in r.items() if f != "chemical_systems"} for k,r in selection["bins"].items()},
                "duplicate_protocol": "same lattice and sites to 8 decimal places, origin/site-order invariant; site properties retained; cross-cell symmetry equivalence not grouped",
                "structure_task_entry_matches": dict(source_matches),
                "official_spotcheck": {"path": str(ROOT / "mp_official_three_check_001.json"), "sha256": digest(ROOT / "mp_official_three_check_001.json"), "status": checks["status"], "n": 3},
                "source_hashes": {name: digest(ROOT / name) for name in ["collect.py", "sampling.py", "finish.py"]},
                "upstream_artifacts": {name: digest(OUT / name) for name in ["metadata.jsonl.gz", "raw_candidates.jsonl.gz", "valid_candidates.jsonl.gz", "inventory_receipt.json", "download_receipt.json"]},
                "artifacts": artifacts, "server_writes": 0, "training_run": False,
                "limitations": ["archived thermo_type is not explicit; three official-label spotchecks do not establish all-row calculation provenance", "no PLaID encoding or generated-crystal evaluation performed"]}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "records": len(chosen), "bins": manifest["bins"], "split_counts": dict(emitted)}), flush=True)


if __name__ == "__main__":
    main()
