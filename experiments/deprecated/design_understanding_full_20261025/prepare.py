"""Export separate full-data Design and PLaID-only Understanding SFT tasks."""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STAGE = Path(__file__).resolve().parent
SOURCE = ROOT / "research/crystal_sharing_20261012/prepared_001"
OLD = ROOT / "research/crystal_training_preflight_20261013/data_001"
DATA = STAGE / "data"
EXPECTED = {"train": 36362, "val": 4501, "test": 4602}
NAMES = {"train": "training", "val": "validation", "test": "test"}
START, END = "<CRYSTAL_START>", "<CRYSTAL_END>"
QUESTION = ("Read the current crystal and report its formula, space group number, "
            "cell atom count, and independent orbit count.")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def dump(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def old_captions():
    result = {}
    manifest = json.loads((OLD / "manifest.json").read_text())
    for split in ("train", "val"):
        path = OLD / f"{split}-triples.jsonl.gz"
        assert sha(path) == manifest["artifacts"][path.name]["sha256"], path.name
        for row in rows(path):
            assert row["material_id"] not in result
            result[row["material_id"]] = row
    assert len(result) == 8172
    return result


def answer(facts):
    return (f"The formula is {facts['cell_formula']}. "
            f"The space group number is {facts['space_group_number']}. "
            f"The unit cell contains {facts['cell_atom_count']} atoms and "
            f"{facts['independent_orbit_count']} independent orbits.")


def crystal(row):
    body = row["plaid"]["encoded_text"]
    assert START not in body and END not in body
    return START + body + END


def understanding(row):
    return {"instruction": QUESTION, "input": crystal(row),
            "output": answer(row["core_facts"])}


def design(row, caption):
    labels = row["property_labels"]
    assert labels["unit"] == "eV/atom" and labels["source_field"] == "energy_above_hull"
    energy = Decimal(str(labels["energy_above_hull"]))
    assert energy.is_finite() and 0 <= energy <= Decimal("0.3")
    bound = next(b for b in ("0.08", "0.1", "0.2", "0.3") if energy <= Decimal(b))
    group = row["core_facts"]["space_group_number"]
    return {"instruction": f"Generate a crystal with space group number {group} and "
            f"energy above hull no greater than {bound} eV/atom.", "input": "",
            "output": caption.strip() + "\n" + crystal(row)}


def check_sources():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    assert manifest["eligible_split_counts"] == EXPECTED
    for name, item in manifest["artifacts"].items():
        assert sha(SOURCE / name) == item["sha256"], name
    return manifest


def registry(tasks):
    return {f"crystal_{task}_{name}": {
        "file_name": f"{task}/{name}.jsonl", "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"}}
        for task in tasks for name in NAMES.values()}


def prepare():
    assert not (STAGE / "annotation_run/annotation.sqlite3").exists(), "Do not regenerate frozen annotation inputs after annotation starts"
    source_manifest = check_sources()
    old = old_captions()
    for directory in (DATA / "understanding", DATA / "metadata", STAGE / "annotation_inputs", STAGE / "examples"):
        directory.mkdir(parents=True, exist_ok=True)
    counts, missing = Counter(), Counter()
    ids, structures = set(), set()
    reused = set()
    files = {}
    for split, expected in EXPECTED.items():
        name = NAMES[split]
        output = DATA / "understanding" / f"{name}.jsonl"
        metadata = DATA / "metadata" / f"{name}.jsonl"
        pending = STAGE / "annotation_inputs" / f"{split}-facts.jsonl.gz"
        with output.open("w") as out, metadata.open("w") as meta, gzip.open(pending, "wt") as todo:
            for index, row in enumerate(rows(SOURCE / f"{split}-facts.jsonl.gz")):
                mid = row["material_id"]
                source_hash = row["source"]["structure_sha256"]
                assert row["split"] == split and row["status"] == "ok"
                assert row["plaid"]["consistency"]["all_scientific_consistency_checks_pass"] is True
                assert mid not in ids and source_hash not in structures
                ids.add(mid)
                structures.add(source_hash)
                dump(out, understanding(row))
                dump(meta, {"row_index": index, "material_id": mid, "source_split": split,
                    "source_structure_sha256": source_hash,
                    "caption_source": "reused_subset" if mid in old else "full_annotation",
                    "core_facts": row["core_facts"], "property_labels": row["property_labels"]})
                counts[split] += 1
                if mid in old:
                    previous = old[mid]
                    assert previous["split"] == split
                    assert previous["plaid_text"] == row["plaid"]["encoded_text"]
                    assert previous["core_facts"] == row["core_facts"]
                    assert previous["property_labels"] == row["property_labels"]
                    assert previous["facts_sha256"] == hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    reused.add(mid)
                else:
                    dump(todo, row)
                    missing[split] += 1
                if mid == "mp-1210092":
                    write_json(STAGE / "examples" / "understanding-mp-1210092.json", understanding(row))
                    write_json(STAGE / "examples" / "design-mp-1210092.json", design(row, old[mid]["structure_description"]))
            assert counts[split] == expected
        for path in (output, metadata, pending):
            files[str(path.relative_to(STAGE))] = {"sha256": sha(path), "bytes": path.stat().st_size}
    assert reused == set(old)
    assert sum(counts.values()) == 45465 and sum(missing.values()) == 37293
    write_json(DATA / "dataset_info.json", registry(["understanding"]))
    manifest = {"status": "understanding_ready_design_awaiting_captions", "counts_per_task": dict(counts),
        "sampling_fraction": 1.0, "split_policy": "unchanged source structure-hash split; no new random split",
        "original_records": 45629, "prior_preprocessing_errors": 164,
        "source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "source_artifacts": source_manifest["artifacts"], "reused_captions": len(reused),
        "reused_caption_sources": {f"{split}-triples.jsonl.gz": sha(OLD / f"{split}-triples.jsonl.gz")
                                   for split in ("train", "val")},
        "missing_captions": dict(missing), "artifacts": files,
        "understanding_input": "PLaID with crystal boundary markers only; no generation request, caption, energy, or history",
        "output_eos": "not stored; the selected LLaMA-Factory chat template appends its EOS",
        "builder_sha256": sha(__file__)}
    write_json(DATA / "manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k != "artifacts"}, indent=2))


def finalize():
    check_sources()
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, expected_hash in manifest["reused_caption_sources"].items():
        assert sha(OLD / name) == expected_hash, name
    sys.path.insert(0, str(ROOT / "research/crystal_training_preflight_20261013"))
    from assemble import build_triple
    old = old_captions()
    receipts = {r["material_id"]: r for r in rows(STAGE / "annotation_run" / "successes.jsonl")}
    assert len(receipts) == 37293 and not set(receipts) & set(old)
    (DATA / "design").mkdir(parents=True, exist_ok=True)
    flags = Counter()
    seen = set()
    for split, expected in EXPECTED.items():
        destination = DATA / "design" / f"{NAMES[split]}.jsonl"
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w") as stream:
            count = 0
            for row in rows(SOURCE / f"{split}-facts.jsonl.gz"):
                mid = row["material_id"]
                triple = old[mid] if mid in old else build_triple(row, receipts[mid], split)
                flags.update(triple["caption_flags"])
                dump(stream, design(row, triple["structure_description"]))
                count += 1
                seen.add(mid)
            assert count == expected
        temporary.replace(destination)
    assert seen == set(old) | set(receipts)
    manifest.update(status="both_tasks_exported_validation_pending", new_captions=len(receipts), caption_flags=dict(flags),
                    finalizer_sha256=sha(__file__),
                    caption_scientific_qa="automated consistency and formatting checks; not exhaustive semantic review")
    for path in (DATA / "design").glob("*.jsonl"):
        manifest["artifacts"][str(path.relative_to(STAGE))] = {"sha256": sha(path), "bytes": path.stat().st_size}
    write_json(DATA / "manifest.json", manifest)
    write_json(DATA / "dataset_info.json", registry(["design", "understanding"]))
    print(json.dumps({"status": manifest["status"], "counts_per_task": manifest["counts_per_task"], "caption_flags": dict(flags)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "finalize"))
    args = parser.parse_args()
    {"prepare": prepare, "finalize": finalize}[args.stage]()
