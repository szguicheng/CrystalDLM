"""Verify source bindings and build split-specific annotation queues from shards."""
import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def structure_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source_path = Path(args.source)
    source_sha = file_hash(source_path)
    expected = {}
    with gzip.open(source_path, "rt") as source:
        for index, line in enumerate(source):
            row = json.loads(line)
            assert row["material_id"] not in expected
            expected[row["material_id"]] = (index, row["split"], row["energy_above_hull"],
                row["energy_above_hull_unit"], structure_hash(row["structure"]))
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    seen = set()
    counts, split_ok, split_error, stages, messages, failed_checks, decoded_atoms = (Counter() for _ in range(7))
    examples = {}
    hashes = {}
    methods = set()
    with ExitStack() as stack:
        queues = {split: stack.enter_context(gzip.open(output / f"{split}-facts.jsonl.gz", "wt"))
                  for split in ["train", "val", "test"]}
        failures = stack.enter_context(gzip.open(output / "errors.jsonl.gz", "wt"))
        for shard in sorted(Path(args.shards).glob("shard-*.jsonl")):
            hashes[shard.name] = file_hash(shard)
            with shard.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    identity = row["material_id"]
                    assert identity not in seen
                    seen.add(identity)
                    index, split, value, unit, original_hash = expected[identity]
                    assert row["source"]["row_index"] == index
                    assert row["source"]["input_sha256"] == source_sha
                    assert row["source"]["structure_sha256"] == original_hash
                    assert row["split"] == split
                    assert row["property_labels"]["energy_above_hull"] == value
                    assert row["property_labels"]["unit"] == unit
                    methods.add((row["feature_method_id"], row["feature_method_sha256"]))
                    counts[row["status"]] += 1
                    if row["status"] == "ok":
                        assert row["plaid"]["consistency"]["all_scientific_consistency_checks_pass"]
                        decoded_atoms[row["core_facts"]["cell_atom_count"]] += 1
                        split_ok[split] += 1
                        queues[split].write(line)
                    else:
                        assert row["status"] == "error"
                        split_error[split] += 1
                        error = row["error"]
                        category = error["stage"] + ":" + error["type"]
                        stages[category] += 1
                        messages[error["message"]] += 1
                        examples.setdefault(category, [])
                        if len(examples[category]) < 5:
                            examples[category].append(identity)
                        consistency = (row.get("plaid") or {}).get("consistency") or {}
                        failed_checks.update(k for k, v in consistency.items() if isinstance(v, bool) and not v)
                        failures.write(line)
    assert seen == set(expected) and len(methods) == 1
    manifest = {"source_records": len(expected), "source_sha256": source_sha,
        "record_status_counts": dict(counts), "annotation_eligible_split_counts": dict(split_ok),
        "error_split_counts": dict(split_error), "error_stage_counts": dict(stages),
        "error_messages": dict(messages), "failed_consistency_checks": dict(failed_checks),
        "error_examples": examples, "decoded_atom_counts": dict(sorted(decoded_atoms.items())),
        "feature_method": list(methods)[0], "shard_hashes": hashes,
        "source_identity_structure_hash_split_and_label_checks": "all records passed",
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                      for path in sorted(output.glob("*.gz"))},
        "bulk_llm_calls": 0}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items()
                     if k not in ["shard_hashes", "artifacts", "error_messages", "decoded_atom_counts"]}, indent=2))


if __name__ == "__main__":
    main()
