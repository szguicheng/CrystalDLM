"""Verify unchanged parent facts and write split queues with sharing metadata."""
from collections import Counter
from contextlib import ExitStack
import gzip
import hashlib
from itertools import zip_longest
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    stage = Path(__file__).resolve().parent
    parent = stage.parent / "crystal_preannotation_20261012"
    baseline = json.loads((parent / "prepared_002/manifest.json").read_text())
    output = stage / "prepared_001"
    output.mkdir(exist_ok=False)
    statuses, splits, base_status, contacts, row_presence = (Counter() for _ in range(5))
    seen = set()
    sharing_methods = set()
    files = sorted((stage / "full_001").glob("shard-*.jsonl"))
    assert len(files) == 32
    with ExitStack() as stack:
        queues = {name: stack.enter_context(gzip.open(output / f"{name}-facts.jsonl.gz", "wt"))
                  for name in ["train", "val", "test"]}
        errors = stack.enter_context(gzip.open(output / "base-errors.jsonl.gz", "wt"))
        for path in files:
            old_path = parent / "full_002" / path.name
            assert sha(old_path) == baseline["shard_hashes"][path.name]
            with old_path.open() as old_stream, path.open() as new_stream:
                for old_line, new_line in zip_longest(old_stream, new_stream):
                    assert old_line is not None and new_line is not None
                    old, new = json.loads(old_line), json.loads(new_line)
                    identity = new["material_id"]
                    assert identity not in seen
                    seen.add(identity)
                    base_status[new["status"]] += 1
                    if new["status"] != "ok":
                        assert new == old
                        errors.write(new_line)
                        continue
                    sharing = new["local_structure_facts"]["polyhedra_sharing"]
                    check = dict(new)
                    check["local_structure_facts"] = dict(new["local_structure_facts"])
                    check["local_structure_facts"]["polyhedra_sharing"] = old["local_structure_facts"]["polyhedra_sharing"]
                    assert check == old, "A source fact changed outside the sharing field"
                    assert sharing["source_feature_method_sha256"] == baseline["feature_method"][1]
                    sharing_methods.add((sharing["method_id"], sharing["method_sha256"]))
                    statuses[sharing["status"]] += 1
                    for kind, count in sharing["summary"]["sharing_type_counts"].items():
                        contacts[kind] += count
                        row_presence[kind] += count > 0
                    splits[new["split"]] += 1
                    queues[new["split"]].write(new_line)
    assert len(seen) == baseline["source_records"] == 45629
    assert dict(base_status) == baseline["record_status_counts"]
    assert dict(splits) == baseline["annotation_eligible_split_counts"]
    assert sum(statuses.values()) == 45465 and len(sharing_methods) == 1
    report = {"source_records": len(seen), "base_status_counts": dict(base_status),
              "eligible_split_counts": dict(splits), "sharing_status_counts": dict(statuses),
              "periodic_contact_counts": dict(contacts), "materials_with_sharing_type": dict(row_presence),
              "all_parent_facts_and_labels_unchanged": True, "sharing_method": list(sharing_methods)[0],
              "artifacts": {file.name: {"sha256": sha(file), "bytes": file.stat().st_size}
                            for file in sorted(output.glob("*.gz"))}, "bulk_llm_calls": 0}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
