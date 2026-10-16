"""Bind the frozen PLaID/ehull/description triples to source facts and receipts."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/crystal_preannotation_20261012"))
import annotation as helpers

STAGE = ROOT / "research/crystal_bulk_annotation_20261013"
OUTPUT = Path(__file__).resolve().parent / "data_001"
TOKENIZER_PATH = ROOT / "experiments/interleaved_ar_diffusion_20261002/data_002/tokenizer/tokenizer.json"
END = "<CRYSTAL_END>"
EOS = "<|im_end|>"
SPLITS = {"train": 7272, "val": 900}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def caption_flags(raw, clean):
    flags = []
    for name, value in (("raw", raw), ("clean", clean)):
        if not value.strip():
            flags.append(f"empty_{name}_caption")
        if value != value.strip():
            flags.append(f"surrounding_whitespace_{name}_caption")
        if "\n" in value or "\r" in value:
            flags.append(f"line_break_{name}_caption")
        if any(unicodedata.name(char, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))
               for char in value):
            flags.append(f"contains_cjk_script_{name}_caption")
        terminal = value.rstrip()
        while terminal and terminal[-1] in ")]}）］｝」』”’\"'":
            terminal = terminal[:-1].rstrip()
        if terminal and not terminal.endswith((".", "!", "?", "。", "！", "？")):
            flags.append(f"missing_terminal_punctuation_{name}_caption")
    return flags


def build_triple(row, receipt, split):
    material_id = row["material_id"]
    if receipt["material_id"] != material_id:
        raise ValueError(f"material_id mismatch: {material_id}")
    if row["split"] != split or receipt["split"] != split:
        raise ValueError(f"split mismatch: {material_id}")
    if row["status"] != "ok" or receipt["status"] != "ok":
        raise ValueError(f"non-ok source or receipt: {material_id}")
    facts_sha256 = helpers.sha(row)
    if receipt["facts_sha256"] != facts_sha256:
        raise ValueError(f"facts_sha256 mismatch: {material_id}")

    labels = row["property_labels"]
    if receipt["property_labels"] != labels:
        raise ValueError(f"property_labels changed: {material_id}")
    if labels["unit"] != "eV/atom" or labels["source_field"] != "energy_above_hull":
        raise ValueError(f"unexpected ehull label schema: {material_id}")

    consistency = row["plaid"]["consistency"]
    checks = [value for value in consistency.values() if isinstance(value, bool)]
    if consistency.get("all_scientific_consistency_checks_pass") is not True or not checks or not all(checks):
        raise ValueError(f"PLaID consistency check failed: {material_id}")

    raw = receipt["structure_description"]
    rendered = receipt["description"]
    if rendered != helpers.render_description(raw, row):
        raise ValueError(f"rendered description mismatch: {material_id}")
    suffix = (" The original archived MP structure has a supplied energy above hull label of "
              + str(labels["energy_above_hull"]) + " " + labels["unit"] + ".")
    if not rendered.endswith(suffix):
        raise ValueError(f"expected exact label suffix is missing: {material_id}")
    clean = rendered[:-len(suffix)]
    identifier = re.escape(material_id)
    clean = re.sub(r"\s*\((?:material\s+)?" + identifier + r"\)", "", clean)
    clean = re.sub(r",\s*material\s+" + identifier + r",\s*", " ", clean)
    clean = re.sub(r"\s+(?:of|for)\s+(?:material\s+)?" + identifier + r"\b", "", clean)
    clean = re.sub(r"\b(?:The|the)\s+(?:material\s+)?" + identifier + r"\b", "The crystal", clean)
    clean = re.sub(r"\b(?:Material|material)\s+" + identifier + r"\b", "The crystal", clean)
    clean = re.sub(r"\b" + identifier + r"\b", "this crystal", clean)
    assert material_id not in clean

    source = row["source"]
    return {
        "material_id": material_id,
        "split": split,
        "plaid_text": row["plaid"]["encoded_text"],
        "structure_description": clean,
        "structure_description_raw": raw,
        "original_structure_description_raw": receipt.get("original_structure_description_raw", raw),
        "caption_cleanups": ["remove_material_id"] if clean != rendered[:-len(suffix)] else [],
        "caption_reannotated": "original_structure_description_raw" in receipt,
        "property_labels": labels,
        "property_condition_text": "energy above hull = " + str(labels["energy_above_hull"]) + " " + labels["unit"],
        "core_facts": row["core_facts"],
        "plaid_consistency": True,
        "plaid_encoder_fallback_used": row["plaid"]["encoder_fallback_used"],
        "facts_sha256": facts_sha256,
        "request_sha256": receipt["request_sha256"],
        "source_fingerprints": {
            "input_sha256": source["input_sha256"],
            "structure_sha256": source["structure_sha256"],
        },
        "caption_flags": caption_flags(raw, clean),
    }


def nearest_rank(values, percentile):
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def token_summary(lengths):
    return {
        "count": len(lengths),
        "p50": nearest_rank(lengths, 0.50),
        "p95": nearest_rank(lengths, 0.95),
        "max": max(lengths),
        "exceeds_K256": sum(length > 256 for length in lengths),
        "exceeds_context_1536": sum(length > 1536 for length in lengths),
    }


def main():
    subset = STAGE / "subset_001"
    run = STAGE / "run_001"
    subset_manifest_path = subset / "manifest.json"
    request_manifest_path = run / "request-manifest.json"
    subset_manifest = json.loads(subset_manifest_path.read_text())
    request_manifest = json.loads(request_manifest_path.read_text())

    frozen_ids = {}
    for split, expected_count in SPLITS.items():
        spec = subset_manifest["splits"][split]
        facts_path = subset / spec["facts_path"]
        ids_path = subset / spec["ids_path"]
        assert spec["selected_count"] == expected_count
        assert file_sha256(facts_path) == spec["facts_sha256"]
        assert file_sha256(ids_path) == spec["ids_sha256"]
        assert spec["facts_sha256"] == next(x["sha256"] for x in request_manifest["inputs"]
                                               if Path(x["path"]).name == facts_path.name)
        ids = ids_path.read_text().splitlines()
        assert len(ids) == expected_count and len(set(ids)) == expected_count
        frozen_ids[split] = set(ids)
    assert not (frozen_ids["train"] & frozen_ids["val"])
    assert file_sha256(ROOT / "research/crystal_preannotation_20261012/annotation.py") == request_manifest["helper_sha256"]

    receipts = {}
    with (run / "successes.jsonl").open() as stream:
        for line in stream:
            receipt = json.loads(line)
            material_id = receipt["material_id"]
            assert material_id not in receipts
            receipts[material_id] = receipt
    expected_ids = frozen_ids["train"] | frozen_ids["val"]
    assert set(receipts) == expected_ids
    assert all(receipt["status"] == "ok" for receipt in receipts.values())
    assert all((material_id in frozen_ids[receipt["split"]]) for material_id, receipt in receipts.items())
    repair_path = Path(__file__).resolve().parent / "caption-repairs.jsonl"
    repaired_ids = set()
    if repair_path.exists():
        for line in repair_path.read_text().splitlines():
            repaired = json.loads(line)
            material_id = repaired["material_id"]
            previous = receipts[material_id]
            assert material_id not in repaired_ids and repaired["status"] == "ok"
            assert repaired["facts_sha256"] == previous["facts_sha256"]
            assert repaired["property_labels"] == previous["property_labels"]
            repaired["original_structure_description_raw"] = previous["structure_description"]
            receipts[material_id] = repaired
            repaired_ids.add(material_id)
        assert repaired_ids == {"mp-1113520", "mp-1184824"}

    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        from tokenizers import Tokenizer, __version__ as tokenizers_version
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
        assert len(tokenizer.encode(END).ids) == 1
        assert len(tokenizer.encode(EOS).ids) == 1
        token_lengths = defaultdict(list)
        token_audit_status = "ready"
    except ImportError:
        tokenizer = None
        tokenizers_version = None
        token_lengths = defaultdict(list)
        token_audit_status = "tokenizers package unavailable"

    counts = Counter()
    caption_flag_counts = Counter()
    flagged_caption_samples = []
    encoder_fallback_count = 0
    cleanup_count = 0
    seen = set()
    artifacts = {}
    for split, expected_count in SPLITS.items():
        facts_path = subset / subset_manifest["splits"][split]["facts_path"]
        target = OUTPUT / f"{split}-triples.jsonl.gz"
        temporary = OUTPUT / f".{split}-triples.jsonl.gz.tmp"
        split_seen = set()
        with gzip.open(facts_path, "rt", encoding="utf-8") as source, \
                gzip.open(temporary, "wt", encoding="utf-8") as output:
            for line in source:
                row = json.loads(line)
                material_id = row["material_id"]
                assert material_id not in split_seen and material_id not in seen
                assert material_id in frozen_ids[split]
                split_seen.add(material_id)
                seen.add(material_id)
                triple = build_triple(row, receipts[material_id], split)
                cleanup_count += bool(triple["caption_cleanups"])
                output.write(json.dumps(triple, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[split] += 1
                caption_flag_counts.update(triple["caption_flags"])
                if triple["caption_flags"]:
                    flagged_caption_samples.append({"material_id": material_id, "split": split,
                                                    "flags": triple["caption_flags"]})
                encoder_fallback_count += triple["plaid_encoder_fallback_used"] is True
                if tokenizer is not None:
                    body = triple["plaid_text"]
                    assert END not in body
                    token_lengths["plaid_body_plus_end"].append(len(tokenizer.encode(body + END).ids))
                    token_lengths["raw_caption"].append(len(tokenizer.encode(triple["structure_description_raw"]).ids))
                    clean = triple["structure_description"]
                    token_lengths["clean_caption"].append(len(tokenizer.encode(clean).ids))
                    token_lengths["clean_caption_plus_eos"].append(len(tokenizer.encode(clean + EOS).ids))
        assert split_seen == frozen_ids[split] and counts[split] == expected_count
        temporary.replace(target)
        artifacts[target.name] = {"sha256": file_sha256(target), "bytes": target.stat().st_size}

    assert seen == expected_ids and sum(counts.values()) == 8172
    token_audit = {
        "status": token_audit_status,
        "tokenizer_path": str(TOKENIZER_PATH.relative_to(ROOT)),
        "tokenizer_sha256": file_sha256(TOKENIZER_PATH) if TOKENIZER_PATH.exists() else None,
        "tokenizers_version": tokenizers_version,
        "method": "local Tokenizer.from_file; no Hugging Face calls; nearest-rank percentiles",
        "field_lengths_only": True,
        "complete_training_context_not_counted": True,
        "distributions": {name: token_summary(lengths) for name, lengths in token_lengths.items()},
    }
    token_audit_path = OUTPUT / "token-length-audit.json"
    token_audit_path.write_text(json.dumps(token_audit, indent=2) + "\n")
    artifacts[token_audit_path.name] = {"sha256": file_sha256(token_audit_path), "bytes": token_audit_path.stat().st_size}

    manifest = {
        "schema_version": 1,
        "counts": dict(counts),
        "total": sum(counts.values()),
        "checks": {
            "frozen_id_sets_match_facts": True,
            "split_ids_unique_and_disjoint": True,
            "all_fact_hashes_match_receipts": True,
            "property_labels_unchanged": True,
            "label_unit": "eV/atom",
            "plaid_consistency_all_true": True,
            "all_receipts_ok": True,
            "excluded_samples": 0,
            "global_caption_scientific_qa": "not performed",
        },
        "caption_flag_counts": dict(caption_flag_counts),
        "caption_flagged_samples": flagged_caption_samples,
        "caption_flags_disposition": "reported only; all samples retained",
        "caption_reannotated_ids": sorted(repaired_ids),
        "caption_material_id_cleanup_count": cleanup_count,
        "plaid_encoder_fallback_used": encoder_fallback_count,
        "sources": {
            "subset_manifest_sha256": file_sha256(subset_manifest_path),
            "request_manifest_sha256": file_sha256(request_manifest_path),
            "successes_sha256": file_sha256(run / "successes.jsonl"),
            "annotation_helper_sha256": file_sha256(ROOT / "research/crystal_preannotation_20261012/annotation.py"),
            "caption_repairs_sha256": file_sha256(repair_path) if repair_path.exists() else None,
        },
        "artifacts": artifacts,
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
