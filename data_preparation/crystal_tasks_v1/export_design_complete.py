"""Publish the completed Design task beside an untouched crystal_tasks_v1 parent."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import statistics


RELEASE_REL = Path("data_preparation/crystal_tasks_v1/exports/design_full_20261026")
TRANSFER_PREFIX = ".transfer_design_full_20261026_"
EXPECTED = {"train": 36362, "val": 4501, "test": 4602}
NAMES = {"train": "training", "val": "validation", "test": "test"}
CONTROL_IDS = [151669, 151670, 151671, 151672]
CUTOFF = 2048


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def summarize_lengths(lengths, cutoff):
    values = sorted(lengths)
    if not values:
        raise ValueError("at least one length is required")

    def nearest_rank(percentile):
        return values[math.ceil(percentile * len(values)) - 1]

    return {
        "count": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p95": nearest_rank(0.95),
        "p99": nearest_rank(0.99),
        "max": values[-1],
        "above_cutoff": sum(value > cutoff for value in values),
    }


def copy_unchanged(source, destination):
    source, destination = Path(source), Path(destination)
    source_hash = sha256(source)
    if destination.exists():
        if sha256(destination) != source_hash:
            raise ValueError(f"resume found changed copied input: {destination}")
        return source_hash
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.copytmp")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    if sha256(temporary) != source_hash:
        raise ValueError(f"copy hash mismatch: {source}")
    temporary.replace(destination)
    return source_hash


def verify_receipts(prepare, transfer_dir, expected_sha, expected_rows):
    archive = transfer_dir / "successes.jsonl.gz"
    receipts_path = transfer_dir / "annotation_run/successes.jsonl"
    if not receipts_path.exists():
        if not archive.is_file():
            raise FileNotFoundError(archive)
        receipts_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipts_path.with_suffix(".jsonl.tmp")
        with gzip.open(archive, "rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        if sha256(temporary) != expected_sha:
            raise ValueError("transferred annotation journal SHA256 mismatch")
        temporary.replace(receipts_path)
    if sha256(receipts_path) != expected_sha:
        raise ValueError("annotation journal SHA256 mismatch")

    receipts = {}
    counts = Counter()
    for receipt in prepare.rows(receipts_path):
        material_id = receipt["material_id"]
        if material_id in receipts:
            raise ValueError(f"duplicate annotation receipt: {material_id}")
        if receipt["status"] != "ok" or receipt["split"] not in EXPECTED:
            raise ValueError(f"invalid annotation receipt: {material_id}")
        receipts[material_id] = receipt
        counts[receipt["split"]] += 1
    if len(receipts) != expected_rows:
        raise ValueError(f"expected {expected_rows} new receipts, found {len(receipts)}")
    return receipts, dict(counts), receipts_path


def verify_source_join(prepare, old, receipts):
    old_ids = set(old)
    receipt_ids = set(receipts)
    if old_ids & receipt_ids:
        raise ValueError("new annotation receipts overlap reused captions")

    source_ids, old_seen, new_by_split = set(), set(), Counter()
    for split in EXPECTED:
        for row in prepare.rows(prepare.SOURCE / f"{split}-facts.jsonl.gz"):
            material_id = row["material_id"]
            if material_id in source_ids:
                raise ValueError(f"duplicate source material ID: {material_id}")
            source_ids.add(material_id)
            if material_id in old:
                caption = old[material_id]
                facts_hash = hashlib.sha256(json.dumps(
                    row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if (caption["split"] != split
                        or caption["plaid_text"] != row["plaid"]["encoded_text"]
                        or caption["core_facts"] != row["core_facts"]
                        or caption["property_labels"] != row["property_labels"]
                        or caption["facts_sha256"] != facts_hash):
                    raise ValueError(f"reused caption/source mismatch: {material_id}")
                old_seen.add(material_id)
            else:
                new_by_split[split] += 1
                receipt = receipts.get(material_id)
                if receipt is None or receipt["split"] != split:
                    raise ValueError(f"missing or wrong-split annotation receipt: {material_id}")

    if old_seen != old_ids or receipt_ids != source_ids - old_ids:
        raise ValueError("source, reused caption, and complete receipt ID sets differ")
    if len(source_ids) != sum(EXPECTED.values()) or len(old_ids) != 8172 or len(receipt_ids) != 37293:
        raise ValueError("full eligible source count differs from the frozen contract")
    return {"source_ids": len(source_ids), "reused_ids": len(old_ids),
            "new_receipt_ids": len(receipt_ids), "new_by_split": dict(new_by_split)}


def collect_finalize_failures(prepare, old, receipts, build_triple):
    failures = []
    for split in EXPECTED:
        for row in prepare.rows(prepare.SOURCE / f"{split}-facts.jsonl.gz"):
            material_id = row["material_id"]
            try:
                caption = (old[material_id]["structure_description"] if material_id in old
                           else build_triple(row, receipts[material_id], split)["structure_description"])
                prepare.design(row, caption)
            except Exception as error:
                failures.append({"material_id": material_id, "split": split,
                                 "reason": f"{type(error).__name__}: {error}"})
    return failures


def copy_parent_files(root, release, parent_manifest_sha):
    parent = root / "datasets/crystal_tasks_v1"
    manifest_path = release / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("parent_manifest_sha256") != parent_manifest_sha
                and manifest.get("source_manifest_sha256") != json.loads(
                    (parent / "manifest.json").read_text()).get("source_manifest_sha256")):
            raise ValueError("resume manifest is not derived from the frozen parent")
    else:
        copy_unchanged(parent / "manifest.json", manifest_path)

    registry_path = release / "dataset_info.json"
    if not registry_path.exists():
        copy_unchanged(parent / "dataset_info.json", registry_path)

    preserved = []
    for group in ("understanding", "metadata", "examples"):
        for source in sorted((parent / group).glob("*")):
            if source.is_file():
                destination = release / group / source.name
                digest = copy_unchanged(source, destination)
                preserved.append({"path": destination.relative_to(root).as_posix(),
                                  "sha256": digest, "bytes": source.stat().st_size})
    if len(preserved) != 8:
        raise ValueError(f"expected 8 unchanged Understanding/metadata/example files, copied {len(preserved)}")
    return preserved


def audit_captions(prepare, release, old, receipts, build_triple, caption_flags,
                   verify_crystal):
    counts, flagged, failures = Counter(), [], []
    for split in EXPECTED:
        source_path = prepare.SOURCE / f"{split}-facts.jsonl.gz"
        design_path = release / "design" / f"{NAMES[split]}.jsonl"
        understanding_path = release / "understanding" / f"{NAMES[split]}.jsonl"
        metadata_path = release / "metadata" / f"{NAMES[split]}.jsonl"
        for index, (row, sample, answer, metadata) in enumerate(zip(
                prepare.rows(source_path), prepare.rows(design_path),
                prepare.rows(understanding_path), prepare.rows(metadata_path), strict=True)):
            material_id = row["material_id"]
            reasons = []
            try:
                verify_crystal(row["plaid"]["encoded_text"], row["core_facts"],
                               row["plaid"]["consistency"])
            except Exception as error:
                reasons.append(f"crystal format: {type(error).__name__}: {error}")
            expected_metadata = {"row_index": index, "material_id": material_id,
                "source_split": split, "source_structure_sha256": row["source"]["structure_sha256"],
                "caption_source": "reused_subset" if material_id in old else "full_annotation",
                "core_facts": row["core_facts"], "property_labels": row["property_labels"]}
            if metadata != expected_metadata:
                reasons.append("metadata fields differ from source row")
            if answer != prepare.understanding(row):
                reasons.append("Understanding sample differs from source facts")
            for marker in ("<|im_start|>", "<|im_end|>", "User:", "Assistant:"):
                if any(marker in value for value in sample.values()):
                    reasons.append(f"Design contains chat marker {marker}")
            try:
                if material_id in old:
                    expected_caption = old[material_id]["structure_description"].strip()
                    raw_caption = old[material_id].get("structure_description_raw", expected_caption)
                else:
                    triple = build_triple(row, receipts[material_id], split)
                    expected_caption = triple["structure_description"]
                    raw_caption = triple["structure_description_raw"]
                expected_design = prepare.design(row, expected_caption)
                if sample != expected_design:
                    reasons.append("Design instruction, caption, or crystal differs from source/builder")
                if not expected_caption:
                    reasons.append("Design caption is empty")
                if re.search(r"[\u4e00-\u9fff]", expected_caption):
                    reasons.append("Design caption contains CJK text")
                if material_id in expected_caption:
                    reasons.append("Design caption contains its material ID")
                flags = caption_flags(raw_caption, expected_caption)
                counts.update(flags)
                if flags:
                    flagged.append({"material_id": material_id, "split": split, "flags": flags})
            except Exception as error:
                reasons.append(f"Design format: {type(error).__name__}: {error}")
            if reasons:
                failures.append({"material_id": material_id, "split": split, "reasons": reasons})
    report = {"status": "recorded", "caption_flag_counts": dict(sorted(counts.items())),
              "flagged_rows": len(flagged), "captions_rewritten": False,
              "semantic_scientific_review": "not performed"}
    return report, flagged, failures


def write_minimum_receipt(root, release, parent_sha, receipt_sha, join,
                          preserved, caption_report, condition_report):
    files = {}
    for task in ("design", "understanding"):
        for name in NAMES.values():
            path = release / task / f"{name}.jsonl"
            files[path.relative_to(root).as_posix()] = sha256(path)
    receipt = {
        "status": "exported_and_cpu_validated_seed32_check_pending",
        "remote_release_path": str(release),
        "parent_manifest_sha256": parent_sha,
        "annotation_receipt_sha256": receipt_sha,
        "source_join": join,
        "registry_path": (release / "dataset_info.json").relative_to(root).as_posix(),
        "registry_sha256": sha256(release / "dataset_info.json"),
        "validation_report_sha256": sha256(release / "validation-report.json"),
        "condition_audit": condition_report,
        "caption_flags": caption_report["caption_flag_counts"],
        "unchanged_parent_artifact_hashes": preserved,
        "jsonl_hashes": files,
        "cutoff_2048_full_design_status": "pending full length audit",
    }
    write_json(release / "minimum-receipt.json", receipt)
    return receipt


def audit_tokens(root, release, condition_report):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from experiments.crystal_tasks_v1.configuration import protocol
    from experiments.crystal_tasks_v1.tokenization import make_processor

    config = protocol()
    processor = make_processor("ar_design")
    tokenizer, template = processor.tokenizer, processor.template
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in config["tokens"]]
    if token_ids != CONTROL_IDS or token_ids != config["lora"]["trainable_token_indices"]:
        raise ValueError(f"crystal control tokens are not atomic/stable: {token_ids}")
    for token, expected_id in zip(config["tokens"], CONTROL_IDS, strict=True):
        if tokenizer.encode(token, add_special_tokens=False) != [expected_id]:
            raise ValueError(f"control token is not one atomic token: {token}")

    selected_samples = list(read_jsonl(release / "design/training.jsonl"))
    selected_lengths = []
    for index in condition_report["seed42_sample_indices"]:
        row = selected_samples[index]
        prompt = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": row["output"]}]
        source, target = template.encode_oneturn(tokenizer, messages)
        selected_lengths.append(len(source) + len(target))
    minimum_path = release / "minimum-receipt.json"
    minimum = json.loads(minimum_path.read_text())
    minimum.update({
        "status": "exported_and_cpu_validated_seed32_length_check_complete_full_audit_in_progress",
        "seed42_sample_full_template_lengths": [
            {"material_id": material_id, "tokens": length}
            for material_id, length in zip(condition_report["seed42_sample_ids"],
                                           selected_lengths, strict=True)],
        "seed42_sample_length_summary": summarize_lengths(selected_lengths, CUTOFF),
        "seed42_sample_fits_cutoff_2048": all(length <= CUTOFF for length in selected_lengths),
    })
    write_json(minimum_path, minimum)
    print("MINIMUM_RECEIPT_READY " + json.dumps(minimum, ensure_ascii=False), flush=True)

    by_name = {name: [] for name in ("complete", "prompt", "response", "description", "crystal")}
    by_split = {}
    above = []
    processed = 0
    for split, name in NAMES.items():
        lengths = []
        data = zip(read_jsonl(release / "design" / f"{name}.jsonl"),
                   read_jsonl(release / "metadata" / f"{name}.jsonl"), strict=True)
        for index, (row, metadata) in enumerate(data):
            prompt = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
            caption, crystal = row["output"].split("\n<CRYSTAL_START>")
            crystal = "<CRYSTAL_START>" + crystal
            messages = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": row["output"]}]
            source, target = template.encode_oneturn(tokenizer, messages)
            complete_length = len(source) + len(target)
            if target.count(CONTROL_IDS[0]) != 1 or target.count(CONTROL_IDS[1]) != 1:
                raise ValueError(f"non-atomic or repeated crystal boundary in {name} row {index}")
            start, end = target.index(CONTROL_IDS[0]), target.index(CONTROL_IDS[1])
            if end <= start:
                raise ValueError(f"invalid crystal token span in {name} row {index}")
            desc_length = len(tokenizer.encode(caption, add_special_tokens=False))
            crystal_length = len(tokenizer.encode(crystal, add_special_tokens=False))
            by_name["complete"].append(complete_length)
            by_name["prompt"].append(len(source))
            by_name["response"].append(len(target))
            by_name["description"].append(desc_length)
            by_name["crystal"].append(crystal_length)
            lengths.append(complete_length)
            if complete_length > CUTOFF:
                above.append({"material_id": metadata["material_id"],
                              "split": split, "row_index": index, "tokens": complete_length})
            processed += 1
            if processed % 5000 == 0:
                print(f"TOKEN_AUDIT_PROGRESS {processed}/{sum(EXPECTED.values())}", flush=True)
        by_split[split] = summarize_lengths(lengths, CUTOFF)
        if len(lengths) != EXPECTED[split]:
            raise ValueError(f"token audit row count differs for {split}: {len(lengths)}")

    return {
        "status": "measured_without_truncation",
        "model": config["model_name"],
        "model_revision": config["model_revision"],
        "tokenizer_path": config["model_path"],
        "template": "qwen3",
        "enable_thinking": False,
        "method": "LLaMA-Factory template.encode_oneturn; no cutoff/truncation applied",
        "control_tokens": dict(zip(config["tokens"], token_ids, strict=True)),
        "control_token_ids_atomic": True,
        "cutoff_under_review": CUTOFF,
        "cutoff_2048_sufficient_for_all_design_rows": not above,
        "rows_above_2048": above,
        "component_definition": {
            "complete": "template.encode_oneturn source plus target, including chat framing",
            "description": "caption encoded alone without chat framing or separator newline",
            "crystal": "<CRYSTAL_START> through <CRYSTAL_END> encoded alone",
        },
        "lengths": {name: summarize_lengths(values, CUTOFF)
                    for name, values in by_name.items()},
        "by_split_complete": by_split,
        "llamafactory_version": importlib.metadata.version("llamafactory"),
    }


def audit_conditions(root, release):
    metadata = list(read_jsonl(release / "metadata/training.jsonl"))
    samples = list(read_jsonl(release / "design/training.jsonl"))
    if len(metadata) != EXPECTED["train"] or len(samples) != EXPECTED["train"]:
        raise ValueError("training condition audit requires the full frozen training split")

    condition_pairs, prompt_pairs, prompt_counts = set(), set(), Counter()
    for meta, sample in zip(metadata, samples, strict=True):
        sg = meta["core_facts"]["space_group_number"]
        ehull = Decimal(str(meta["property_labels"]["energy_above_hull"]))
        condition_pairs.add((sg, format(ehull.normalize(), "f")))
        bound = re.fullmatch(
            r"Generate a crystal with space group number (\d+) and energy above hull no greater than ([\d.]+) eV/atom\.",
            sample["instruction"])
        if not bound or int(bound[1]) != sg or ehull > Decimal(bound[2]):
            raise ValueError(f"training condition does not match source label: {meta['material_id']}")
        prompt_pairs.add((sg, bound[2]))
        prompt_counts[sample["instruction"]] += 1

    selected = sorted(random.Random(42).sample(range(len(metadata)), 32))
    selected_counts = Counter(samples[index]["instruction"] for index in selected)
    repeated = [{"instruction": prompt, "selected_rows": count}
                for prompt, count in sorted(selected_counts.items()) if count > 1]
    return {
        "training_rows": len(metadata),
        "unique_space_group_exact_ehull_pairs": len(condition_pairs),
        "unique_space_group_ehull_upper_bound_prompts": len(prompt_pairs),
        "unique_training_input_prompts": len(prompt_counts),
        "seed": 42,
        "seed42_sample_indices": selected,
        "seed42_sample_ids": [metadata[index]["material_id"] for index in selected],
        "seed42_sample_unique_input_prompts": len(selected_counts),
        "seed42_sample_repeated_input_prompts": repeated,
        "recommendation": (
            "Use 32 distinct condition prompts for a separate diagnostic; leave full training data unchanged."
            if repeated else "The seed-42 32-row diagnostic sample already has distinct condition prompts."),
    }


def export(project_root, transfer_dir, release_dir, expected_receipt_sha, resume=False):
    root = Path(project_root).resolve()
    transfer_dir = Path(transfer_dir).resolve()
    release = Path(release_dir).resolve()
    if release != (root / RELEASE_REL).resolve():
        raise ValueError(f"release path must be {root / RELEASE_REL}")
    if (transfer_dir.parent != root / "data_preparation/crystal_tasks_v1"
            or not transfer_dir.name.startswith(TRANSFER_PREFIX)):
        raise ValueError("transfer files must use a unique project-owned transfer directory")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only export requires CUDA_VISIBLE_DEVICES to be empty")

    root_manifest = root / "datasets/crystal_tasks_v1/manifest.json"
    parent_sha = sha256(root_manifest)
    sys.path.insert(0, str(root))
    prep_dir = root / "data_preparation/crystal_tasks_v1"
    sys.path.insert(0, str(prep_dir))
    import prepare
    from experiments.crystal_tasks_v1.configuration import protocol

    if parent_sha != protocol()["manifest_sha256"]:
        raise ValueError("frozen v1 parent manifest differs from the approved protocol")
    parent_manifest = json.loads(root_manifest.read_text())
    if parent_manifest.get("dataset_version") != "crystal_tasks_v1":
        raise ValueError("unexpected parent dataset version")

    marker = release / ".export_design_complete-incomplete.json"
    if release.exists():
        if not resume or not marker.is_file():
            raise FileExistsError(f"release exists; only explicit resume of this incomplete release is allowed: {release}")
        state = json.loads(marker.read_text())
        if (state.get("release_path") != RELEASE_REL.as_posix()
                or state.get("parent_manifest_sha256") != parent_sha
                or state.get("transfer_dir") != transfer_dir.name
                or state.get("receipt_sha256") != expected_receipt_sha):
            raise ValueError("incomplete release marker does not match this run")
    else:
        if resume:
            raise FileNotFoundError("--resume was requested but no incomplete release exists")
        release.parent.mkdir(parents=True, exist_ok=True)
        release.mkdir(exist_ok=False)
        write_json(marker, {"release_path": RELEASE_REL.as_posix(),
                            "parent_manifest_sha256": parent_sha,
                            "transfer_dir": transfer_dir.name,
                            "receipt_sha256": expected_receipt_sha,
                            "started_at": datetime.now(timezone.utc).isoformat()})

    prepare.check_sources()
    source_manifest_path = root / "datasets/sources/crystal_facts_v1/manifest.json"
    if sha256(source_manifest_path) != parent_manifest["source_manifest_sha256"]:
        raise ValueError("parent's source manifest hash no longer matches")
    prepare.SOURCE = root / "datasets/sources/crystal_facts_v1"
    prepare.OLD = root / "datasets/sources/subset_captions_20261013"
    prepare.DATA = release
    prepare.STAGE = transfer_dir
    old = prepare.old_captions()
    receipts, receipt_split_counts, receipt_path = verify_receipts(
        prepare, transfer_dir, expected_receipt_sha, 37293)
    join = verify_source_join(prepare, old, receipts)
    if receipt_split_counts != join["new_by_split"]:
        raise ValueError("annotation receipt split counts differ from missing source IDs")

    preserved = copy_parent_files(root, release, parent_sha)
    print("EXPORT_FINALIZE_START", flush=True)
    from data_preparation.crystal_training_preflight_20261013.assemble import build_triple
    try:
        prepare.finalize()
    except Exception as error:
        failures = collect_finalize_failures(prepare, old, receipts, build_triple)
        write_json(release / "format-validation-failures.json", {
            "status": "failed", "filtered_rows": 0,
            "finalizer_error": f"{type(error).__name__}: {error}",
            "sample_failures": failures,
        })
        print(json.dumps({"format_validation_failed": True,
                          "sample_failures": failures}, ensure_ascii=False), flush=True)
        raise
    if sha256(root_manifest) != parent_sha:
        raise ValueError("the frozen parent manifest changed during export")

    import validate
    from data_preparation.crystal_training_preflight_20261013.assemble import caption_flags
    caption_report, flagged_rows, format_failures = audit_captions(
        prepare, release, old, receipts, build_triple, caption_flags,
        validate.verify_crystal)
    write_json(release / "caption-flags.json", caption_report)
    write_json(release / "caption-flagged-rows.json", flagged_rows)
    write_json(release / "format-validation-failures.json", {
        "status": "passed" if not format_failures else "failed",
        "filtered_rows": 0, "sample_failures": format_failures})
    if format_failures:
        print(json.dumps({"format_validation_failed": True,
                          "sample_failures": format_failures}, ensure_ascii=False), flush=True)
        raise ValueError(f"format validator found {len(format_failures)} row(s); see format-validation-failures.json")

    print("CPU_SHARED_VALIDATOR_START", flush=True)
    try:
        validate.main(understanding_only=False)
    except Exception as error:
        write_json(release / "format-validation-failures.json", {
            "status": "failed", "filtered_rows": 0,
            "shared_validator_error": f"{type(error).__name__}: {error}",
            "sample_failures": format_failures,
            "material_level_precheck_passed": True,
        })
        raise
    print("CPU_SHARED_VALIDATOR_PASSED", flush=True)

    condition_report = audit_conditions(root, release)
    write_json(release / "condition-audit.json", condition_report)

    validation_path = release / "validation-report.json"
    validation = json.loads(validation_path.read_text())
    if validation.get("status") != "passed" or validation.get("unique_materials") != 45465:
        raise ValueError("shared Design+Understanding validator did not pass the full dataset")
    registry = json.loads((release / "dataset_info.json").read_text())
    if len(registry) != 6:
        raise ValueError(f"expected six LLaMA-Factory task/split entries, found {len(registry)}")
    validation.update({
        "parent_manifest_sha256": parent_sha,
        "source_artifact_hashes_verified": True,
        "source_manifest_sha256": sha256(source_manifest_path),
        "receipt_sha256": expected_receipt_sha,
        "exact_receipt_and_source_id_sets_match": True,
        "reused_caption_source_facts_hashes_match": True,
        "design_descriptions_match_existing_builder": True,
        "unchanged_understanding_metadata_examples_byte_hashes_verified": True,
        "caption_flags_recorded_without_rewriting": True,
        "caption_flags": caption_report["caption_flag_counts"],
        "six_task_registry_entries": True,
    })
    for item in preserved:
        relative = Path(item["path"]).relative_to(RELEASE_REL)
        if sha256(release / relative) != item["sha256"]:
            raise ValueError(f"copied Understanding/metadata/example hash changed: {relative}")
        if sha256(root / "datasets/crystal_tasks_v1" / relative) != item["sha256"]:
            raise ValueError(f"frozen v1 source artifact hash changed: {relative}")
    write_json(validation_path, validation)

    minimum = write_minimum_receipt(root, release, parent_sha, expected_receipt_sha,
                                    join, preserved, caption_report,
                                    condition_report)
    print("MINIMUM_RECEIPT_READY " + json.dumps(minimum, ensure_ascii=False), flush=True)

    token_report = audit_tokens(root, release, condition_report)
    write_json(release / "token-length-audit.json", token_report)

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("annotation_state", None)
    manifest.pop("new_captions_saved", None)
    manifest.update({
        "status": "both_tasks_ready",
        "dataset_version": "crystal_tasks_v1_design_full_20261026",
        "parent_dataset_version": "crystal_tasks_v1",
        "parent_manifest_sha256": parent_sha,
        "source_manifest_sha256": sha256(source_manifest_path),
        "new_captions": len(receipts),
        "reused_captions": len(old),
        "missing_captions": {split: 0 for split in EXPECTED},
        "annotation_receipt_sha256": expected_receipt_sha,
        "annotation_receipt_rows": len(receipts),
        "annotation_receipt_split_counts": receipt_split_counts,
        "source_join": join,
        "preserved_parent_artifacts": preserved,
        "caption_flags": caption_report["caption_flag_counts"],
        "caption_flags_disposition": "reported_only_all_rows_retained_captions_unchanged",
        "caption_scientific_qa": "automated consistency and formatting checks; semantic review not performed",
        "token_audit_path": (release / "token-length-audit.json").relative_to(root).as_posix(),
        "condition_audit_path": (release / "condition-audit.json").relative_to(root).as_posix(),
        "validation_report_sha256": sha256(validation_path),
        "token_audit_sha256": sha256(release / "token-length-audit.json"),
        "condition_audit_sha256": sha256(release / "condition-audit.json"),
        "artifact_path_base": "project_root",
        "cutoff_2048_sufficient_for_all_design_rows": token_report[
            "cutoff_2048_sufficient_for_all_design_rows"],
        "seed42_32_cutoff_2048_fit": json.loads(
            (release / "minimum-receipt.json").read_text())["seed42_sample_fits_cutoff_2048"],
    })
    artifacts = {}
    artifact_dirs = ("design", "understanding", "metadata", "examples")
    for directory in artifact_dirs:
        for path in sorted((release / directory).glob("*")):
            if path.is_file():
                artifacts[path.relative_to(root).as_posix()] = {
                    "sha256": sha256(path), "bytes": path.stat().st_size}
    for name in ("dataset_info.json", "validation-report.json", "token-length-audit.json",
                 "condition-audit.json", "caption-flags.json", "caption-flagged-rows.json",
                 "format-validation-failures.json"):
        path = release / name
        artifacts[path.relative_to(root).as_posix()] = {
            "sha256": sha256(path), "bytes": path.stat().st_size}
    manifest["artifacts"] = artifacts
    write_json(manifest_path, manifest)

    receipt = {
        "status": "passed",
        "remote_release_path": str(release),
        "parent_manifest_sha256": parent_sha,
        "manifest_sha256": sha256(manifest_path),
        "validation_report_sha256": sha256(validation_path),
        "token_audit_sha256": sha256(release / "token-length-audit.json"),
        "condition_audit_sha256": sha256(release / "condition-audit.json"),
        "caption_flag_report_sha256": sha256(release / "caption-flags.json"),
        "annotation_receipt_sha256": expected_receipt_sha,
        "annotation_receipt_rows": len(receipts),
        "condition_audit": condition_report,
        "caption_flags": caption_report["caption_flag_counts"],
        "token_summary": token_report["lengths"],
        "cutoff_2048_sufficient_for_all_design_rows": token_report[
            "cutoff_2048_sufficient_for_all_design_rows"],
        "seed42_32_cutoff_2048_fit": json.loads(
            (release / "minimum-receipt.json").read_text())["seed42_sample_fits_cutoff_2048"],
        "code_sha256": sha256(Path(__file__)),
        "artifact_hashes": artifacts,
    }
    write_json(release / "compact-receipt.json", receipt)
    if sha256(root_manifest) != parent_sha:
        raise ValueError("frozen v1 parent manifest changed")
    marker.unlink()
    receipt_path.unlink()
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--transfer-dir", required=True, type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--receipts-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    export(args.project_root, args.transfer_dir, args.release_dir,
           args.receipts_sha256, args.resume)


if __name__ == "__main__":
    main()
