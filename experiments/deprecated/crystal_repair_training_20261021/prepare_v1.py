"""Prepare strict-audited, deterministic repair examples from the frozen 7272/900 data."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DATA = ROOT / "research/crystal_training_preflight_20261013/data_001"
SCORE_PATH = ROOT / "experiments/interleaved_ar_diffusion_20261002/score.py"
TASKS_PATH = ROOT / "experiments/structure_conditioned_ar8b_20261017/prepare.py"
TOKENIZER_TRAIN_PATH = ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015/train.py"
TOKENIZER_PATH = ROOT / "models/Qwen3-8B-b968826d9c46"
SPLITS = {"train": 7272, "val": 900}
EXPECTED_INPUT_HASHES = {
    "manifest_sha256": "aa21f3fbb4e9a19a676d1012405575932134675c3684d336d2c41283a281e741",
    "train_sha256": "b74afafbca5a52eb1a63c9a7502c9fe1f5055cfef0c25f122de6d140c137080b",
    "val_sha256": "4e97dbe73aa2c4903aa11c4438f3c7c4da05a87eb1028632f121db4e8cd97f12",
}
SEED = 42
TYPES = ("formula_count", "sites_count", "wyckoff_multiplicity", "coordinate_constraint")
EXPECTED_REASON = {
    "formula_count": "formula_header_mismatch",
    "sites_count": "sites_header_mismatch",
    "wyckoff_multiplicity": "Wyckoff multiplicity does not match label",
    "coordinate_constraint": "wyckoff_coordinate_constraint",
}
MAX_CONTEXT = 4096
START = "<CRYSTAL_START>"
_SCORE = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_score():
    global _SCORE
    if _SCORE is None:
        _SCORE = _load_module("crystal_repair_frozen_score", SCORE_PATH)
    return _SCORE


def _worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    _load_score()


def _cache_write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _cache_read(path, cache_key):
    if path.exists():
        record = json.loads(path.read_text())
        if record["cache_key"] == cache_key:
            return record
    return None


def _expected_failure(audit, kind):
    if audit.get("valid") is not False:
        return False
    reason = EXPECTED_REASON[kind]
    return any(reason in str(value) for value in audit.get("reasons", []))


def _candidate(lines, line_index, fields, details, audit_fn, kind):
    changed = list(lines)
    changed[line_index] = " ".join(fields)
    text = "\n".join(changed)
    audit = audit_fn(text)
    return (text, details, audit) if _expected_failure(audit, kind) else None


def _mutations(text, kind, material_id, seed, audit_fn, symmetry_group=None):
    lines = text.splitlines()
    rng = random.Random(int(hashlib.sha256(f"{seed}:{material_id}:{kind}".encode()).hexdigest()[:16], 16))
    if kind == "formula_count":
        pieces = list(re.finditer(r"([A-Z][a-z]?)(\d*)", lines[0]))
        rng.shuffle(pieces)
        for piece in pieces:
            old_count = int(piece[2] or "1")
            new_count = old_count + 1
            header = lines[0][:piece.start()] + piece[1] + str(new_count) + lines[0][piece.end():]
            result = _candidate(lines, 0, [header], {
                "field": "formula_header.count", "element": piece[1],
                "target_count": old_count, "corrupted_count": new_count,
            }, audit_fn, kind)
            if result:
                return result
    elif kind == "sites_count":
        count = int(re.fullmatch(r"Sites \((\d+)\)", lines[4])[1])
        result = _candidate(lines, 4, [f"Sites ({count + 1})"], {
            "field": "Sites header count", "target_count": count, "corrupted_count": count + 1,
        }, audit_fn, kind)
        if result:
            return result
    elif kind == "wyckoff_multiplicity":
        indices = list(range(5, len(lines)))
        rng.shuffle(indices)
        for index in indices:
            fields = lines[index].split()
            multiplicity, letter = int(fields[4][:-1]), fields[4][-1]
            new_label = f"{multiplicity + 1}{letter}"
            fields[4] = new_label
            result = _candidate(lines, index, fields, {
                "field": "Wyckoff multiplicity", "element": fields[0],
                "target_label": f"{multiplicity}{letter}", "corrupted_label": new_label,
            }, audit_fn, kind)
            if result:
                return result
    elif kind == "coordinate_constraint":
        if symmetry_group is None:
            return None
        group = symmetry_group(lines[1].split(": ", 1)[1])
        indices = list(range(5, len(lines)))
        rng.shuffle(indices)
        for index in indices:
            fields = lines[index].split()
            if group.get_wyckoff_position(fields[4]).get_dof() >= 3:
                continue
            axes = list(range(1, 4))
            rng.shuffle(axes)
            offsets = [0.137, 0.281, 0.419]
            rng.shuffle(offsets)
            for axis in axes:
                target = float(fields[axis])
                for offset in offsets:
                    value = f"{(target + offset) % 1:.3f}"
                    if value == fields[axis] or value == "1.000":
                        continue
                    changed_fields = fields.copy()
                    changed_fields[axis] = value
                    result = _candidate(lines, index, changed_fields, {
                        "field": "Wyckoff representative coordinate", "element": fields[0],
                        "label": fields[4], "axis": "xyz"[axis - 1],
                        "target_coordinate": fields[axis], "corrupted_coordinate": value,
                    }, audit_fn, kind)
                    if result:
                        return result
    else:
        raise ValueError(f"unknown corruption type: {kind}")
    return None


def corrupt_structure(text, kind, material_id, seed, audit_fn, symmetry_group=None):
    """Return one mutation only when the frozen auditor reports its intended failure."""
    result = _mutations(text, kind, material_id, seed, audit_fn, symmetry_group)
    if result is None:
        raise ValueError(f"{kind} is ineligible for {material_id}")
    return result


def build_corruption(text, requested_type, material_id, seed, audit_fn, symmetry_group=None):
    start = TYPES.index(requested_type)
    for offset in range(len(TYPES)):
        kind = TYPES[(start + offset) % len(TYPES)]
        result = _mutations(text, kind, material_id, seed, audit_fn, symmetry_group)
        if result is not None:
            structure, edit, audit = result
            return {"requested_type": requested_type, "type": kind, "edit": edit,
                    "structure": structure, "audit": audit}
    raise RuntimeError(f"no verified single-field corruption available for {material_id}")


def build_task_r(generation_task, corrupted, feedback, explanation):
    request = generation_task["prefix"].removeprefix("User: ").removesuffix("\nAssistant:")
    prefix = ("User: " + request + "\nPrevious failed crystal (input data):\n" + corrupted +
              "\nChecker feedback:\n" + feedback + "\nAssistant:")
    return {"prefix": prefix, "lead": " " + explanation.strip() + "\n" + START,
            "crystal": generation_task["crystal"]}


def checker_feedback(request, audit):
    actual = audit.get("detected_sg")
    actual_text = str(actual) if actual is not None else "undetermined"
    facts = audit.get("facts") or {}
    fact_text = ", ".join(f"{key}={facts[key]}" for key in ("formula", "sg", "atoms", "orbits")
                           if key in facts) or "none recorded"
    reasons = ", ".join(map(str, audit.get("reasons") or [])) or "none recorded"
    return (f"Checker feedback: the original request requires {request}. The decoded space group is "
            f"{actual_text}. The strict structural audit failed; recorded reasons: {reasons}. "
            f"Recorded structure facts: {fact_text}. The ehull target has not been evaluated. "
            "Write a brief revised design description, then generate a new crystal in the same format.")


def correction_explanation(edit):
    if edit["field"] == "formula_header.count":
        return (f"The formula header count for {edit['element']} is corrected from "
                f"{edit['corrupted_count']} to {edit['target_count']}.")
    if edit["field"] == "Sites header count":
        return (f"The declared Sites count is corrected from {edit['corrupted_count']} "
                f"to {edit['target_count']}.")
    if edit["field"] == "Wyckoff multiplicity":
        return (f"The {edit['element']} orbit label is corrected from "
                f"{edit['corrupted_label']} to {edit['target_label']}.")
    return (f"The {edit['element']} orbit at {edit['label']} has its {edit['axis']} coordinate "
            f"corrected from {edit['corrupted_coordinate']} to {edit['target_coordinate']}.")


def _atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _read_source():
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = {}
    source_hashes = {"manifest_sha256": sha256_file(manifest_path)}
    for split, expected_count in SPLITS.items():
        path = DATA / f"{split}-triples.jsonl.gz"
        source_hashes[f"{split}_sha256"] = sha256_file(path)
        if source_hashes[f"{split}_sha256"] != manifest["artifacts"][path.name]["sha256"]:
            raise ValueError(f"frozen source hash changed: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            split_rows = [json.loads(line) for line in source]
        if len(split_rows) != expected_count or manifest["counts"][split] != expected_count:
            raise ValueError(f"frozen {split} row count changed")
        ids = [row["material_id"] for row in split_rows]
        if len(set(ids)) != expected_count:
            raise ValueError(f"duplicate material IDs in frozen {split}")
        if any(row.get("split") != split for row in split_rows):
            raise ValueError(f"frozen split label changed in {split}")
        rows[split] = [{"row": row, "source_row_sha256": json_sha256(row)} for row in split_rows]
    train_ids = {item["row"]["material_id"] for item in rows["train"]}
    val_ids = {item["row"]["material_id"] for item in rows["val"]}
    if train_ids & val_ids:
        raise ValueError("frozen train and validation IDs overlap")
    if source_hashes != EXPECTED_INPUT_HASHES:
        raise ValueError("the frozen preflight input hashes changed")
    return manifest, source_hashes, rows


def _target_audit_job(job):
    split, row, source_row_sha256, cache_dir, key = job
    path = Path(cache_dir) / "target" / split / f"{row['material_id']}.json"
    cached = _cache_read(path, key)
    if cached:
        return cached
    audit = _load_score().audit(row["plaid_text"])
    record = {"cache_key": key, "material_id": row["material_id"], "split": split,
              "source_row_sha256": source_row_sha256, "audit": audit}
    _cache_write(path, record)
    return record


def _corruption_job(job):
    split, row, source_row_sha256, requested_type, seed, cache_dir, key = job
    path = Path(cache_dir) / "corrupted" / split / f"{row['material_id']}.json"
    cached = _cache_read(path, key)
    if cached:
        return cached
    score = _load_score()
    result = build_corruption(row["plaid_text"], requested_type, row["material_id"], seed,
                              score.audit, score.group)
    record = {"cache_key": key, "material_id": row["material_id"], "split": split,
              "source_row_sha256": source_row_sha256, "seed": seed, **result}
    _cache_write(path, record)
    return record


def _pool_map(worker, jobs, workers, label):
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_init) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if count % 500 == 0 or count == len(futures):
                print(f"{label}: {count}/{len(futures)}", flush=True)
    return results


def _tokenizer(path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    train_dir = str(TOKENIZER_TRAIN_PATH.parent)
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)
    train = _load_module("crystal_repair_tokenizer_train_contract", TOKENIZER_TRAIN_PATH)
    tokenizer = train.load_tokenizer(path)
    task_module = _load_module("crystal_repair_source_tasks", TASKS_PATH)
    tokens = (task_module.START, task_module.END, task_module.MASK, task_module.PAD)
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in tokens}
    files = {name: sha256_file(path / name) for name in
             ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")
             if (path / name).exists()}
    return tokenizer, {"model_path": str(path.relative_to(ROOT)), "files_sha256": files,
                       "eos_token": task_module.EOS, "special_tokens": token_ids}


def _token_lengths(task_r, original_tasks, corrupted, tokenizer, end):
    def length(text):
        return len(tokenizer.encode(text, add_special_tokens=False))
    source_g, source_q = original_tasks
    return {
        "source_g_full": length(source_g["prefix"] + source_g["lead"] + source_g["crystal"] + end),
        "source_q_full": length(source_q["prefix"] + source_q["target"]),
        "repair_prefix": length(task_r["prefix"]),
        "repair_input_crystal": length(corrupted),
        "repair_lead": length(task_r["lead"]),
        "repair_target_crystal_end": length(task_r["crystal"] + end),
        "repair_full_exact": length(task_r["prefix"] + task_r["lead"] + task_r["crystal"] + end),
    }


def _percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {"count": len(ordered), "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[(95 * len(ordered) + 99) // 100 - 1], "max": ordered[-1]}


def run(args):
    if args.workers < 1:
        raise ValueError("workers must be positive")
    out = args.out.resolve()
    if not out.is_relative_to(EXPERIMENT.resolve()):
        raise ValueError(f"output must stay under {EXPERIMENT}")
    if (out / "manifest.json").exists():
        raise FileExistsError(f"completed output already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest, source_hashes, splits = _read_source()
    source_identity = {
        "source_hashes": source_hashes,
        "score_sha256": sha256_file(SCORE_PATH),
        "plaid_utils_sha256": sha256_file(ROOT / "work/PLaID/utils.py"),
        "prepare_sha256": sha256_file(Path(__file__)),
        "task_prepare_sha256": sha256_file(TASKS_PATH),
        "seed": SEED,
        "types": TYPES,
        "pyxtal": importlib.metadata.version("pyxtal"),
        "pymatgen": importlib.metadata.version("pymatgen"),
    }
    audit_key = json_sha256(source_identity)
    flat = [(split, item) for split in SPLITS for item in splits[split]]
    target_jobs = [(split, item["row"], item["source_row_sha256"], str(out / "audit-cache"), audit_key)
                   for split, item in flat]
    print(f"source audit started: train={SPLITS['train']} val={SPLITS['val']} workers={args.workers}", flush=True)
    target_records = _pool_map(_target_audit_job, target_jobs, args.workers, "target audit")
    audited = {(record["split"], record["material_id"]): record["audit"] for record in target_records}
    eligibility, invalid = [], []
    eligible = {"train": [], "val": []}
    for split, item in flat:
        row = item["row"]
        audit = audited[(split, row["material_id"])]
        accepted = audit.get("valid") is True
        record = {"material_id": row["material_id"], "split": split, "eligible": accepted,
                  "source_row_sha256": item["source_row_sha256"], "targetaudit": audit}
        eligibility.append(record)
        if accepted:
            eligible[split].append(item)
        else:
            invalid.append({"material_id": row["material_id"], "split": split,
                            "source_row_sha256": item["source_row_sha256"],
                            "reasons": audit.get("reasons", []), "targetaudit": audit})
    _atomic_jsonl(out / "eligibility.jsonl", eligibility)
    _atomic_jsonl(out / "invalid-targets.jsonl", invalid)

    type_by_id = {}
    ordered = sorted(((split, item) for split in SPLITS for item in eligible[split]),
                     key=lambda pair: hashlib.sha256(f"{SEED}:{pair[1]['row']['material_id']}".encode()).digest())
    for index, (split, item) in enumerate(ordered):
        type_by_id[(split, item["row"]["material_id"])] = TYPES[index % len(TYPES)]
    corruption_jobs = []
    for split in SPLITS:
        for item in eligible[split]:
            material_id = item["row"]["material_id"]
            requested = type_by_id[(split, material_id)]
            key = json_sha256({**source_identity, "requested_type": requested,
                               "corruption_algorithm": "single-field-v1"})
            corruption_jobs.append((split, item["row"], item["source_row_sha256"], requested,
                                    SEED, str(out / "audit-cache"), key))
    corruption_records = _pool_map(_corruption_job, corruption_jobs, args.workers, "corruption audit")
    corruptions = {(record["split"], record["material_id"]): record for record in corruption_records}

    task_module = _load_module("crystal_repair_source_tasks", TASKS_PATH)
    tokenizer, tokenizer_info = _tokenizer(args.model_path.resolve())
    data_rows = {"train": [], "val": []}
    token_summary = {key: [] for key in (
        "source_g_full", "source_q_full", "repair_prefix", "repair_input_crystal",
        "repair_lead", "repair_target_crystal_end", "repair_full_exact")}
    requested_counts, actual_counts, fallback_counts = Counter(), Counter(), Counter()
    for split in SPLITS:
        for item in eligible[split]:
            row = item["row"]
            material_id = row["material_id"]
            g, q = task_module.tasks(row)
            if (g["task"], q["task"]) != ("G", "Q"):
                raise ValueError(f"source task contract changed for {material_id}")
            record = corruptions[(split, material_id)]
            audit = audited[(split, material_id)]
            request = g["prefix"].removeprefix("User: ").removesuffix("\nAssistant:")
            feedback = checker_feedback(request, record["audit"])
            explanation = correction_explanation(record["edit"])
            task_r = build_task_r(g, record["structure"], feedback, explanation)
            case_hash = hashlib.sha256(f"{SEED}:{material_id}:{item['source_row_sha256']}".encode()).hexdigest()[:12]
            corruption = {"case_id": f"r42-{material_id}-{case_hash}", "seed": SEED,
                          "requested_type": record["requested_type"], "type": record["type"],
                          "edit": record["edit"], "explanation": explanation,
                          "source_row_sha256": item["source_row_sha256"],
                          "target_crystal_sha256": hashlib.sha256(row["plaid_text"].encode()).hexdigest(),
                          "corrupted_crystal_sha256": hashlib.sha256(record["structure"].encode()).hexdigest()}
            output = {"material_id": material_id, "taskR": task_r, "original_tasks": [g, q],
                      "corruption": corruption, "targetaudit": audit, "corrupted_audit": record["audit"]}
            output["tokenlengths"] = _token_lengths(task_r, [g, q], record["structure"],
                                                    tokenizer, task_module.END)
            data_rows[split].append(output)
            for key, value in output["tokenlengths"].items():
                token_summary[key].append(value)
            requested_counts[record["requested_type"]] += 1
            actual_counts[record["type"]] += 1
            if record["type"] != record["requested_type"]:
                fallback_counts[f"{record['requested_type']}->{record['type']}"] += 1

    train_ids = {row["material_id"] for row in data_rows["train"]}
    val_ids = {row["material_id"] for row in data_rows["val"]}
    if train_ids & val_ids:
        raise ValueError("eligible train and validation IDs overlap")
    for split in SPLITS:
        _atomic_jsonl(out / f"{split}.jsonl", data_rows[split])
    artifacts = {}
    for name in ("train.jsonl", "val.jsonl", "eligibility.jsonl", "invalid-targets.jsonl"):
        artifacts[name] = {"sha256": sha256_file(out / name), "bytes": (out / name).stat().st_size}
    output_manifest = {
        "schema_version": 1,
        "source": {"directory": str(DATA.relative_to(ROOT)), **source_hashes,
                   "immutable_manifest_sha256": source_hashes["manifest_sha256"],
                   "counts": manifest["counts"]},
        "audit_identity": source_identity,
        "corruption_seed": SEED,
        "corruption_types": list(TYPES),
        "counts": {"input": dict(SPLITS), "eligible": {split: len(eligible[split]) for split in SPLITS},
                   "invalid_target": {split: sum(row["split"] == split for row in invalid) for split in SPLITS},
                   "output": {split: len(data_rows[split]) for split in SPLITS}},
        "eligibility": "only source targets with frozen score.audit(valid=True) receive repair labels; invalid targets are recorded and excluded",
        "corruption_counts": {"requested": dict(requested_counts), "actual": dict(actual_counts),
                              "fallback": dict(fallback_counts)},
        "row_schema": {"material_id": "frozen source identifier", "taskR": ["prefix", "lead", "crystal"],
                       "original_tasks": ["G", "Q"], "corruption": ["case_id", "seed", "requested_type",
                       "type", "edit", "explanation", "source_row_sha256", "target_crystal_sha256",
                       "corrupted_crystal_sha256"], "audits": ["targetaudit", "corrupted_audit"],
                       "tokenlengths": "exact Qwen3 token counts; full sequences include CRYSTAL_END"},
        "tokenizer": tokenizer_info,
        "token_lengths": {name: _percentiles(values) for name, values in token_summary.items()},
        "context_limits": {"source_GQ_review_threshold": 2048, "repair_full_max": MAX_CONTEXT,
                           "source_G_over_2048": sum(row["tokenlengths"]["source_g_full"] > 2048
                               for rows in data_rows.values() for row in rows),
                           "source_Q_over_2048": sum(row["tokenlengths"]["source_q_full"] > 2048
                               for rows in data_rows.values() for row in rows),
                           "source_GQ_materials_over_2048": sum(1 for row in data_rows["train"] + data_rows["val"]
                               if max(row["tokenlengths"]["source_g_full"], row["tokenlengths"]["source_q_full"]) > 2048),
                           "repair_over_4096": sum(row["tokenlengths"]["repair_full_exact"] > MAX_CONTEXT
                               for rows in data_rows.values() for row in rows)},
        "excluded_benchmark_sources": ["70 benchmark groups", "122 retries"],
        "artifacts": artifacts,
    }
    temporary_manifest = out / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n")
    temporary_manifest.replace(out / "manifest.json")
    summary = {key: output_manifest[key] for key in ("counts", "corruption_counts", "token_lengths", "context_limits")}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output directory beneath experiments/crystal_repair_training_20261021")
    parser.add_argument("--workers", type=int, default=8, help="isolated CPU audit processes")
    parser.add_argument("--model-path", type=Path, default=TOKENIZER_PATH,
                        help="local Qwen3-8B tokenizer directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
