"""Compare parent FLARE and repair-trained FLARE on fixed, corrupted validation requests."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
TRAINER_PATH = EXPERIMENT / "run.py"
LOOP_PATH = ROOT / "experiments/structure_feedback_loop_20261020/run.py"
SEED = 42
ITEMS_PER_TYPE = 8
CORRUPTION_TYPES = (
    "formula_count", "sites_count", "wyckoff_multiplicity", "coordinate_constraint",
)
DATASET_HASH_KEYS = ("train_sha256", "val_sha256", "manifest_sha256", "source_manifest_sha256")
INPUT_MARKER = "Previous failed crystal (input data):\n"
FEEDBACK_MARKER = "\nChecker feedback:\n"
ARMS = ("parent", "new")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{id(value)}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ensure_shared_json(path, expected):
    path = Path(path)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"existing manifest differs: {path}")
        return
    _write_json_atomic(path, expected)


def _input_crystal_body(prefix):
    if prefix.count(INPUT_MARKER) != 1 or prefix.count(FEEDBACK_MARKER) != 1:
        raise ValueError("repair prefix must contain one corrupted-crystal section")
    _, _, content = prefix.partition(INPUT_MARKER)
    body, marker, _ = content.partition(FEEDBACK_MARKER)
    if not marker:
        raise ValueError("repair prefix has a malformed corrupted-crystal section")
    return body


def _outer_whitespace_normalized_body_match(generated, reference):
    return generated.strip() == reference.strip()


def _row_identity(row):
    material_id = row.get("material_id")
    if (not isinstance(material_id, str) or not material_id
            or Path(material_id).name != material_id or material_id in (".", "..")):
        raise ValueError("validation row has an invalid material_id")
    task = row.get("taskR")
    corruption = row.get("corruption")
    target_audit = row.get("targetaudit")
    if (not isinstance(task, dict) or not isinstance(task.get("prefix"), str)
            or not isinstance(task.get("lead"), str) or not isinstance(task.get("crystal"), str)
            or not isinstance(corruption, dict) or not isinstance(target_audit, dict)):
        raise ValueError(f"repair validation row is incomplete: {material_id}")
    prefix = task["prefix"]
    if "<CRYSTAL_START>" in prefix or "<CRYSTAL_END>" in prefix:
        raise ValueError(f"repair prefix contains a forced crystal boundary: {material_id}")
    body = _input_crystal_body(prefix)
    if _sha256_text(body) != corruption.get("corrupted_crystal_sha256"):
        raise ValueError(f"repair input crystal hash differs: {material_id}")
    target = task["crystal"]
    if _sha256_text(target) != corruption.get("target_crystal_sha256"):
        raise ValueError(f"repair target crystal hash differs: {material_id}")
    kind = corruption.get("type")
    facts = target_audit.get("facts")
    if kind not in CORRUPTION_TYPES or not isinstance(facts, dict):
        raise ValueError(f"repair validation metadata is incomplete: {material_id}")
    source_row_sha256 = corruption.get("source_row_sha256")
    if not isinstance(source_row_sha256, str) or len(source_row_sha256) != 64:
        raise ValueError(f"repair source row hash is missing: {material_id}")
    requested_sg = facts.get("sg")
    if type(requested_sg) is not int:
        raise ValueError(f"target audit has no integer space group: {material_id}")
    return {
        "material_id": material_id,
        "corruption_type": kind,
        "plan_index": None,
        "seed": None,
        "prompt_sha256": _sha256_text(prefix),
        "input_crystal_sha256": _sha256_text(body),
        "target_body_sha256": _sha256_text(target),
        "source_row_sha256": source_row_sha256,
        "requested_sg": requested_sg,
    }


def _material_seed(material_id, seed=SEED):
    digest = hashlib.sha256(f"{seed}:{material_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _build_selection_manifest(val_rows, train_ids, data_hashes):
    train_ids = list(train_ids)
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("training data contains duplicate material IDs")
    identities = [_row_identity(row) for row in val_rows]
    val_ids = [item["material_id"] for item in identities]
    if len(val_ids) != len(set(val_ids)):
        raise ValueError("validation data contains duplicate material IDs")
    if set(train_ids) & set(val_ids):
        raise ValueError("selected validation IDs overlap the training split")

    buckets = {kind: [] for kind in CORRUPTION_TYPES}
    for row, identity in zip(val_rows, identities):
        buckets[identity["corruption_type"]].append((row, identity))
    counts = {kind: len(buckets[kind]) for kind in CORRUPTION_TYPES}
    insufficient = {kind: count for kind, count in counts.items() if count < ITEMS_PER_TYPE}
    if insufficient:
        raise ValueError(f"need eight actual validation examples per corruption type; found {counts}")

    chosen = []
    for kind in CORRUPTION_TYPES:
        ordered = sorted(
            buckets[kind],
            key=lambda pair: (hashlib.sha256(
                f"{SEED}:{kind}:{pair[1]['material_id']}".encode("utf-8")).digest(),
                pair[1]["material_id"]),
        )
        chosen.extend(identity for _, identity in ordered[:ITEMS_PER_TYPE])
    items = []
    for plan_index, identity in enumerate(chosen):
        items.append({**identity, "plan_index": plan_index,
                      "seed": _material_seed(identity["material_id"])})
    return {
        "schema": 1,
        "seed": SEED,
        "selection": "lowest SHA256(seed:type:material_id), eight per actual corruption type",
        "corruption_types": list(CORRUPTION_TYPES),
        "items_per_type": ITEMS_PER_TYPE,
        "available_validation_counts": counts,
        "data_hashes": data_hashes,
        "items": items,
    }


def _evaluation_item(row, selected):
    prompt = row["taskR"]["prefix"]
    if _sha256_text(prompt) != selected["prompt_sha256"]:
        raise ValueError(f"selected prompt changed: {selected['material_id']}")
    return {
        "prompt": prompt,
        "space_group_number": selected["requested_sg"],
        "request_id": selected["material_id"],
        "attempt_id": 0,
        "plan_index": selected["plan_index"],
        "flat_index": selected["plan_index"],
        "seed": selected["seed"],
    }


def _validate_checkpoint_dataset(metadata, data_hashes, train_ids, val_ids, selected_ids):
    checkpoint_hashes = metadata.get("dataset_hashes")
    if (not isinstance(checkpoint_hashes, dict)
            or any(checkpoint_hashes.get(key) != data_hashes.get(key)
                   for key in DATASET_HASH_KEYS)):
        raise ValueError("repair checkpoint dataset hashes differ from evaluation data")

    def id_set(key):
        values = metadata.get(key)
        if (not isinstance(values, list) or not all(isinstance(value, str) for value in values)
                or len(values) != len(set(values))):
            raise ValueError(f"repair checkpoint {key} are invalid")
        return set(values)

    checkpoint_train_ids = id_set("train_material_ids")
    checkpoint_val_ids = id_set("validation_material_ids")
    if (checkpoint_train_ids != set(train_ids) or checkpoint_val_ids != set(val_ids)
            or checkpoint_train_ids & checkpoint_val_ids):
        raise ValueError("repair checkpoint split IDs differ from evaluation data")
    selected_ids = set(selected_ids)
    if selected_ids & checkpoint_train_ids:
        raise ValueError("selected validation IDs overlap repair checkpoint training IDs")
    if not selected_ids <= checkpoint_val_ids:
        raise ValueError("selected IDs are absent from repair checkpoint validation IDs")


def _evaluation_sources(trainer, loop):
    paths = {
        "evaluate.py": Path(__file__).resolve(),
        "repair_trainer_run.py": TRAINER_PATH,
        "repair_trainer_model.py": trainer.MODEL_PATH,
        "repair_prepare.py": EXPERIMENT / "prepare.py",
        "parent_trainer_run.py": trainer.FULL_TRAINER_PATH,
        "parent_trainer_model.py": trainer.FULL_MODEL_PATH,
        "feedback_loop_run.py": LOOP_PATH,
        "legacy_evaluate.py": Path(loop.E.__file__),
        "structure_conditioned_evaluate.py": Path(loop.E.AR_EVALUATOR.__file__),
        "structure_request_runner.py": Path(loop.RUNNER.__file__),
        "strict_structure_score.py": Path(loop.E.REFERENCE_SCORE_PATH),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _project_path(path, name, strict=False):
    path = Path(path).expanduser().resolve(strict=strict)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"{name} must stay under project root: {path}")
    return path


def _arm_manifest(arm, path, checkpoint_hash, state_hash, source_hash, identity,
                  code_hashes, selection_hash, data_hashes, num_shards, software):
    return {
        "schema": 1,
        "arm": arm,
        "checkpoint_path": str(path.relative_to(ROOT)),
        "checkpoint_fingerprint_sha256": checkpoint_hash,
        "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
        "source_trainable_state_sha256": source_hash,
        "trainable_state_sha256": state_hash,
        "evaluation_source_sha256": code_hashes,
        "selection_manifest_sha256": selection_hash,
        "data_hashes": data_hashes,
        "seed": SEED,
        "num_shards": num_shards,
        "sampling": {
            "lead": "natural stochastic top-p AR; START is not forced",
            "crystal": "B=4 FLARE shifted-block denoising",
            "temperature": 0.8,
            "top_p": 0.95,
            "max_lead_tokens": 256,
            "crystal_capacity_including_END": 544,
            "max_context": 4096,
        },
        "software": software,
        "property_evaluation_status": "not_evaluated",
        "readback_status": "not_run",
    }


def _validate_receipt(receipt, arm, item, selected, manifest):
    expected = {
        "status": "complete",
        "arm": arm,
        "request_id": item["request_id"],
        "attempt_id": 0,
        "plan_index": item["plan_index"],
        "flat_index": item["flat_index"],
        "seed": item["seed"],
        "policy": arm,
        "checkpoint_fingerprint_sha256": manifest["checkpoint_fingerprint_sha256"],
        "source_checkpoint_identity_sha256": manifest["source_checkpoint_identity_sha256"],
        "source_trainable_state_sha256": manifest["source_trainable_state_sha256"],
        "trainable_state_sha256": manifest["trainable_state_sha256"],
        "evaluation_source_sha256": manifest["evaluation_source_sha256"],
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "data_hashes": manifest["data_hashes"],
        "corruption_type": selected["corruption_type"],
        "source_row_sha256": selected["source_row_sha256"],
        "input_crystal_sha256": selected["input_crystal_sha256"],
        "target_body_sha256": selected["target_body_sha256"],
        "requested_sg": selected["requested_sg"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"existing {arm} receipt identity differs: {item['request_id']}")
    prompt = receipt.get("prompt")
    if not isinstance(prompt, str) or _sha256_text(prompt) != selected["prompt_sha256"]:
        raise ValueError(f"existing {arm} receipt prompt differs: {item['request_id']}")
    if "<CRYSTAL_START>" in prompt or "<CRYSTAL_END>" in prompt:
        raise ValueError(f"existing {arm} receipt contains a forced boundary: {item['request_id']}")
    if _sha256_text(_input_crystal_body(prompt)) != selected["input_crystal_sha256"]:
        raise ValueError(f"existing {arm} receipt corrupted input differs: {item['request_id']}")
    if not isinstance(receipt.get("failure_categories"), list):
        raise ValueError(f"existing {arm} receipt has no failure categories: {item['request_id']}")
    if receipt.get("diagnostics", {}).get("natural_start_emitted") is not (
            receipt.get("lead_status") == "ok"):
        raise ValueError(f"existing {arm} receipt has inconsistent START status: {item['request_id']}")
    if not isinstance(receipt.get("timing_seconds", {}).get("total"), (int, float)):
        raise ValueError(f"existing {arm} receipt has no total timing: {item['request_id']}")


def _sample_arm(arm, shard_index, num_shards, rows_by_id, selection, selection_hash,
                data_hashes, out, model, tok, train, score, identity, source_hash,
                manifest, loop, resume):
    shard_dir = out / arm / f"shard-{shard_index:02d}-of-{num_shards:02d}"
    for selected in selection["items"]:
        if selected["plan_index"] % num_shards != shard_index:
            continue
        row = rows_by_id[selected["material_id"]]
        item = _evaluation_item(row, selected)
        path = shard_dir / "attempts" / f"{item['request_id']}.json"
        if path.exists():
            if not resume:
                raise FileExistsError(f"{path} exists; pass --resume")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            _validate_receipt(receipt, arm, item, selected, manifest)
            continue

        context_ids = tok.encode(item["prompt"], add_special_tokens=False)
        if not context_ids:
            raise ValueError(f"repair prompt tokenized to an empty context: {item['request_id']}")
        cif_root = shard_dir / "cifs" / item["request_id"]
        before = train.trainable_sha256(model)
        if before != manifest["trainable_state_sha256"]:
            raise RuntimeError(f"{arm} state changed before generation: {item['request_id']}")
        generation = loop._candidate_attempt(
            item, arm, 0, item["seed"], context_ids, None, False,
            model, tok, train, score, identity, manifest["checkpoint_fingerprint_sha256"],
            before, manifest["evaluation_source_sha256"], cif_root)
        if train.trainable_sha256(model) != before:
            raise RuntimeError(f"{arm} state changed during generation: {item['request_id']}")

        body = None
        if generation.get("crystal_raw_token_ids"):
            try:
                body = loop.RUNNER._boundary_content(
                    generation["crystal_raw_token_ids"], model, tok)
            except loop.RUNNER.GenerationFailure:
                pass
        diagnostics = {
            "natural_start_emitted": generation.get("lead_status") == "ok",
            "outer_whitespace_normalized_input_body_match": (
                _outer_whitespace_normalized_body_match(
                    body, _input_crystal_body(item["prompt"])) if body is not None else None),
            "outer_whitespace_normalized_target_body_match": (
                _outer_whitespace_normalized_body_match(
                    body, row["taskR"]["crystal"]) if body is not None else None),
            "generated_body_sha256": _sha256_text(body) if body is not None else None,
        }
        receipt = {
            **generation,
            "schema": 1,
            "status": "complete",
            "arm": arm,
            "request_id": item["request_id"],
            "attempt_id": 0,
            "plan_index": item["plan_index"],
            "flat_index": item["flat_index"],
            "seed": item["seed"],
            "checkpoint_fingerprint_sha256": manifest["checkpoint_fingerprint_sha256"],
            "source_checkpoint_identity_sha256": manifest["source_checkpoint_identity_sha256"],
            "source_trainable_state_sha256": source_hash,
            "trainable_state_sha256": before,
            "evaluation_source_sha256": manifest["evaluation_source_sha256"],
            "selection_manifest_sha256": selection_hash,
            "data_hashes": data_hashes,
            "corruption_type": selected["corruption_type"],
            "source_row_sha256": selected["source_row_sha256"],
            "input_crystal_sha256": selected["input_crystal_sha256"],
            "target_body_sha256": selected["target_body_sha256"],
            "requested_sg": selected["requested_sg"],
            "target_facts": row["targetaudit"]["facts"],
            "diagnostics": diagnostics,
            "ehull_evaluation_status": "not_evaluated",
            "readback_status": "not_run",
        }
        _validate_receipt(receipt, arm, item, selected, manifest)
        _write_json_atomic(path, receipt)


def _metric(records):
    categories = Counter(category for receipt in records
                         for category in receipt.get("failure_categories", []))
    seconds = [float(receipt["timing_seconds"]["total"]) for receipt in records]
    structural = sum(receipt.get("structural_valid") is True for receipt in records)
    requested_sg = sum(receipt.get("requested_sg_match") is True for receipt in records)
    valid_and_sg = sum(receipt.get("structural_valid") is True
                       and receipt.get("requested_sg_match") is True for receipt in records)
    natural_start = sum(receipt["diagnostics"].get("natural_start_emitted") is True
                        for receipt in records)
    copied_input = sum(receipt["diagnostics"].get(
        "outer_whitespace_normalized_input_body_match") is True for receipt in records)
    target_match = sum(receipt["diagnostics"].get(
        "outer_whitespace_normalized_target_body_match") is True for receipt in records)
    return {
        "denominator": len(records),
        "strict_valid": structural,
        "requested_space_group_match": requested_sg,
        "strict_valid_and_requested_space_group_match": valid_and_sg,
        "natural_start_emitted": natural_start,
        "outer_whitespace_normalized_input_body_matches": copied_input,
        "outer_whitespace_normalized_target_body_matches": target_match,
        "mean_total_seconds": statistics.fmean(seconds),
        "failure_category_counts": dict(sorted(categories.items())),
    }


def summarize(out):
    out = Path(out).expanduser().resolve(strict=True)
    run_manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    selection_hash = _json_sha256(selection)
    if (run_manifest.get("selection_manifest_sha256") != selection_hash
            or selection.get("data_hashes") != run_manifest.get("data_hashes")
            or selection.get("seed") != SEED or run_manifest.get("seed") != SEED
            or run_manifest.get("num_shards") not in (1, 3)):
        raise ValueError("selection manifest hash differs from the run manifest")
    items = selection["items"]
    expected_ids = [item["material_id"] for item in items]
    if (len(items) != 32 or len(set(expected_ids)) != 32
            or Counter(item["corruption_type"] for item in items)
            != Counter({kind: ITEMS_PER_TYPE for kind in CORRUPTION_TYPES})):
        raise ValueError("frozen selection must contain 32 unique IDs balanced across four types")

    records = {arm: {} for arm in ARMS}
    for arm in ARMS:
        manifest = json.loads((out / arm / "manifest.json").read_text(encoding="utf-8"))
        if manifest != run_manifest["arms"][arm]:
            raise ValueError(f"{arm} arm manifest differs from the run manifest")
        expected_paths = {
            (out / arm / f"shard-{item['plan_index'] % run_manifest['num_shards']:02d}-of-"
             f"{run_manifest['num_shards']:02d}" / "attempts" / f"{item['material_id']}.json").resolve()
            for item in items
        }
        actual_paths = {
            path.resolve() for path in (out / arm).glob("shard-*-of-*/attempts/*.json")
        }
        if actual_paths != expected_paths:
            raise ValueError(f"{arm} receipts are incomplete, duplicated, or outside their assigned shards")
        for selected in items:
            shard_index = selected["plan_index"] % run_manifest["num_shards"]
            path = (out / arm / f"shard-{shard_index:02d}-of-{run_manifest['num_shards']:02d}"
                    / "attempts" / f"{selected['material_id']}.json")
            if not path.is_file():
                raise ValueError(f"incomplete evaluation: missing {arm} receipt for {selected['material_id']}")
            receipt = json.loads(path.read_text(encoding="utf-8"))
            item = {"request_id": selected["material_id"], "attempt_id": 0,
                    "plan_index": selected["plan_index"], "flat_index": selected["plan_index"],
                    "seed": selected["seed"]}
            _validate_receipt(receipt, arm, item, selected, manifest)
            if selected["material_id"] in records[arm]:
                raise ValueError(f"duplicate {arm} receipt for {selected['material_id']}")
            records[arm][selected["material_id"]] = receipt

    if set(records["parent"]) != set(expected_ids) or set(records["new"]) != set(expected_ids):
        raise ValueError("parent and new arms do not contain the same 32 selected IDs")
    for material_id in expected_ids:
        parent, new = records["parent"][material_id], records["new"][material_id]
        for field in ("seed", "plan_index", "prompt", "context_token_ids", "source_row_sha256",
                      "input_crystal_sha256", "target_body_sha256", "requested_sg"):
            if parent.get(field) != new.get(field):
                raise ValueError(f"paired parent/new input differs for {material_id}: {field}")

    summaries = {}
    per_type = {}
    for arm in ARMS:
        ordered = [records[arm][material_id] for material_id in expected_ids]
        summaries[arm] = _metric(ordered)
        per_type[arm] = {
            kind: _metric([receipt for receipt in ordered if receipt["corruption_type"] == kind])
            for kind in CORRUPTION_TYPES
        }
    paired = Counter()
    for material_id in expected_ids:
        parent = records["parent"][material_id]
        new = records["new"][material_id]
        parent_success = parent.get("structural_valid") is True and parent.get("requested_sg_match") is True
        new_success = new.get("structural_valid") is True and new.get("requested_sg_match") is True
        paired[(parent_success, new_success)] += 1
    result = {
        "schema": 1,
        "status": "complete",
        "selection_manifest_sha256": selection_hash,
        "data_hashes": selection["data_hashes"],
        "num_shards": run_manifest["num_shards"],
        "selected_ids": expected_ids,
        "selected_counts_by_corruption_type": {
            kind: sum(item["corruption_type"] == kind for item in items)
            for kind in CORRUPTION_TYPES
        },
        "arms": summaries,
        "per_corruption_type": per_type,
        "paired_success": {
            "both": paired[(True, True)],
            "parent_only": paired[(True, False)],
            "new_only": paired[(False, True)],
            "neither": paired[(False, False)],
        },
        "ehull_evaluation_status": "not_evaluated",
        "readback_status": "not_run",
    }
    _write_json_atomic(out / "summary.json", result)
    (out / "RESULTS.md").write_text(_results_markdown(result), encoding="utf-8")
    return result


def _results_markdown(result):
    lines = [
        "# Parent versus repair-trained crystal generation",
        "",
        "The comparison uses the same 32 validation repair prompts, with eight examples per actual corruption type. Both arms receive the corrupted input in the repair prefix; the reference lead and target crystal are excluded from model context. START is sampled naturally. Success requires strict structural validity and the requested space group. ehull was not evaluated; factual readback was not run.",
        "",
        "| Arm | Strict valid | Requested SG match | Valid and SG match | Natural START | Whitespace-normalized input-body match | Whitespace-normalized target-body match | Mean seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metric = result["arms"][arm]
        lines.append(
            f"| {arm} | {metric['strict_valid']}/{metric['denominator']} | "
            f"{metric['requested_space_group_match']}/{metric['denominator']} | "
            f"{metric['strict_valid_and_requested_space_group_match']}/{metric['denominator']} | "
            f"{metric['natural_start_emitted']}/{metric['denominator']} | "
            f"{metric['outer_whitespace_normalized_input_body_matches']}/{metric['denominator']} | "
            f"{metric['outer_whitespace_normalized_target_body_matches']}/{metric['denominator']} | "
            f"{metric['mean_total_seconds']:.2f} |"
        )
    paired = result["paired_success"]
    lines += [
        "",
        f"Paired structural-and-SG outcomes: both {paired['both']}, parent only {paired['parent_only']}, new only {paired['new_only']}, neither {paired['neither']}",
        "",
        "Failure categories are counted from each saved attempt receipt:",
        "",
    ]
    for arm in ARMS:
        lines.append(f"- {arm}: `{json.dumps(result['arms'][arm]['failure_category_counts'], sort_keys=True)}`")
    lines += ["", "Per-corruption-type strict-valid and valid-plus-SG counts:", "",
              "| Corruption type | Parent valid | Parent valid+SG | New valid | New valid+SG |",
              "|---|---:|---:|---:|---:|"]
    for kind in CORRUPTION_TYPES:
        parent, new = result["per_corruption_type"]["parent"][kind], result["per_corruption_type"]["new"][kind]
        lines.append(
            f"| {kind} | {parent['strict_valid']}/8 | "
            f"{parent['strict_valid_and_requested_space_group_match']}/8 | "
            f"{new['strict_valid']}/8 | {new['strict_valid_and_requested_space_group_match']}/8 |"
        )
    lines.append("")
    return "\n".join(lines)


def run_sample(args):
    if args.data is None or args.run_dir is None or args.gpu is None:
        raise ValueError("sample mode requires --data, --run-dir, and --gpu")
    if args.num_shards not in (1, 3) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("use one or three shards and a shard index within that range")
    if type(args.gpu) is not int or not 0 <= args.gpu <= 3:
        raise ValueError("--gpu must name one physical GPU in 0..3")
    data_dir = _project_path(args.data, "data", strict=True)
    run_dir = _project_path(args.run_dir, "run directory", strict=True)
    out = _project_path(args.out, "output directory")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    trainer = _load_module("crystal_repair_evaluation_trainer", TRAINER_PATH)
    loop = _load_module("crystal_repair_evaluation_feedback_loop", LOOP_PATH)
    data_hashes, train_rows, val_rows, _data_manifest = trainer.read_data(data_dir)
    train_ids = [row["material_id"] for row in train_rows]
    selection = _build_selection_manifest(val_rows, train_ids, data_hashes)
    selection_hash = _json_sha256(selection)
    _ensure_shared_json(out / "selection.json", selection)

    train, model, tok, identity, parent_hash, software = trainer.load_model()
    torch = train.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each evaluation worker must expose exactly its selected GPU")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    model.eval()
    parent_checkpoint = _project_path(trainer.PARENT_CHECKPOINT, "parent checkpoint", strict=True)
    child_checkpoint = _project_path(run_dir / "checkpoint.pt", "repair checkpoint", strict=True)
    parent_checkpoint_hash = _sha256_file(parent_checkpoint)
    child_checkpoint_hash = _sha256_file(child_checkpoint)
    if train.trainable_sha256(model) != parent_hash:
        raise ValueError("parent FLARE state hash differs from the repair trainer loader")
    code_hashes = _evaluation_sources(trainer, loop)
    score = loop.RUNNER._load_score_module()

    parent_manifest = _arm_manifest(
        "parent", parent_checkpoint, parent_checkpoint_hash, parent_hash, parent_hash,
        identity, code_hashes, selection_hash, data_hashes, args.num_shards, software)
    child_saved = torch.load(child_checkpoint, map_location="cpu", weights_only=False)
    trainer.MODEL.validate_checkpoint_identity(
        child_saved, "repair", parent_checkpoint_hash, parent_hash)
    child_run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if (child_run_manifest.get("schema") != 1 or child_run_manifest.get("arm") != "repair"
            or child_run_manifest.get("status") != "complete"
            or child_run_manifest.get("metadata") != child_saved.get("metadata")):
        raise ValueError("repair run manifest and completed checkpoint differ")
    _validate_checkpoint_dataset(
        child_saved["metadata"], data_hashes, train_ids,
        [row["material_id"] for row in val_rows],
        [item["material_id"] for item in selection["items"]])
    child_hash = train.state_sha256(child_saved["trainable_state"])
    if child_saved.get("trainable_state_sha256") != child_hash:
        raise ValueError("repair checkpoint trainable tensors failed their hash check")
    del child_saved
    child_manifest = _arm_manifest(
        "new", child_checkpoint, child_checkpoint_hash, child_hash, parent_hash,
        identity, code_hashes, selection_hash, data_hashes, args.num_shards, software)
    run_manifest = {
        "schema": 1,
        "experiment": EXPERIMENT.name,
        "data_directory": str(data_dir.relative_to(ROOT)),
        "run_directory": str(run_dir.relative_to(ROOT)),
        "selection_manifest_sha256": selection_hash,
        "data_hashes": data_hashes,
        "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
        "num_shards": args.num_shards,
        "seed": SEED,
        "evaluation_source_sha256": code_hashes,
        "arms": {"parent": parent_manifest, "new": child_manifest},
    }
    _ensure_shared_json(out / "run_manifest.json", run_manifest)
    for arm, manifest in (("parent", parent_manifest), ("new", child_manifest)):
        _ensure_shared_json(out / arm / "manifest.json", manifest)

    rows_by_id = {row["material_id"]: row for row in val_rows}
    if set(item["material_id"] for item in selection["items"]) - rows_by_id.keys():
        raise ValueError("selection contains an ID missing from validation data")
    # Generate the frozen parent stage before restoring the repair-trained state.
    if train.trainable_sha256(model) != parent_hash:
        raise RuntimeError("parent state changed before generation")
    _sample_arm("parent", args.shard_index, args.num_shards, rows_by_id, selection,
                selection_hash, data_hashes, out, model, tok, train, score, identity,
                parent_hash, parent_manifest, loop, args.resume)
    if train.trainable_sha256(model) != parent_hash:
        raise RuntimeError("parent state changed during generation")

    # The same model object is updated in place to avoid keeping a second 8B model copy.
    trainer.load_checkpoint(
        model, child_checkpoint, train, "repair",
        parent_checkpoint_sha256=parent_checkpoint_hash,
        parent_trainable_sha256=parent_hash)
    model.eval()
    if train.trainable_sha256(model) != child_hash:
        raise RuntimeError("repair state changed while restoring the child checkpoint")
    _sample_arm("new", args.shard_index, args.num_shards, rows_by_id, selection,
                selection_hash, data_hashes, out, model, tok, train, score, identity,
                parent_hash, child_manifest, loop, args.resume)
    if train.trainable_sha256(model) != child_hash:
        raise RuntimeError("repair-trained state changed during generation")
    if (_sha256_file(parent_checkpoint) != parent_checkpoint_hash
            or _sha256_file(child_checkpoint) != child_checkpoint_hash):
        raise RuntimeError("a frozen checkpoint file changed during generation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "summarize"), default="sample")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, choices=(1, 3), default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "summarize":
        result = summarize(args.out)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        run_sample(args)


if __name__ == "__main__":
    main()
