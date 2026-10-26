"""Train the shared AR/diffusion model on audited crystal-repair requests."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time
from types import MethodType


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
FULL_TRAINER_PATH = ROOT / "experiments/flare_crystal_full_20261019/run.py"
FULL_MODEL_PATH = ROOT / "experiments/flare_crystal_full_20261019/model.py"
PARENT_CHECKPOINT = ROOT / "experiments/flare_crystal_full_20261019/flare/checkpoint.pt"
PARENT_MANIFEST = PARENT_CHECKPOINT.parent / "manifest.json"
PILOT_RUN_PATH = ROOT / "experiments/flare_crystal_qwen3_8b_20261019/run.py"
PILOT_MODEL_PATH = ROOT / "experiments/flare_crystal_qwen3_8b_20261019/model.py"
MODEL_PATH = EXPERIMENT / "model.py"
SEED = 42
GLOBAL_MATERIALS_PER_UPDATE = 6
RANKS = (1, 3)
LOCAL_MATERIALS_PER_RANK = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
CHECKPOINT_INTERVAL = 50
START, END = "<CRYSTAL_START>", "<CRYSTAL_END>"
STAT_KEYS = (
    "repair_joint", "generation_joint", "reading_ar", "repair_clean_ar",
    "repair_denoise", "generation_clean_ar", "generation_denoise",
)

CONFIG = {
    "experiment": EXPERIMENT.name,
    "seed": SEED,
    "order": "random.Random(42).shuffle(train_rows)",
    "materials_per_update": GLOBAL_MATERIALS_PER_UPDATE,
    "objective": "0.5 R_FLARE + 0.25 G_FLARE + 0.25 Q_AR; R and G each use clean AR plus complementary crystal denoising",
    "optimizer": "fresh AdamW, constant learning rate",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_norm": GRAD_CLIP_NORM,
    "precision": "BF16 autocast; parent FLARE trainable LoRA and control-token parameters",
    "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
    "dummy_slots": "repeat train row 0 with zero weight to keep equal DDP microsteps",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FULLTRAINER = _load_module("repair_parent_fulltrainer", FULL_TRAINER_PATH)
MODEL = _load_module("crystal_repair_training_model", MODEL_PATH)
load_checkpoint = MODEL.load_checkpoint
_write_json = FULLTRAINER._write_json
_append_jsonl = FULLTRAINER._append_jsonl
_atomic_torch_save = FULLTRAINER._atomic_torch_save


def isolate_rank_gpu(environ=None):
    """Limit each process to its selected GPU before importing torch."""
    environ = os.environ if environ is None else environ
    visible = [device.strip() for device in environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if device.strip()]
    rank = int(environ.get("RANK", "0"))
    world_size = int(environ.get("WORLD_SIZE", "1"))
    local_rank = int(environ.get("LOCAL_RANK", "0"))
    if (not visible or world_size not in RANKS or len(visible) != world_size
            or not 0 <= rank < world_size or not 0 <= local_rank < world_size):
        raise ValueError("set CUDA_VISIBLE_DEVICES to one or three selected GPUs matching the launcher")
    environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank]
    return rank, world_size, local_rank


def total_updates(material_count):
    if material_count < 1:
        raise ValueError("training data must contain at least one material")
    return math.ceil(material_count / GLOBAL_MATERIALS_PER_UPDATE)


def partition_update(ordered_rows, update, rank, world_size):
    if world_size not in RANKS or world_size > GLOBAL_MATERIALS_PER_UPDATE:
        raise ValueError("training supports one or three ranks")
    if not ordered_rows or not 1 <= update <= total_updates(len(ordered_rows)):
        raise ValueError("update is outside the nonempty training plan")
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside the training world")
    local_size = GLOBAL_MATERIALS_PER_UPDATE // world_size
    start = (update - 1) * GLOBAL_MATERIALS_PER_UPDATE + rank * local_size
    return [(index, ordered_rows[index] if index < len(ordered_rows) else ordered_rows[0])
            for index in range(start, start + local_size)]


def actual_count(ordered_rows, update):
    if not 1 <= update <= total_updates(len(ordered_rows)):
        raise ValueError("update is outside the nonempty training plan")
    return min(GLOBAL_MATERIALS_PER_UPDATE,
               len(ordered_rows) - (update - 1) * GLOBAL_MATERIALS_PER_UPDATE)


def gradient_scale(world_size, actual_count):
    if world_size not in RANKS or not 1 <= actual_count <= GLOBAL_MATERIALS_PER_UPDATE:
        raise ValueError("invalid world size or actual material count")
    return world_size / actual_count


def needs_final_checkpoint(target, completed, total):
    if target < completed:
        raise ValueError("max-updates precedes the resumed update")
    if target == completed:
        if completed != total:
            raise ValueError("max-updates must exceed the resumed update")
        return True
    return False


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_example(row, example, name):
    if not isinstance(example, dict) or not isinstance(example.get("prefix"), str):
        raise ValueError(f"{name} requires a string prefix")
    if name in ("R", "G"):
        if not all(isinstance(example.get(key), str) for key in ("lead", "crystal")):
            raise ValueError(f"{name} requires string lead and crystal fields")
        if not example["lead"].endswith(START):
            raise ValueError(f"{name} lead must end in CRYSTAL_START for {row['material_id']}")
        if name == "R" and any(tag in example["prefix"] for tag in (START, END)):
            raise ValueError(f"repair prefix must keep corrupted PLaID as plain text for {row['material_id']}")
    elif not isinstance(example.get("target"), str):
        raise ValueError(f"Q requires a string reading target for {row['material_id']}")


def _read_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            material_id = row.get("material_id")
            if not isinstance(material_id, str) or not material_id:
                raise ValueError(f"{path}:{line_number} has no material_id")
            _validate_example(row, row.get("taskR"), "R")
            original = row.get("original_tasks")
            if (not isinstance(original, list) or len(original) != 2
                    or not all(isinstance(item, dict) for item in original)
                    or [item.get("task") for item in original] != ["G", "Q"]):
                raise ValueError(f"{path}:{line_number} requires original_tasks [G, Q]")
            if (not isinstance(row.get("targetaudit"), dict)
                    or row["targetaudit"].get("valid") is not True
                    or not isinstance(row.get("corrupted_audit"), dict)
                    or row["corrupted_audit"].get("valid") is not False):
                raise ValueError(f"{path}:{line_number} requires valid target and failed-corruption audits")
            if not isinstance(row.get("corruption"), dict):
                raise ValueError(f"{path}:{line_number} requires a corruption record")
            _validate_example(row, original[0], "G")
            _validate_example(row, original[1], "Q")
            rows.append(row)
    return rows


def read_data(data_dir):
    data_dir = Path(data_dir).expanduser().resolve(strict=True)
    train_path, val_path = data_dir / "train.jsonl", data_dir / "val.jsonl"
    train_rows, val_rows = _read_rows(train_path), _read_rows(val_path)
    train_ids = [row["material_id"] for row in train_rows]
    val_ids = [row["material_id"] for row in val_rows]
    if not train_rows or not val_rows:
        raise ValueError("repair train and validation files must both be nonempty")
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("repair data contains duplicate material IDs")
    if set(train_ids) & set(val_ids):
        raise ValueError("repair train and validation material IDs overlap")
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("repair data manifest schema differs")
    hashes = {"train_sha256": _sha256_file(train_path), "val_sha256": _sha256_file(val_path)}
    artifacts = manifest.get("artifacts", {})
    for name, actual in (("train.jsonl", hashes["train_sha256"]),
                         ("val.jsonl", hashes["val_sha256"])):
        artifact = artifacts.get(name)
        if not isinstance(artifact, dict) or artifact.get("sha256") != actual:
            raise ValueError(f"repair data artifact hash differs: {name}")
    output_counts = manifest.get("counts", {}).get("output", {})
    if output_counts != {"train": len(train_rows), "val": len(val_rows)}:
        raise ValueError("repair data manifest row counts differ")
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("repair data manifest source record is invalid")
    source_manifest_hash = (manifest.get("source_manifest_sha256")
                            or source.get("immutable_manifest_sha256")
                            or source.get("manifest_sha256"))
    if not isinstance(source_manifest_hash, str) or not source_manifest_hash:
        raise ValueError("repair data manifest is missing its frozen source manifest hash")
    hashes.update(manifest_sha256=_sha256_file(manifest_path),
                  source_manifest_sha256=source_manifest_hash)
    return hashes, train_rows, val_rows, manifest


def validate_source_alignment(repair_train, repair_val, source_train, source_val, task_builder):
    """Bind eligible repair rows to their frozen split, original tasks, and target crystal."""
    for split, repair_rows, source_rows in (("train", repair_train, source_train),
                                             ("val", repair_val, source_val)):
        source_by_id = {row["material_id"]: row for row in source_rows}
        if len(source_by_id) != len(source_rows):
            raise ValueError(f"frozen source {split} split contains duplicate IDs")
        for row in repair_rows:
            material_id = row["material_id"]
            source = source_by_id.get(material_id)
            if source is None:
                raise ValueError(f"repair row {material_id} is outside the frozen {split} split")
            original = task_builder(source)
            corruption = row.get("corruption", {})
            if row["original_tasks"] != original:
                raise ValueError(f"original G/Q tasks differ from the frozen source for {material_id}")
            if row["taskR"]["crystal"] != original[0]["crystal"]:
                raise ValueError(f"repair target differs from the original crystal for {material_id}")
            if corruption.get("source_row_sha256") != _json_sha256(source):
                raise ValueError(f"repair source row hash differs for {material_id}")
            target_hash = hashlib.sha256(source["plaid_text"].encode()).hexdigest()
            if corruption.get("target_crystal_sha256") != target_hash:
                raise ValueError(f"repair target crystal hash differs for {material_id}")
            request = original[0]["prefix"].removeprefix("User: ").removesuffix("\nAssistant:")
            required_prefix = f"User: {request}\nPrevious failed crystal (input data):\n"
            if not row["taskR"]["prefix"].startswith(required_prefix):
                raise ValueError(f"repair prompt does not retain the frozen request for {material_id}")


def read_raw_source_splits(data_dir, verified_hashes):
    """Read canonical source rows after the frozen loader has checked their file hashes."""
    result = {}
    for split in ("train", "val"):
        path = Path(data_dir) / f"{split}-triples.jsonl.gz"
        if _sha256_file(path) != verified_hashes[f"{split}_sha256"]:
            raise ValueError(f"frozen source {split} file changed after verification")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"frozen source row has the wrong {split} label")
        if len({row["material_id"] for row in rows}) != len(rows):
            raise ValueError(f"frozen source {split} split contains duplicate IDs")
        result[split] = rows
    return result["train"], result["val"]


def _metadata(identity, pure_ar_hash, parent_hash, parent_file_hash, parent_metadata_hash,
              versions, data_hashes, ordered_rows, val_rows, world_size):
    train_ids = [row["material_id"] for row in ordered_rows]
    val_ids = [row["material_id"] for row in val_rows]
    if set(train_ids) & set(val_ids):
        raise ValueError("repair train and validation material IDs overlap")
    config = {
        **CONFIG,
        "train_materials": len(train_ids),
        "validation_materials": len(val_ids),
        "total_updates": total_updates(len(train_ids)),
        "arm": "repair",
    }
    code_paths = {
        "run.py": Path(__file__).resolve(),
        "model.py": MODEL_PATH,
        "fulltrainer_run.py": FULL_TRAINER_PATH,
        "fulltrainer_model.py": FULL_MODEL_PATH,
        "pilot_run.py": PILOT_RUN_PATH,
        "pilot_model.py": PILOT_MODEL_PATH,
    }
    return {
        "config": config,
        "config_sha256": _json_sha256(config),
        "arm": "repair",
        "training_world_size": world_size,
        "source_identity": identity,
        "pure_ar_trainable_sha256": pure_ar_hash,
        "parent_trainable_sha256": parent_hash,
        "parent_checkpoint_sha256": parent_file_hash,
        "parent_checkpoint_metadata_sha256": parent_metadata_hash,
        "parent_checkpoint_path": str(PARENT_CHECKPOINT.relative_to(ROOT)),
        "runtime_versions": versions,
        "dataset_hashes": data_hashes,
        "train_material_ids": train_ids,
        "validation_material_ids": val_ids,
        "epoch_order_sha256": _json_sha256(train_ids),
        "validation_ids_sha256": _json_sha256(val_ids),
        "source_sha256": {name: _sha256_file(path) for name, path in code_paths.items()},
    }


def _load_parent_model():
    full = FULLTRAINER
    train, model, tok, identity, pure_ar_hash, versions = full.load_model()
    saved = full.load_checkpoint(model, PARENT_CHECKPOINT, train, expected_arm="flare")
    parent_hash = saved["trainable_state_sha256"]
    if train.trainable_sha256(model) != parent_hash:
        raise ValueError("restored parent FLARE state failed its hash check")
    manifest = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or manifest.get("arm") != "flare"
            or manifest.get("status") != "complete"
            or manifest.get("metadata") != saved.get("metadata")):
        raise ValueError("parent FLARE manifest and completed checkpoint differ")
    return full, train, model, tok, identity, pure_ar_hash, versions, saved, parent_hash


def load_model():
    """Return the six-value parent loader tuple after restoring the completed FLARE state."""
    _full, train, model, tok, identity, _pure_hash, versions, _saved, parent_hash = _load_parent_model()
    return train, model, tok, identity, parent_hash, versions


def _initialize_runtime():
    rank, world_size, _local_rank = isolate_rank_gpu()
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each process must expose exactly its selected CUDA device")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, torch, dist


def _rng_state(torch, generator):
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
            "generator": generator.get_state()}


def _gather_rng_states(torch, dist, generator, world_size):
    state = _rng_state(torch, generator)
    if world_size == 1:
        return [state]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, state)
    return gathered


def _save_checkpoint(paths, model, optimizer, train, metadata, parent_hash,
                     parent_file_hash, completed, total, world_size, rank, generator):
    torch, dist = train.torch, train.dist
    rng_states = _gather_rng_states(torch, dist, generator, world_size)
    if rank == 0:
        state = MODEL.trainable_state(model)
        payload = {
            "schema": 1,
            "arm": "repair",
            "metadata": metadata,
            "metadata_sha256": _json_sha256(metadata),
            "parent_checkpoint_sha256": parent_file_hash,
            "parent_trainable_sha256": parent_hash,
            "completed_updates": completed,
            "total_updates": total,
            "next_material_index": min(completed * GLOBAL_MATERIALS_PER_UPDATE,
                                        metadata["config"]["train_materials"]),
            "world_size": world_size,
            "trainable_parameter_names": sorted(state),
            "trainable_state": state,
            "trainable_state_sha256": train.state_sha256(state),
            "optimizer_state": optimizer.state_dict(),
            "rng_states": rng_states,
        }
        for path in paths:
            _atomic_torch_save(path, payload, torch)
        _write_json(Path(paths[0]).parent / "progress.json", {
            "status": "checkpointed", "arm": "repair", "completed_updates": completed,
            "total_updates": total, "trainable_state_sha256": payload["trainable_state_sha256"],
        })
    if world_size > 1:
        dist.barrier()


def _trim_updates(path, completed):
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    kept = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if record["update"] <= completed:
            kept.append(json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("".join(line + "\n" for line in kept))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _restore_training(path, model, optimizer, train, metadata, parent_hash,
                      parent_file_hash, world_size, rank, generator):
    saved = MODEL.load_checkpoint(
        model, path, train, expected_arm="repair", parent_checkpoint_sha256=parent_file_hash,
        parent_trainable_sha256=parent_hash, metadata=metadata, world_size=world_size,
        allow_partial=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    rng = saved["rng_states"][rank]
    random.setstate(rng["python"])
    train.torch.set_rng_state(rng["torch_cpu"])
    train.torch.cuda.set_rng_state(rng["torch_cuda"], train.torch.cuda.current_device())
    generator.set_state(rng["generator"])
    return saved["completed_updates"]


def _output_path(args):
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(EXPERIMENT.resolve()) or out == EXPERIMENT.resolve():
        raise ValueError(f"output must be a child of {EXPERIMENT}")
    if args.mode == "train":
        if args.resume:
            if not out.is_dir() or not (out / "manifest.json").is_file():
                raise FileNotFoundError("--resume requires an existing training manifest")
        elif out.exists():
            raise FileExistsError(f"refusing to overwrite run output: {out}")
    elif not out.is_dir():
        raise FileNotFoundError("validation requires the training output directory")
    return out


def _mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


def _train_update(train, objective, model, ddp_model, optimizer, ordered_rows,
                  update, rank, world_size, generator):
    torch, dist = train.torch, train.dist
    local_slots = partition_update(ordered_rows, update, rank, world_size)
    count = actual_count(ordered_rows, update)
    scale = gradient_scale(world_size, count)
    optimizer.zero_grad(set_to_none=True)
    local_records = []
    parameters = [param for param in model.parameters() if param.requires_grad]
    for slot, (index, row) in enumerate(local_slots):
        active = index < len(ordered_rows)
        sync = slot == len(local_slots) - 1
        context = ddp_model.no_sync() if world_size > 1 and not sync else nullcontext()
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if world_size > 1:
                    loss, stats = ddp_model(row["taskR"], row["original_tasks"], generator)
                else:
                    loss, stats = objective.objective_tasks(
                        model, train.tokenizer, row["taskR"], row["original_tasks"], generator)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at update {update}")
                (loss * (scale if active else 0.0)).backward()
        if active:
            local_records.append({"index": index, "material_id": row["material_id"],
                                  "loss": float(loss.detach().float()),
                                  **{key: float(stats[key]) for key in STAT_KEYS}})
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM,
                                                error_if_nonfinite=True)
    optimizer.step()
    local = {"records": local_records, "gradient_norm": float(grad_norm.detach().float())}
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return None
    records = sorted((record for part in gathered for record in part["records"]),
                     key=lambda record: record["index"])
    if len(records) != count:
        raise ValueError("distributed update did not cover the expected number of unique materials")
    return {
        "update": update,
        "arm": "repair",
        "material_ids": [record["material_id"] for record in records],
        "actual_materials": count,
        "loss": _mean(records, "loss"),
        **{key: _mean(records, key) for key in STAT_KEYS},
        "gradient_norm": sum(part["gradient_norm"] for part in gathered) / world_size,
    }


def _run_train(args, rank, world_size, train, model, identity, pure_hash, versions,
               parent, parent_hash, data_hashes, ordered_rows, val_rows, out, dist):
    torch = train.torch
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("repair training requires BF16 support on the selected GPU")
    total = total_updates(len(ordered_rows))
    target = total if args.max_updates is None else args.max_updates
    if not 1 <= target <= total:
        raise ValueError(f"max-updates must be in 1..{total}")
    parent_file_hash = _sha256_file(PARENT_CHECKPOINT)
    metadata = _metadata(identity, pure_hash, parent_hash, parent_file_hash,
                         parent["metadata_sha256"], versions, data_hashes,
                         ordered_rows, val_rows, world_size)
    manifest_path = out / "manifest.json"
    if args.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != 1 or manifest.get("arm") != "repair"
                or manifest.get("metadata") != metadata):
            raise ValueError("resume manifest differs in parent, source, data, code, or protocol")
    elif rank == 0:
        out.mkdir(parents=True)
        _write_json(manifest_path, {"schema": 1, "arm": "repair",
                                    "status": "running", "metadata": metadata})
    if world_size > 1:
        dist.barrier()

    objective = _load_module("crystal_repair_training_objective", MODEL_PATH)
    model.train()
    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        raise ValueError("parent loader returned no trainable LoRA/control-token parameters")
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    completed = 0
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED + rank)
    if args.resume:
        completed = _restore_training(out / "latest.pt", model, optimizer, train,
                                      metadata, parent_hash, parent_file_hash,
                                      world_size, rank, generator)
        if rank == 0:
            _trim_updates(out / "updates.jsonl", completed)
        if world_size > 1:
            dist.barrier()
    if needs_final_checkpoint(target, completed, total):
        _save_checkpoint([out / "latest.pt", out / "checkpoint.pt"], model, optimizer,
                         train, metadata, parent_hash, parent_file_hash, completed,
                         total, world_size, rank, generator)
        if rank == 0:
            _write_json(out / "progress.json", {"status": "complete", "arm": "repair",
                                                 "completed_updates": completed,
                                                 "target_update": total, "total_updates": total})
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(status="complete", completed_updates=completed)
            _write_json(manifest_path, manifest)
        if world_size > 1:
            dist.barrier()
        return

    ddp_model = model
    if world_size > 1:
        def forward(module, repair, original_tasks, noise_generator):
            return objective.objective_tasks(module, train.tokenizer, repair,
                                             original_tasks, noise_generator)
        model.forward = MethodType(forward, model)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[0], output_device=0, find_unused_parameters=False)

    train.dist = dist
    if rank == 0:
        _write_json(out / "progress.json", {"status": "running", "arm": "repair",
                                             "completed_updates": completed,
                                             "target_update": target, "total_updates": total})
    started = time.monotonic()
    for update in range(completed + 1, target + 1):
        record = _train_update(train, objective, model, ddp_model, optimizer, ordered_rows,
                               update, rank, world_size, generator)
        if rank == 0:
            _append_jsonl(out / "updates.jsonl", record)
            _write_json(out / "progress.json", {
                "status": "running", "arm": "repair", "completed_updates": update,
                "target_update": target, "total_updates": total,
                "elapsed_seconds": time.monotonic() - started, "last_update": record,
            })
        if update % CHECKPOINT_INTERVAL == 0 or update == target:
            paths = [out / "latest.pt"]
            if update == total:
                paths.append(out / "checkpoint.pt")
            _save_checkpoint(paths, model, optimizer, train, metadata, parent_hash,
                             parent_file_hash, update, total, world_size, rank, generator)
    status = "complete" if target == total else "stopped"
    if rank == 0:
        _write_json(out / "progress.json", {"status": status, "arm": "repair",
                                             "completed_updates": target,
                                             "target_update": target, "total_updates": total})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(status=status, completed_updates=target)
        _write_json(manifest_path, manifest)


def per_material_seed(material_id):
    digest = hashlib.sha256(f"{SEED}:{material_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _read_validation_records(path, expected_ids, state_hash):
    if not path.exists():
        return {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        raw = b"".join(lines[:-1])
        path.write_bytes(raw)
        lines = lines[:-1]
    records = {}
    allowed = set(expected_ids)
    for line in lines:
        record = json.loads(line)
        material_id = record["material_id"]
        if material_id not in allowed or material_id in records:
            raise ValueError("validation JSONL contains an unexpected or duplicate material")
        if record.get("state_sha256") != state_hash:
            raise ValueError("validation JSONL belongs to a different repair checkpoint")
        records[material_id] = record
    return records


def _run_validate(args, rank, world_size, train, model, identity, pure_hash, versions,
                  parent, parent_hash, data_hashes, ordered_rows, val_rows, out, dist):
    torch = train.torch
    parent_file_hash = _sha256_file(PARENT_CHECKPOINT)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    training_world_size = manifest.get("metadata", {}).get("training_world_size")
    if training_world_size not in RANKS:
        raise ValueError("training manifest has no supported training world size")
    metadata = _metadata(identity, pure_hash, parent_hash, parent_file_hash,
                         parent["metadata_sha256"], versions, data_hashes,
                         ordered_rows, val_rows, training_world_size)
    if (manifest.get("schema") != 1 or manifest.get("arm") != "repair"
            or manifest.get("status") != "complete" or manifest.get("metadata") != metadata):
        raise ValueError("validation manifest differs from the completed training checkpoint")
    saved = MODEL.load_checkpoint(
        model, out / "checkpoint.pt", train, expected_arm="repair",
        parent_checkpoint_sha256=parent_file_hash, parent_trainable_sha256=parent_hash,
        metadata=metadata)
    state_hash = saved["trainable_state_sha256"]
    model.eval()
    objective = _load_module("crystal_repair_validation_objective", MODEL_PATH)
    shard_rows = [(index, row) for index, row in enumerate(val_rows) if index % world_size == rank]
    records_path = out / f"validation-rank-{rank:02d}.jsonl"
    shard_ids = [row["material_id"] for _index, row in shard_rows]
    records = _read_validation_records(records_path, shard_ids, state_hash)
    if records and not args.resume:
        raise FileExistsError("validation records already exist; pass --resume")
    for index, row in shard_rows:
        material_id = row["material_id"]
        if material_id in records:
            continue
        seed = per_material_seed(material_id)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, stats = objective.objective_tasks(
                model, train.tokenizer, row["taskR"], row["original_tasks"], generator)
            start_ok = MODEL.teacher_forced_start_correct(model, train.tokenizer, row["taskR"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss for {material_id}")
        record = {"index": index, "material_id": material_id, "arm": "repair",
                  "noise_seed": seed, "state_sha256": state_hash,
                  "loss": float(loss.float()), "repair_start_teacher_forced_correct": start_ok,
                  **{key: float(stats[key]) for key in STAT_KEYS}}
        _append_jsonl(records_path, record)
        records[material_id] = record
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        all_records = []
        for shard_rank in range(world_size):
            path = out / f"validation-rank-{shard_rank:02d}.jsonl"
            expected = [row["material_id"] for index, row in enumerate(val_rows)
                        if index % world_size == shard_rank]
            shard = _read_validation_records(path, expected, state_hash)
            all_records.extend(shard.values())
        all_records.sort(key=lambda record: record["index"])
        expected_ids = [row["material_id"] for row in val_rows]
        if ([record["material_id"] for record in all_records] != expected_ids
                or len(all_records) != len(expected_ids)):
            raise ValueError("distributed validation did not cover every unique validation material")
        keys = ("loss",) + STAT_KEYS
        _write_json(out / "validation-summary.json", {
            "schema": 1,
            "status": "complete",
            "arm": "repair",
            "n": len(all_records),
            "checkpoint_state_sha256": state_hash,
            "dataset_hashes": data_hashes,
            "validation_ids_sha256": _json_sha256(expected_ids),
            "metrics": {key: {"count": len(all_records), "mean": _mean(all_records, key)}
                        for key in keys},
            "repair_start_teacher_forced_correct": {
                "correct": sum(record["repair_start_teacher_forced_correct"]
                                for record in all_records),
                "count": len(all_records),
            },
        })


def run(args):
    if args.mode == "validate" and args.max_updates is not None:
        raise ValueError("--max-updates applies only to training")
    data_hashes, train_rows, val_rows, manifest = read_data(args.data)
    ordered_rows = list(train_rows)
    random.Random(SEED).shuffle(ordered_rows)
    rank, world_size, torch, dist = _initialize_runtime()
    out = _output_path(args)
    try:
        full, train, model, tok, identity, pure_hash, versions, parent, parent_hash = _load_parent_model()
        if train.torch.cuda.device_count() != 1:
            raise RuntimeError("parent model must see exactly the rank's selected GPU")
        train.dist = dist
        train.tokenizer = tok
        _source_manifest, source_hashes, _source_train, _source_val = train.read_frozen_data(
            train.DEFAULT_DATA)
        source_fields = ("manifest_sha256", "train_sha256", "val_sha256")
        manifest_source = manifest.get("source", {})
        for key in source_fields:
            if (manifest_source.get(key) is not None
                    and manifest_source[key] != source_hashes[key]):
                raise ValueError(f"repair manifest frozen source hash differs: {key}")
        if data_hashes["source_manifest_sha256"] != source_hashes["manifest_sha256"]:
            raise ValueError("repair manifest points to a different frozen source dataset")
        raw_source_train, raw_source_val = read_raw_source_splits(train.DEFAULT_DATA, source_hashes)
        validate_source_alignment(train_rows, val_rows, raw_source_train, raw_source_val, train.tasks)
        data_hashes["frozen_source_hashes"] = {key: source_hashes[key] for key in source_fields}
        if args.mode == "train":
            _run_train(args, rank, world_size, train, model, identity, pure_hash, versions,
                       parent, parent_hash, data_hashes, ordered_rows, val_rows, out, dist)
        else:
            _run_validate(args, rank, world_size, train, model, identity, pure_hash, versions,
                          parent, parent_hash, data_hashes, ordered_rows, val_rows, out, dist)
    finally:
        if world_size > 1:
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "validate"), default="train")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
