"""Train and validate the frozen full-set Qwen3-8B FLARE transfer."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
from types import MethodType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
PILOT_RUN_PATH = ROOT / "experiments/flare_crystal_qwen3_8b_20261019/run.py"
PILOT_MODEL_PATH = ROOT / "experiments/flare_crystal_qwen3_8b_20261019/model.py"
SOURCE_CHECKPOINT = ROOT / "experiments/structure_conditioned_ar8b_20261017/run_001/checkpoint.pt"
SEED = 42
TRAIN_MATERIALS = 7272
VAL_MATERIALS = 900
GLOBAL_MATERIALS_PER_UPDATE = 8
UPDATES = TRAIN_MATERIALS // GLOBAL_MATERIALS_PER_UPDATE
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
CHECKPOINT_INTERVAL = 50
ARMS = ("ar", "flare")
STAT_KEYS = ("clean_ar", "denoise", "reading_ar")

CONFIG = {
    "experiment": "flare_crystal_full_20261019",
    "train_materials": TRAIN_MATERIALS,
    "validation_materials": VAL_MATERIALS,
    "seed": SEED,
    "order": "source train.epoch_order(train_rows), seed 42",
    "updates": UPDATES,
    "materials_per_update": GLOBAL_MATERIALS_PER_UPDATE,
    "tasks_per_material": ["G", "Q"],
    "objective": "AR: G and Q causal AR; FLARE: G clean AR plus complementary crystal denoising, Q causal AR",
    "material_loss": "(G + Q) / 2",
    "optimizer": "fresh AdamW per arm; constant learning rate",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_norm": GRAD_CLIP_NORM,
    "precision": "BF16 autocast; source LoRA and trainable token parameter set",
    "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
    "validation": "all 900 frozen validation materials; no gradient; per-material deterministic noise seed",
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model():
    """Use the pilot's verified six-value source loader, initialized from pure AR."""
    pilot = _load_module("flare_crystal_full_source_pilot", PILOT_RUN_PATH)
    return pilot.load_model()


def isolate_rank_gpu(environ=None):
    """Expose only this torchrun rank's selected physical GPU before CUDA import."""
    environ = os.environ if environ is None else environ
    visible = [device.strip() for device in environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
               if device.strip()]
    if not visible:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to the explicitly selected GPU list")
    rank = int(environ.get("RANK", "0"))
    world_size = int(environ.get("WORLD_SIZE", "1"))
    local_rank = int(environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 2) or not 0 <= local_rank < len(visible):
        raise ValueError("launcher rank or visible-GPU count differs from the one/two-GPU protocol")
    environ["CUDA_VISIBLE_DEVICES"] = visible[local_rank]
    return rank, world_size, local_rank


def partition_update(ordered_rows, update, rank, world_size):
    if world_size not in (1, 2) or GLOBAL_MATERIALS_PER_UPDATE % world_size:
        raise ValueError("training supports one or two ranks with equal local material batches")
    if not 1 <= update <= UPDATES or not 0 <= rank < world_size:
        raise ValueError("update or rank is outside the frozen training plan")
    local_size = GLOBAL_MATERIALS_PER_UPDATE // world_size
    start = (update - 1) * GLOBAL_MATERIALS_PER_UPDATE + rank * local_size
    return list(ordered_rows[start:start + local_size])


def per_material_seed(material_id):
    digest = hashlib.sha256(f"{SEED}:{material_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


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


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_torch_save(path, value, torch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _metadata(identity, source_hash, versions, data_hashes, ordered_rows, val_rows,
              arm, train_world_size):
    train_ids = [row["material_id"] for row in ordered_rows]
    val_ids = [row["material_id"] for row in val_rows]
    if len(train_ids) != TRAIN_MATERIALS or len(set(train_ids)) != TRAIN_MATERIALS:
        raise ValueError("frozen train order must contain 7272 unique materials")
    if len(val_ids) != VAL_MATERIALS or len(set(val_ids)) != VAL_MATERIALS:
        raise ValueError("frozen validation split must contain 900 unique materials")
    if set(train_ids) & set(val_ids):
        raise ValueError("frozen train and validation IDs overlap")
    code_paths = {
        "run.py": Path(__file__).resolve(),
        "model.py": EXPERIMENT / "model.py",
        "pilot_run.py": PILOT_RUN_PATH,
        "pilot_model.py": PILOT_MODEL_PATH,
        "source_evaluator.py": ROOT / "experiments/structure_conditioned_ar8b_20261017/evaluate.py",
    }
    config = {**CONFIG, "arm": arm, "training_world_size": train_world_size}
    return {
        "config": config,
        "config_sha256": _json_sha256(config),
        "arm": arm,
        "training_world_size": train_world_size,
        "source_identity": identity,
        "source_trainable_sha256": source_hash,
        "source_checkpoint": {
            "path": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
            "sha256": _sha256_file(SOURCE_CHECKPOINT),
        },
        "runtime_versions": versions,
        "dataset_hashes": data_hashes,
        "epoch_order_sha256": _json_sha256(train_ids),
        "validation_ids_sha256": _json_sha256(val_ids),
        "source_sha256": {name: _sha256_file(path) for name, path in code_paths.items()},
    }


def trainable_state(model):
    return {name: param.detach().cpu().contiguous().clone()
            for name, param in model.named_parameters() if param.requires_grad}


def validate_checkpoint_identity(saved, expected_arm, metadata=None, world_size=None,
                                 allow_partial=False):
    if saved.get("schema") != 1 or saved.get("arm") != expected_arm:
        raise ValueError("checkpoint schema or arm differs")
    saved_metadata = saved.get("metadata")
    if not isinstance(saved_metadata, dict) or saved.get("metadata_sha256") != _json_sha256(saved_metadata):
        raise ValueError("checkpoint metadata is missing or failed its hash check")
    required = {"config", "config_sha256", "source_identity", "source_trainable_sha256",
                "source_checkpoint", "runtime_versions", "dataset_hashes",
                "epoch_order_sha256", "validation_ids_sha256", "source_sha256",
                "arm", "training_world_size"}
    if not required <= saved_metadata.keys():
        raise ValueError("checkpoint metadata is incomplete")
    if saved_metadata["config_sha256"] != _json_sha256(saved_metadata["config"]):
        raise ValueError("checkpoint config failed its hash check")
    if saved_metadata["source_trainable_sha256"] != saved.get("source_trainable_sha256"):
        raise ValueError("checkpoint metadata source hash differs")
    if not {"run.py", "model.py", "pilot_run.py", "pilot_model.py"} <= saved_metadata["source_sha256"].keys():
        raise ValueError("checkpoint source-code hashes are incomplete")
    if saved_metadata.get("arm") != expected_arm:
        raise ValueError("checkpoint metadata arm differs")
    saved_world_size = saved.get("world_size")
    if saved_metadata.get("training_world_size") != saved_world_size:
        raise ValueError("checkpoint metadata world size differs")
    if world_size is not None and saved_world_size != world_size:
        raise ValueError("checkpoint world size differs")
    if metadata is not None and saved_metadata != metadata:
        raise ValueError("checkpoint metadata differs from current code, data, or protocol")
    completed = saved.get("completed_updates", 0)
    if not 0 < completed <= UPDATES or (not allow_partial and completed != UPDATES):
        raise ValueError("checkpoint is partial or has an invalid update count")
    if len(saved.get("rng_states", [])) != saved_world_size:
        raise ValueError("checkpoint does not contain RNG state for every training rank")


def load_checkpoint(model, path, train, expected_arm, allow_partial=False):
    """Verify checkpoint identity and complete trainable state before restoring it."""
    torch = train.torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(saved, expected_arm, allow_partial=allow_partial)
    if saved.get("source_trainable_sha256") != train.trainable_sha256(model):
        raise ValueError("checkpoint source trainable state differs")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    state = saved.get("trainable_state", {})
    names = sorted(params)
    if saved.get("trainable_parameter_names") != names or sorted(state) != names:
        raise ValueError("checkpoint trainable parameter names differ")
    for name, param in params.items():
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"checkpoint trainable parameter shape differs: {name}")
    if train.state_sha256(state) != saved.get("trainable_state_sha256"):
        raise ValueError("checkpoint trainable state failed its hash check")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
    if train.trainable_sha256(model) != saved["trainable_state_sha256"]:
        raise ValueError("restored trainable state failed its hash check")
    return saved


def _rng_state(torch, generator):
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
        "generator": generator.get_state(),
    }


def _gather_rng_states(torch, dist, generator, world_size):
    local = _rng_state(torch, generator)
    if world_size == 1:
        return [local]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered


def _save_checkpoint(paths, model, optimizer, train, metadata, arm, source_hash,
                     completed, world_size, rank, generator):
    torch = train.torch
    dist = train.dist
    rng_states = _gather_rng_states(torch, dist, generator, world_size)
    if rank == 0:
        state = trainable_state(model)
        payload = {
            "schema": 1,
            "arm": arm,
            "metadata": metadata,
            "metadata_sha256": _json_sha256(metadata),
            "source_trainable_sha256": source_hash,
            "completed_updates": completed,
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
            "status": "checkpointed", "arm": arm, "completed_updates": completed,
            "trainable_state_sha256": payload["trainable_state_sha256"],
        })
    if world_size > 1:
        dist.barrier()


def _restore_training(path, model, optimizer, train, arm, source_hash,
                      metadata, world_size, rank, generator):
    torch = train.torch
    saved = load_checkpoint(model, path, train, arm, allow_partial=True)
    validate_checkpoint_identity(saved, arm, metadata, world_size, allow_partial=True)
    if saved["source_trainable_sha256"] != source_hash:
        raise ValueError("resume source checkpoint differs")
    optimizer.load_state_dict(saved["optimizer_state"])
    rng = saved["rng_states"][rank]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"], torch.cuda.current_device())
    generator.set_state(rng["generator"])
    return saved["completed_updates"]


def _output_path(args):
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(EXPERIMENT.resolve()) or out == EXPERIMENT.resolve():
        raise ValueError(f"output must be a child of {EXPERIMENT}")
    if args.mode == "train":
        if args.resume:
            if not out.is_dir() or not (out / "manifest.json").is_file():
                raise FileNotFoundError("--resume requires an existing run manifest")
        elif out.exists():
            raise FileExistsError(f"refusing to overwrite run output: {out}")
    elif not out.is_dir():
        raise FileNotFoundError("validation requires the completed training output directory")
    return out


def _load_training_sources():
    train, model, tok, identity, source_hash, versions = load_model()
    if train.torch.cuda.device_count() != 1 or not train.torch.cuda.is_available():
        raise RuntimeError("source loader must see exactly the rank's selected CUDA device")
    if source_hash != train.trainable_sha256(model):
        raise ValueError("source evaluator hash differs from the loaded pure-AR checkpoint")
    return train, model, tok, identity, source_hash, versions


def _read_data(train):
    _manifest, data_hashes, train_rows, val_rows = train.read_frozen_data(train.DEFAULT_DATA)
    if len(train_rows) != TRAIN_MATERIALS or len(val_rows) != VAL_MATERIALS:
        raise ValueError("frozen train/validation split must be exactly 7272/900")
    ordered = train.epoch_order(train_rows)
    return data_hashes, ordered, val_rows


def _mean(records, key):
    return sum(float(record[key]) for record in records) / len(records)


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
            kept.append(json.dumps(record, sort_keys=True, allow_nan=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("".join(line + "\n" for line in kept))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _train_update(train, objective, model, ddp_model, optimizer, ordered,
                  update, arm, rank, world_size, generator):
    torch, dist = train.torch, train.dist
    local_rows = partition_update(ordered, update, rank, world_size)
    optimizer.zero_grad(set_to_none=True)
    local_records = []
    parameters = [param for param in model.parameters() if param.requires_grad]
    for local_index, row in enumerate(local_rows):
        examples = train.tasks(row)
        if len(examples) != 2 or [item["task"] for item in examples] != ["G", "Q"]:
            raise ValueError(f"expected one G and one Q task for {row['material_id']}")
        sync = local_index == len(local_rows) - 1
        context = ddp_model.no_sync() if world_size > 1 and not sync else nullcontext()
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if world_size > 1:
                    loss, stats = ddp_model(examples, train.tokenizer, arm, generator)
                else:
                    loss, stats = objective.objective_tasks(model, train.tokenizer,
                                                             examples, generator, arm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {update}")
            (loss / len(local_rows)).backward()
        local_records.append({"index": (update - 1) * GLOBAL_MATERIALS_PER_UPDATE
                                      + rank * len(local_rows) + local_index,
                              "material_id": row["material_id"],
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
    if len(records) != GLOBAL_MATERIALS_PER_UPDATE:
        raise ValueError("distributed update did not cover eight unique materials")
    return {
        "update": update,
        "arm": arm,
        "material_ids": [record["material_id"] for record in records],
        "loss": _mean(records, "loss"),
        **{key: _mean(records, key) for key in STAT_KEYS},
        "gradient_norm": sum(part["gradient_norm"] for part in gathered) / world_size,
    }


def _initialize_runtime():
    rank, world_size, _local_rank = isolate_rank_gpu()
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("each process must expose exactly one selected CUDA device")
    torch.cuda.set_device(0)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, torch, dist


def _check_world_size(arm, world_size):
    expected = 2 if arm == "flare" else 1
    if world_size != expected:
        raise ValueError(f"{arm} training requires world size {expected}")


def _run_train(args, rank, world_size, train, model, tok, identity, source_hash,
               versions, out, dist):
    torch = train.torch
    _check_world_size(args.arm, world_size)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("training requires BF16 support on the selected GPU")
    target = UPDATES if args.stop_after_update is None else args.stop_after_update
    if not 1 <= target <= UPDATES:
        raise ValueError(f"stop-after-update must be in 1..{UPDATES}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED + rank)
    data_hashes, ordered, val_rows = _read_data(train)
    metadata = _metadata(identity, source_hash, versions, data_hashes, ordered,
                         val_rows, args.arm, world_size)
    manifest_path = out / "manifest.json"
    if args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (existing.get("schema") != 1 or existing.get("arm") != args.arm
                or existing.get("metadata") != metadata):
            raise ValueError("resume manifest differs in arm, source, data, code, or protocol")
    elif rank == 0:
        out.mkdir(parents=True)
        _write_json(manifest_path, {"schema": 1, "arm": args.arm,
                                    "status": "running", "metadata": metadata})
    if world_size > 1:
        dist.barrier()

    objective = _load_module("flare_crystal_full_objective", EXPERIMENT / "model.py")
    model.train()
    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        raise ValueError("source loader returned no trainable LoRA/token parameters")
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    completed = 0
    if args.resume:
        completed = _restore_training(out / "latest.pt", model, optimizer, train,
                                      args.arm, source_hash, metadata, world_size,
                                      rank, generator)
        if rank == 0:
            _trim_updates(out / "updates.jsonl", completed)
        if world_size > 1:
            dist.barrier()
    if target <= completed:
        raise ValueError("stop-after-update must exceed the resumed update")

    ddp_model = model
    if world_size > 1:
        def forward(module, examples, tokenizer, arm, noise_generator):
            return objective.objective_tasks(module, tokenizer, examples, noise_generator, arm)
        model.forward = MethodType(forward, model)
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[0], output_device=0, find_unused_parameters=False)

    if rank == 0:
        _write_json(out / "progress.json", {"status": "running", "arm": args.arm,
                                             "completed_updates": completed,
                                             "target_update": target})
    started = time.monotonic()
    for update in range(completed + 1, target + 1):
        record = _train_update(train, objective, model, ddp_model, optimizer, ordered,
                               update, args.arm, rank, world_size, generator)
        if rank == 0:
            _append_jsonl(out / "updates.jsonl", record)
            _write_json(out / "progress.json", {"status": "running", "arm": args.arm,
                                                 "completed_updates": update,
                                                 "target_update": target,
                                                 "elapsed_seconds": time.monotonic() - started,
                                                 "last_update": record})
        if update % CHECKPOINT_INTERVAL == 0 or update == target:
            paths = [out / "latest.pt"]
            if update == UPDATES:
                paths.append(out / "checkpoint.pt")
            _save_checkpoint(paths, model, optimizer, train, metadata, args.arm,
                             source_hash, update, world_size, rank, generator)
    status = "complete" if target == UPDATES else "stopped"
    if rank == 0:
        _write_json(out / "progress.json", {"status": status, "arm": args.arm,
                                             "completed_updates": target,
                                             "target_update": target})
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(status=status, completed_updates=target)
        _write_json(manifest_path, manifest)


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
            raise ValueError("validation JSONL belongs to a different checkpoint")
        records[material_id] = record
    return records


def _run_validate(args, train, model, tok, identity, source_hash, versions, out):
    torch = train.torch
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("validation must run on one visible GPU in a single process")
    arm_world_size = 2 if args.arm == "flare" else 1
    data_hashes, ordered, val_rows = _read_data(train)
    metadata = _metadata(identity, source_hash, versions, data_hashes, ordered,
                         val_rows, args.arm, arm_world_size)
    saved = load_checkpoint(model, out / "checkpoint.pt", train, args.arm)
    validate_checkpoint_identity(saved, args.arm, metadata, arm_world_size)
    state_hash = saved["trainable_state_sha256"]
    model.eval()
    objective = _load_module("flare_crystal_full_validation_objective", EXPERIMENT / "model.py")
    records_path = out / "validation.jsonl"
    val_ids = [row["material_id"] for row in val_rows]
    records = _read_validation_records(records_path, val_ids, state_hash)
    for index, row in enumerate(val_rows):
        material_id = row["material_id"]
        if material_id in records:
            continue
        examples = train.tasks(row)
        if len(examples) != 2 or [item["task"] for item in examples] != ["G", "Q"]:
            raise ValueError(f"expected one G and one Q validation task for {material_id}")
        seed = per_material_seed(material_id)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, stats = objective.objective_tasks(model, tok, examples, generator, args.arm)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite validation loss for {material_id}")
        record = {"index": index, "material_id": material_id, "arm": args.arm,
                  "noise_seed": seed, "state_sha256": state_hash,
                  "loss": float(loss.float()),
                  **{key: float(stats[key]) for key in STAT_KEYS}}
        _append_jsonl(records_path, record)
        records[material_id] = record
    ordered_records = [records[material_id] for material_id in val_ids]
    summary = {
        "schema": 1,
        "status": "complete",
        "arm": args.arm,
        "n": len(ordered_records),
        "checkpoint_state_sha256": state_hash,
        "dataset_hashes": data_hashes,
        "validation_ids_sha256": _json_sha256(val_ids),
        "metrics": {key: {"count": len(ordered_records),
                           "mean": _mean(ordered_records, key)}
                    for key in ("loss",) + STAT_KEYS},
    }
    _write_json(out / "validation-summary.json", summary)


def run(args):
    if args.arm not in ARMS:
        raise ValueError(f"unknown arm: {args.arm}")
    if args.mode == "validate" and (args.resume or args.stop_after_update is not None):
        raise ValueError("--resume and --stop-after-update apply only to training")
    rank, world_size, torch, dist = _initialize_runtime()
    if args.mode == "train":
        _check_world_size(args.arm, world_size)
    elif world_size != 1:
        raise ValueError("validation requires one process")
    out = _output_path(args)
    train, model, tok, identity, source_hash, versions = _load_training_sources()
    train.dist = dist
    train.tokenizer = tok
    if args.mode == "train":
        _run_train(args, rank, world_size, train, model, tok, identity,
                   source_hash, versions, out, dist)
    else:
        _run_validate(args, train, model, tok, identity, source_hash, versions, out)
    if world_size > 1:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "validate"), default="train")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-update", type=int)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
