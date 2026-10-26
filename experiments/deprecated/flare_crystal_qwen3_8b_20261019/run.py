"""Run the fixed eight-material AR versus FLARE-objective crystal pilot."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
EVALUATOR_PATH = ROOT / "experiments/structure_conditioned_ar8b_20261017/evaluate.py"
SOURCE_CHECKPOINT = ROOT / "experiments/structure_conditioned_ar8b_20261017/run_001/checkpoint.pt"
MODEL_PATH = EXPERIMENT / "model.py"
SEED = 42
TRAIN_MATERIALS = 8
VAL_MATERIALS = 8
MATERIALS_PER_UPDATE = 2
UPDATES = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
CHECKPOINT_INTERVAL = 16
ARMS = ("ar", "flare")

CONFIG = {
    "experiment": "flare_crystal_qwen3_8b_20261019",
    "purpose": "eight-material overfit probe; not a generalization or performance experiment",
    "source": "frozen structure-conditioned pure-AR checkpoint; no hybrid checkpoint",
    "selection": "sample sorted train and validation rows independently with random.Random(42)",
    "train_materials": TRAIN_MATERIALS,
    "validation_materials": VAL_MATERIALS,
    "selection_seed": SEED,
    "updates": UPDATES,
    "materials_per_update": MATERIALS_PER_UPDATE,
    "material_order": "cycle the selected eight train rows in fixed sampled order",
    "task": "train.tasks(row)[0] generation task only",
    "flare_objective": "clean causal AR plus B=4 crystal denoising on complementary noisy views, with shifted targets and duplicated previous-token anchors; semantic FLARE-style transfer, not a bitwise reproduction of the official loss",
    "optimizer": "fresh AdamW per arm",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_norm": GRAD_CLIP_NORM,
    "precision": "bfloat16 autocast on one selected CUDA device",
    "loss_accumulation": "two material losses, each divided by two before backward",
    "noise_generator": "dedicated CPU torch.Generator seeded 42",
    "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evaluator():
    return _load_module("flare_crystal_source_evaluator", EVALUATOR_PATH)


def load_model():
    """Return the source evaluator's verified six-value pure-AR model tuple."""
    evaluator = _load_evaluator()
    return evaluator.RUNNER._load_model(SimpleNamespace(
        model_path=evaluator.MODEL, data=evaluator.DATA,
        checkpoint=evaluator.DEFAULT_CHECKPOINT))


def select_pilot_rows(train_rows, val_rows):
    train = random.Random(SEED).sample(
        sorted(train_rows, key=lambda row: row["material_id"]), TRAIN_MATERIALS)
    val = random.Random(SEED).sample(
        sorted(val_rows, key=lambda row: row["material_id"]), VAL_MATERIALS)
    return train, val


def materials_for_update(rows, update):
    start = (update - 1) * MATERIALS_PER_UPDATE
    return [rows[(start + offset) % TRAIN_MATERIALS]
            for offset in range(MATERIALS_PER_UPDATE)]


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _trim_updates(path, completed_update):
    path = Path(path)
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            break
        if record["update"] <= completed_update:
            kept.append(json.dumps(record, sort_keys=True, allow_nan=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path, value, torch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _trainable_state(model):
    return {name: param.detach().cpu().contiguous().clone()
            for name, param in model.named_parameters() if param.requires_grad}


def load_checkpoint(model, path, train, expected_arm, allow_partial=False):
    """Restore only trainable weights after checking their frozen source and exact state."""
    torch = train.torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["schema"] != 1 or saved["arm"] != expected_arm:
        raise ValueError("checkpoint schema or arm differs")
    completed = saved["completed_updates"]
    if not 0 < completed <= UPDATES or (not allow_partial and completed != UPDATES):
        raise ValueError("checkpoint is partial or has an invalid update count")
    if saved["source_trainable_sha256"] != train.trainable_sha256(model):
        raise ValueError("checkpoint source trainable state differs")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    state = saved["trainable_state"]
    names = sorted(params)
    if saved["trainable_parameter_names"] != names or sorted(state) != names:
        raise ValueError("checkpoint trainable parameter names differ")
    for name, param in params.items():
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"checkpoint trainable parameter shape differs: {name}")
    if train.state_sha256(state) != saved["trainable_state_sha256"]:
        raise ValueError("checkpoint trainable state failed its hash check")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
    if train.trainable_sha256(model) != saved["trainable_state_sha256"]:
        raise ValueError("restored trainable state failed its hash check")
    return saved


def _output_path(args):
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(EXPERIMENT.resolve()):
        raise ValueError(f"output must stay under {EXPERIMENT}")
    if args.resume:
        if not out.is_dir() or not (out / "manifest.json").is_file():
            raise FileNotFoundError("--resume requires an existing pilot output with a manifest")
    else:
        if out.exists():
            raise FileExistsError(f"refusing to overwrite pilot output: {out}")
    return out


def _metadata(identity, source_hash, versions, data_hashes, train_rows, val_rows):
    train_ids = [row["material_id"] for row in train_rows]
    val_ids = [row["material_id"] for row in val_rows]
    if set(train_ids) & set(val_ids):
        raise ValueError("pilot train and validation IDs overlap")
    source_sha256 = {"run.py": _sha256_file(Path(__file__).resolve()),
                     "model.py": _sha256_file(MODEL_PATH)}
    return {
        "config": CONFIG,
        "config_sha256": _json_sha256(CONFIG),
        "arm_source_trainable_sha256": source_hash,
        "source_identity": identity,
        "source_checkpoint": {"path": str(SOURCE_CHECKPOINT.relative_to(ROOT)),
                              "sha256": _sha256_file(SOURCE_CHECKPOINT)},
        "runtime_versions": versions,
        "dataset_hashes": data_hashes,
        "train_material_ids": train_ids,
        "validation_material_ids": val_ids,
        "source_sha256": source_sha256,
    }


def _save_checkpoint(path, model, optimizer, train, metadata, arm, source_hash,
                     completed_update, generator):
    torch = train.torch
    state = _trainable_state(model)
    _atomic_torch_save(path, {
        "schema": 1,
        "arm": arm,
        "metadata": metadata,
        "source_trainable_sha256": source_hash,
        "completed_updates": completed_update,
        "trainable_parameter_names": sorted(state),
        "trainable_state": state,
        "trainable_state_sha256": train.state_sha256(state),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
            "noise_generator": generator.get_state(),
        },
    }, torch)


def _restore_training(path, model, optimizer, train, arm, source_hash,
                      metadata, generator):
    saved = load_checkpoint(model, path, train, arm, allow_partial=True)
    if saved["source_trainable_sha256"] != source_hash or saved["metadata"] != metadata:
        raise ValueError("resume checkpoint source, data, code, or protocol identity differs")
    optimizer.load_state_dict(saved["optimizer_state"])
    rng = saved["rng_state"]
    random.setstate(rng["python"])
    train.torch.set_rng_state(rng["torch_cpu"])
    train.torch.cuda.set_rng_state(rng["torch_cuda"], train.torch.cuda.current_device())
    generator.set_state(rng["noise_generator"])
    return saved["completed_updates"]


def run(args):
    if args.arm not in ARMS:
        raise ValueError(f"unknown arm: {args.arm}")
    target_update = UPDATES if args.stop_after_update is None else args.stop_after_update
    if not 1 <= target_update <= UPDATES:
        raise ValueError(f"stop-after-update must be in 1..{UPDATES}")
    out = _output_path(args)
    train, model, tok, identity, source_hash, versions = load_model()
    torch = train.torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("pilot requires exactly one explicitly selected CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("pilot requires BF16 support on the selected CUDA device")
    if source_hash != train.trainable_sha256(model):
        raise ValueError("source evaluator hash does not match the loaded pure-AR checkpoint")
    torch.set_num_threads(4)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    model.train()

    _dataset_manifest, data_hashes, all_train, all_val = train.read_frozen_data(train.DEFAULT_DATA)
    train_rows, val_rows = select_pilot_rows(all_train, all_val)
    metadata = _metadata(identity, source_hash, versions, data_hashes,
                         train_rows, val_rows)
    manifest_path = out / "manifest.json"
    manifest = {"schema": 1, "arm": args.arm, "status": "running", "metadata": metadata}
    if args.resume:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest and not (
                existing.get("schema") == 1 and existing.get("arm") == args.arm
                and existing.get("metadata") == metadata):
            raise ValueError("existing pilot manifest does not match source, data, arm, or protocol")
    else:
        out.mkdir(parents=True)
        _write_json(manifest_path, manifest)

    objective = _load_module("flare_crystal_objective", MODEL_PATH).objective
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    completed = 0
    resume_path = out / "latest.pt" if (out / "latest.pt").exists() else out / "checkpoint.pt"
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError("resume output has no latest or final checkpoint")
        completed = _restore_training(resume_path, model, optimizer, train, args.arm,
                                      source_hash, metadata, generator)
        _trim_updates(out / "updates.jsonl", completed)
    if target_update <= completed:
        raise ValueError("stop-after-update must be greater than the resumed update")

    _write_json(out / "progress.json", {"status": "running", "arm": args.arm,
                                         "completed_updates": completed,
                                         "target_update": target_update})
    started = time.monotonic()
    for update in range(completed + 1, target_update + 1):
        optimizer.zero_grad(set_to_none=True)
        losses, clean_losses, denoise_losses = [], [], []
        material_ids = []
        for row in materials_for_update(train_rows, update):
            material_ids.append(row["material_id"])
            example = train.tasks(row)[0]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, stats = objective(model, tok, example, generator, args.arm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite objective at update {update}")
            (loss / MATERIALS_PER_UPDATE).backward()
            losses.append(float(loss.detach().float().item()))
            clean_losses.append(float(stats["clean_ar"]))
            denoise_losses.append(float(stats["denoise"]))
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, GRAD_CLIP_NORM, error_if_nonfinite=True)
        optimizer.step()
        record = {"update": update, "arm": args.arm, "material_ids": material_ids,
                  "loss": sum(losses) / len(losses),
                  "clean_ar": sum(clean_losses) / len(clean_losses),
                  "denoise": sum(denoise_losses) / len(denoise_losses),
                  "gradient_norm": float(grad_norm.detach().float().item()),
                  "elapsed_seconds": time.monotonic() - started}
        _append_jsonl(out / "updates.jsonl", record)
        _write_json(out / "progress.json", {"status": "running", "arm": args.arm,
                                             "completed_updates": update,
                                             "target_update": target_update,
                                             "last_update": record})
        if update % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(out / "latest.pt", model, optimizer, train, metadata,
                             args.arm, source_hash, update, generator)

    if target_update % CHECKPOINT_INTERVAL:
        _save_checkpoint(out / "latest.pt", model, optimizer, train, metadata,
                         args.arm, source_hash, target_update, generator)
    _save_checkpoint(out / "checkpoint.pt", model, optimizer, train, metadata,
                     args.arm, source_hash, target_update, generator)
    final_status = "complete" if target_update == UPDATES else "stopped"
    _write_json(out / "progress.json", {"status": final_status, "arm": args.arm,
                                         "completed_updates": target_update,
                                         "target_update": target_update})
    manifest["status"] = final_status
    manifest["completed_updates"] = target_update
    _write_json(manifest_path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-after-update", type=int)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
