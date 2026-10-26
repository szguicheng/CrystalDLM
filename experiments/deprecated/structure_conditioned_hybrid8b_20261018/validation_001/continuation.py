"""Paired 128-update continuation from the frozen completed hybrid checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(__file__).resolve().parent
EVALUATOR_PATH = ROOT / "experiments/structure_conditioned_hybrid8b_20261018/evaluate.py"
UPDATES = 128
MATERIALS_PER_UPDATE = 8
SEED = 42
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
CHECKPOINT_INTERVAL = 32

PROTOCOL = {
    "updates": UPDATES,
    "materials_per_update": MATERIALS_PER_UPDATE,
    "ordered_materials": "first 1024 rows from train.epoch_order(frozen_train_rows)",
    "seed": SEED,
    "optimizer": "fresh AdamW per arm",
    "learning_rate": LEARNING_RATE,
    "learning_rate_schedule": "constant",
    "weight_decay": WEIGHT_DECAY,
    "gradient_clip_norm": GRAD_CLIP_NORM,
    "autocast": "bfloat16 on the selected CUDA device",
    "tasks": "unchanged G and Q examples from train.tasks(row)",
    "control_noise": "uniform integer k in 1..32",
    "highmask_noise": "u<0.5 selects k=32; otherwise the paired uniform k in 1..32",
    "noise_generator": "dedicated CPU torch.Generator; seed 42 in both arms",
    "mask_generator": "separate CPU torch.Generator; seed 42 in both arms",
    "crystal_loss_weights": "existing 32/k and PAD 0.1 weights retained; noise objective is intentionally reweighted",
    "checkpoint_interval_updates": CHECKPOINT_INTERVAL,
}


def select_noise_step(arm, uniform_k, branch_u, steps=32):
    if arm not in ("control", "highmask"):
        raise ValueError(f"unknown arm: {arm}")
    return steps if arm == "highmask" and branch_u < 0.5 else uniform_k


def draw_noise_step(arm, generator, steps=32):
    import torch

    uniform_k = int(torch.randint(1, steps + 1, (1,), generator=generator))
    branch_u = float(torch.rand((), generator=generator))
    return {
        "uniform_k": uniform_k,
        "branch_u": branch_u,
        "k": select_noise_step(arm, uniform_k, branch_u, steps),
    }


def load_continuation(model, checkpoint_path, train):
    """Apply a completed arm checkpoint after verifying its frozen source and state."""
    torch = train.torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["schema"] != 1 or checkpoint["completed_updates"] != UPDATES:
        raise ValueError("continuation checkpoint is not a completed 128-update run")
    source_hash = train.trainable_sha256(model)
    if checkpoint["source_trainable_sha256"] != source_hash:
        raise ValueError("continuation checkpoint was trained from a different source state")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    state = checkpoint["trainable_state"]
    names = sorted(params)
    if checkpoint["trainable_parameter_names"] != names or sorted(state) != names:
        raise ValueError("continuation checkpoint trainable parameter names differ")
    for name, param in params.items():
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"continuation parameter shape differs: {name}")
    if train.state_sha256(state) != checkpoint["trainable_state_sha256"]:
        raise ValueError("continuation checkpoint trainable state failed its hash check")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
    if train.trainable_sha256(model) != checkpoint["trainable_state_sha256"]:
        raise ValueError("restored continuation trainable state failed its hash check")
    return checkpoint["metadata"]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_output(args):
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"continuation outputs must stay under {OUTPUT_ROOT}")
    if args.resume:
        resume = args.resume.expanduser().resolve(strict=True)
        if not resume.is_relative_to(out):
            raise ValueError("resume checkpoint must be inside --out")
        if not out.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {out}")
    else:
        if out.exists():
            raise FileExistsError(f"refusing to overwrite continuation output: {out}")
        out.mkdir(parents=True)
        resume = None
    return out, resume


def _append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _trim_updates(path, completed_updates):
    if not path.exists():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break
        if entry["update"] <= completed_updates:
            kept.append(json.dumps(entry, sort_keys=True, allow_nan=False))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    temporary.replace(path)


def _checkpoint_path(out, completed_updates):
    if completed_updates == UPDATES:
        return out / "checkpoint.pt"
    return out / "checkpoints" / f"step_{completed_updates:03d}.pt"


def _save_checkpoint(path, model, optimizer, train, metadata, arm,
                     source_trainable_sha256, completed_updates,
                     noise_generator, mask_generator):
    torch = train.torch
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    state = train.trainable_state(model)
    checkpoint = {
        "schema": 1,
        "arm": arm,
        "source_trainable_sha256": source_trainable_sha256,
        "metadata": metadata,
        "completed_updates": completed_updates,
        "trainable_parameter_names": sorted(state),
        "trainable_state": state,
        "trainable_state_sha256": train.state_sha256(state),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(torch.cuda.current_device()),
            "noise_generator": noise_generator.get_state(),
            "mask_generator": mask_generator.get_state(),
        },
    }
    train.atomic_torch_save(path, checkpoint)


def _restore_resume(path, model, optimizer, train, metadata, arm,
                    source_trainable_sha256, noise_generator, mask_generator):
    torch = train.torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    completed = checkpoint["completed_updates"]
    if checkpoint["schema"] != 1 or checkpoint["arm"] != arm or not 0 < completed < UPDATES:
        raise ValueError("resume checkpoint has an invalid schema, arm, or update count")
    if checkpoint["metadata"] != metadata:
        raise ValueError("resume checkpoint source, data, order, or protocol identity differs")
    if checkpoint["source_trainable_sha256"] != source_trainable_sha256:
        raise ValueError("resume checkpoint source trainable state differs")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    state = checkpoint["trainable_state"]
    names = sorted(params)
    if checkpoint["trainable_parameter_names"] != names or sorted(state) != names:
        raise ValueError("resume checkpoint trainable parameter names differ")
    if train.state_sha256(state) != checkpoint["trainable_state_sha256"]:
        raise ValueError("resume checkpoint trainable state failed its hash check")
    for name, param in params.items():
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"resume parameter shape differs: {name}")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
    if train.trainable_sha256(model) != checkpoint["trainable_state_sha256"]:
        raise ValueError("restored resume trainable state failed its hash check")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state(rng["torch_cuda"], torch.cuda.current_device())
    noise_generator.set_state(rng["noise_generator"])
    mask_generator.set_state(rng["mask_generator"])
    return completed


def _mean(values):
    return sum(values) / len(values)


def _source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run(args):
    out, resume_path = _resolve_output(args)
    evaluator = _load_module("bounded_hybrid8b_evaluator", EVALUATOR_PATH)
    train, model, tok, identity, source_hash, versions = evaluator._load_model(
        SimpleNamespace(model_path=evaluator.MODEL, data=evaluator.DATA,
                        checkpoint=evaluator.DEFAULT_CHECKPOINT))
    torch = train.torch
    if source_hash != train.trainable_sha256(model):
        raise ValueError("evaluator source hash does not match the loaded model")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("continuation requires exactly one explicitly selected CUDA device")
    torch.set_num_threads(4)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model.train()

    _manifest, data_hashes, train_rows, _val_rows = train.read_frozen_data(evaluator.DATA)
    ordered = train.epoch_order(train_rows)[:UPDATES * MATERIALS_PER_UPDATE]
    ordered_ids = [row["material_id"] for row in ordered]
    if len(ordered) != UPDATES * MATERIALS_PER_UPDATE or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("frozen first-1024 material order is incomplete or duplicated")
    metadata = {
        "arm": args.arm,
        "source_trainable_sha256": source_hash,
        "continuation_source_sha256": _source_sha256(),
        "source_identity": identity,
        "source_checkpoint": {
            "path": str(evaluator.DEFAULT_CHECKPOINT.relative_to(ROOT)),
            "sha256": evaluator._sha256_file(evaluator.DEFAULT_CHECKPOINT),
            "completed_updates": train.UPDATES,
        },
        "runtime_versions": versions,
        "data_hashes": data_hashes,
        "ordered_material_ids": ordered_ids,
        "ordered_material_ids_sha256": train.json_sha256(ordered_ids),
        "protocol": PROTOCOL,
    }
    manifest_path = out / "manifest.json"
    manifest = {"schema": 1, "status": "running", "metadata": metadata}
    if resume_path:
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_manifest["schema"] != 1 or saved_manifest["metadata"] != metadata:
            raise ValueError("existing run manifest differs from the frozen continuation identity")
    else:
        train.write_json(manifest_path, manifest)

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    noise_generator = torch.Generator(device="cpu").manual_seed(SEED)
    mask_generator = torch.Generator(device="cpu").manual_seed(SEED)
    completed = 0
    if resume_path:
        completed = _restore_resume(
            resume_path, model, optimizer, train, metadata, args.arm, source_hash,
            noise_generator, mask_generator)
        _trim_updates(out / "updates.jsonl", completed)

    stop_after = args.stop_after_update or UPDATES
    if not completed < stop_after <= UPDATES:
        raise ValueError("--stop-after-update must exceed the resumed step and be at most 128")

    model_device = model.time.weight.device
    assert model_device.type == "cuda"
    updates_path = out / "updates.jsonl"
    for update in range(completed + 1, stop_after + 1):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        losses, text_losses, crystal_losses = [], [], []
        material_ids, noise_draws = [], []
        for index in range((update - 1) * MATERIALS_PER_UPDATE,
                           update * MATERIALS_PER_UPDATE):
            row = ordered[index]
            examples = train.tasks(row)
            generation = [example for example in examples if "crystal" in example]
            if len(generation) != 1:
                raise ValueError(f"expected exactly one G crystal example for {row['material_id']}")
            noise = draw_noise_step(args.arm, noise_generator)
            generation[0]["noise_step"] = noise["k"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, stats = model.loss(examples, tok, "hybrid", mask_generator)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at update {update}")
                (loss / MATERIALS_PER_UPDATE).backward()
            losses.append(float(loss.detach()))
            text_losses.append(float(stats["text"]))
            crystal_losses.append(float(stats["crystal"]))
            material_ids.append(row["material_id"])
            noise_draws.append(noise)

        finite_gradients = all(
            torch.isfinite(param.grad).all().item()
            for param in params if param.grad is not None)
        if not finite_gradients:
            raise FloatingPointError(f"nonfinite gradient at update {update}")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite gradient norm at update {update}")
        optimizer.step()
        record = {
            "update": update,
            "arm": args.arm,
            "learning_rate": LEARNING_RATE,
            "loss": _mean(losses),
            "text_loss": _mean(text_losses),
            "crystal_loss": _mean(crystal_losses),
            "grad_norm": float(grad_norm.detach()),
            "finite_gradients": finite_gradients,
            "first_material_index": (update - 1) * MATERIALS_PER_UPDATE,
            "next_material_index": update * MATERIALS_PER_UPDATE,
            "material_ids": material_ids,
            "noise_draws": noise_draws,
            "seconds": time.monotonic() - started,
        }
        _append_record(updates_path, record)
        if update % CHECKPOINT_INTERVAL == 0 or update == stop_after:
            _save_checkpoint(
                _checkpoint_path(out, update), model, optimizer, train, metadata,
                args.arm, source_hash, update, noise_generator, mask_generator)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    manifest["status"] = "complete" if stop_after == UPDATES else "paused"
    manifest["completed_updates"] = stop_after
    manifest["checkpoint"] = str(_checkpoint_path(out, stop_after).relative_to(out))
    train.write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "highmask"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int,
                        help="pause after this update for a checkpoint/resume smoke")
    args = parser.parse_args(argv)
    if args.stop_after_update is not None and not 1 <= args.stop_after_update <= UPDATES:
        parser.error("--stop-after-update must be in 1..128")
    run(args)


if __name__ == "__main__":
    main()
