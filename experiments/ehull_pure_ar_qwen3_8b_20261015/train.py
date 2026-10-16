"""Preflight, train, resume, or evaluate the frozen Qwen3-8B pure-AR task."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer

from model import Qwen3ARBaseline
from prepare import END, EOS, K, MASK, MODEL, PAD, REVISION, START, STEPS, tasks

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "experiments/ehull_pure_ar_baseline_20261013"
SHARED_DIR = ROOT / "experiments/ehull_interleaved_smoke_20261013"
OLD_MODEL_DIR = ROOT / "experiments/interleaved_ar_diffusion_20261002"
DEFAULT_DATA = ROOT / "research/crystal_training_preflight_20261013/data_001"

SEED = 42
TRAINING_GENERATOR_SEED = SEED + 1000
EVALUATION_GENERATOR_SEED = SEED + 2000
TRAIN_MATERIALS = 7272
VAL_MATERIALS = 900
MICRO_BATCH_TASKS = 2
MATERIALS_PER_UPDATE = 8
TASK_EXAMPLES_PER_UPDATE = MICRO_BATCH_TASKS * MATERIALS_PER_UPDATE
UPDATES = TRAIN_MATERIALS // MATERIALS_PER_UPDATE
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
CHECKPOINT_EVERY = 50
PREFLIGHT_UPDATES = 3
CHECKPOINT_SCHEMA = 1

EXPECTED_DATA_HASHES = {
    "manifest_sha256": "aa21f3fbb4e9a19a676d1012405575932134675c3684d336d2c41283a281e741",
    "train_sha256": "b74afafbca5a52eb1a63c9a7502c9fe1f5055cfef0c25f122de6d140c137080b",
    "val_sha256": "4e97dbe73aa2c4903aa11c4438f3c7c4da05a87eb1028632f121db4e8cd97f12",
}
EXPECTED_ORDER_SHA256 = "7abf514e32c5d5aa109049c09e2a3b9d6a3ede86ffa01c17e78aac0167423247"
EXPECTED_EVAL_IDS_SHA256 = "d3cad1379df8d17089e5bc196fc6abedc09ca1f11f204370839df2471aceaa8c"


def load_reference_baseline():
    sys.path.insert(0, str(EXPERIMENT))
    spec = importlib.util.spec_from_file_location(
        "ehull_pure_ar_baseline_reference", BASELINE_DIR / "baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = load_reference_baseline()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def trainable_state(model):
    return {name: param.detach().cpu().contiguous().clone()
            for name, param in model.named_parameters() if param.requires_grad}


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_sha256(model):
    return state_sha256(trainable_state(model))


def control_token_rows(model, token_ids):
    embedding = model.base.get_input_embeddings().token_adapter
    output_head = model.base.lm_head.base_layer
    ids = torch.tensor(token_ids, dtype=torch.long, device=embedding.base_layer.weight.device)
    return {
        "token_ids": ids.cpu(),
        "input_base_rows": embedding.base_layer.weight.detach().index_select(0, ids).cpu().clone(),
        "input_trainable_tokens_original": embedding.trainable_tokens_original["default"].detach().cpu().clone(),
        "output_base_rows": output_head.weight.detach().index_select(0, ids).cpu().clone(),
    }


def restore_control_token_rows(model, saved):
    embedding = model.base.get_input_embeddings().token_adapter
    output_head = model.base.lm_head.base_layer
    ids = saved["token_ids"].to(device=embedding.base_layer.weight.device)
    with torch.no_grad():
        embedding.base_layer.weight.index_copy_(
            0, ids, saved["input_base_rows"].to(embedding.base_layer.weight))
        embedding.trainable_tokens_original["default"].copy_(
            saved["input_trainable_tokens_original"].to(embedding.trainable_tokens_original["default"]))
        output_head.weight.index_copy_(0, ids, saved["output_base_rows"].to(output_head.weight))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def epoch_order(rows):
    ordered = list(rows)
    random.Random(SEED).shuffle(ordered)
    return ordered


def local_material_positions(update, rank, world_size):
    """Split each frozen block by rank without changing its global material order."""
    assert world_size in (1, 2) and MATERIALS_PER_UPDATE % world_size == 0
    first = (update - 1) * MATERIALS_PER_UPDATE
    return list(range(first + rank, first + MATERIALS_PER_UPDATE, world_size))


def learning_rate(update):
    warmup = math.ceil(0.05 * UPDATES)
    if update <= warmup:
        return LEARNING_RATE * update / warmup
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * (update - warmup) /
                                                (UPDATES - warmup)))


def load_tokenizer(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    added = tok.add_special_tokens({"additional_special_tokens": [START, END, MASK, PAD]})
    assert added == 4, f"expected four new control tokens, tokenizer added {added}"
    assert tok.eos_token == EOS
    token_ids = [tok.convert_tokens_to_ids(value) for value in (START, END, MASK, PAD)]
    assert len(set(token_ids)) == 4 and all(0 <= value < len(tok) for value in token_ids)
    return tok


def read_frozen_data(data_dir):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path, val_path = data_dir / "train-triples.jsonl.gz", data_dir / "val-triples.jsonl.gz"
    data_hashes = {
        "manifest_sha256": sha256_file(manifest_path),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "manifest_sources": manifest["sources"],
    }
    assert {key: data_hashes[key] for key in EXPECTED_DATA_HASHES} == EXPECTED_DATA_HASHES
    for name in ("train-triples.jsonl.gz", "val-triples.jsonl.gz"):
        assert sha256_file(data_dir / name) == manifest["artifacts"][name]["sha256"]
    train_rows = BASELINE.read_rows(train_path, "train")
    val_rows = BASELINE.read_rows(val_path, "val")
    assert len(train_rows) == manifest["counts"]["train"] == TRAIN_MATERIALS
    assert len(val_rows) == manifest["counts"]["val"] == VAL_MATERIALS
    train_ids = [row["material_id"] for row in train_rows]
    val_ids = [row["material_id"] for row in val_rows]
    assert len(set(train_ids)) == TRAIN_MATERIALS and set(train_ids).isdisjoint(val_ids)
    return manifest, data_hashes, train_rows, val_rows


def model_file_fingerprints(model_path):
    files = sorted(path for path in model_path.rglob("*") if path.is_file())
    assert files, f"no local model files under {model_path}"
    return {str(path.relative_to(model_path)): {"bytes": path.stat().st_size,
                                               "sha256": sha256_file(path)}
            for path in files}


def source_fingerprints():
    sources = {
        "train.py": EXPERIMENT / "train.py",
        "model.py": EXPERIMENT / "model.py",
        "prepare.py": EXPERIMENT / "prepare.py",
        "config.json": EXPERIMENT / "config.json",
        "baseline.py": BASELINE_DIR / "baseline.py",
        "baseline_model.py": BASELINE_DIR / "model.py",
        "baseline_prepare.py": BASELINE_DIR / "prepare.py",
        "shared_model.py": SHARED_DIR / "model.py",
        "shared_prepare.py": SHARED_DIR / "prepare.py",
        "strict_plaid_parser.py": OLD_MODEL_DIR / "score.py",
        "plaid_utils.py": ROOT / "work/PLaID/utils.py",
    }
    return {name: sha256_file(path) for name, path in sources.items()}


def runtime_versions():
    peft_version = importlib.metadata.version("peft")
    accelerate_version = importlib.metadata.version("accelerate")
    assert transformers.__version__ == "4.57.6"
    assert peft_version == "0.18.0"
    assert torch.__version__.split("+")[0] == "2.6.0"
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "peft": peft_version, "accelerate": accelerate_version}


def experiment_identity(model_path, revision, base_config, data_hashes, order_ids, eval_ids):
    model_files = model_file_fingerprints(model_path)
    model_config_sha256 = json_sha256(base_config)
    model_identity_sha256 = json_sha256({
        "model": MODEL, "revision": revision, "model_files": model_files,
        "model_config_sha256": model_config_sha256,
    })
    source_hashes = source_fingerprints()
    identity = {
        "experiment": EXPERIMENT.name,
        "model": MODEL,
        "revision": revision,
        "model_files": model_files,
        "model_config_sha256": model_config_sha256,
        "model_identity_sha256": model_identity_sha256,
        "data_hashes": data_hashes,
        "source_sha256": source_hashes,
        "sample_order_sha256": json_sha256(order_ids),
        "eval_ids_sha256": json_sha256(eval_ids),
        "training": {"updates": UPDATES, "materials_per_update": MATERIALS_PER_UPDATE,
                     "task_examples_per_update": TASK_EXAMPLES_PER_UPDATE,
                     "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                     "warmup_fraction": 0.05, "gradient_clip_norm": GRAD_CLIP_NORM,
                     "gradient_checkpointing": True, "world_size_cap": 2},
        "auxiliary_trainable_parameters": {
            "time.weight": {"shape": [STEPS + 1, base_config["hidden_size"]],
                             "dtype": "float32", "initialization": "zeros"}},
        "adapter": {"rank": 32, "alpha": 64, "dropout": 0.0,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                       "gate_proj", "up_proj", "down_proj", "lm_head"],
                    "input_token_ids": [START, END, MASK, PAD]},
    }
    identity["fingerprint_sha256"] = json_sha256(identity)
    return identity


def finalize_identity(identity, data_dir, model_path, tok):
    generation_config_path = model_path / "generation_config.json"
    generation_config = (json.loads(generation_config_path.read_text())
                         if generation_config_path.exists() else {})
    identity.update({
        "data_path": str(data_dir),
        "model_path": str(model_path),
        "token_ids": {token: tok.convert_tokens_to_ids(token)
                      for token in (START, END, MASK, PAD)},
        "tokenizer_eos_token": tok.eos_token,
        "tokenizer_eos_token_id": tok.eos_token_id,
        "generation_config_eos_token_id": generation_config.get("eos_token_id"),
    })
    identity["fingerprint_sha256"] = json_sha256({key: value for key, value in identity.items()
                                                   if key != "fingerprint_sha256"})
    return identity


def longest_training_row(rows, tok):
    longest = None
    longest_length = -1
    for row in rows:
        generation, reading = tasks(row)
        generation_length = (len(tok.encode(generation["prefix"] + generation["lead"],
                                             add_special_tokens=False)) + K)
        reading_length = len(tok.encode(reading["prefix"] + reading["target"],
                                        add_special_tokens=False))
        length = max(generation_length, reading_length)
        if length > longest_length:
            longest, longest_length = row, length
    return longest, longest_length


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    assert world_size in (1, 2), f"approved run uses one or two GPUs, got {world_size} ranks"
    assert torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, torch.device("cuda", local_rank)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def check_new_token_gradients(model, token_ids):
    input_grads = [param.grad[:2] for name, param in model.named_parameters()
                   if "trainable_tokens_delta" in name and param.grad is not None]
    active_output_ids = list(token_ids[:2])
    output_rows = [param.grad[active_output_ids] for name, param in model.named_parameters()
                   if "lm_head" in name and "lora_B" in name and param.grad is not None
                   and param.grad.ndim == 2 and param.grad.shape[0] > max(active_output_ids)]
    assert input_grads, "trainable input token rows received no gradients"
    assert output_rows, "lm_head LoRA rows received no gradients for added tokens"
    assert all(torch.isfinite(grad).all() for grad in input_grads + output_rows)
    input_norm = torch.sqrt(sum(grad.float().square().sum() for grad in input_grads))
    output_norm = torch.sqrt(sum(grad.float().square().sum() for grad in output_rows))
    assert input_norm > 0, "trainable input token gradients are zero"
    assert output_norm > 0, "lm_head added-token gradients are zero"
    return {"input_token_grad_norm": float(input_norm),
            "output_token_lm_head_lora_b_grad_norm": float(output_norm),
            "finite": True, "nonzero": True}


def local_rng_state(train_generator, include_cuda):
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if include_cuda else None,
        "training_generator": train_generator.get_state(),
    }


def gather_rng_states(train_generator, world_size, include_cuda):
    local = local_rng_state(train_generator, include_cuda)
    if world_size == 1:
        return [local]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered


def save_checkpoint(path, model, optimizer, identity, completed_updates, world_size,
                    rank, train_generator):
    rng_states = gather_rng_states(
        train_generator, world_size,
        include_cuda=next(model.parameters()).device.type == "cuda")
    if rank == 0:
        state = trainable_state(model)
        token_ids = list(identity["token_ids"].values())
        added_token_rows = control_token_rows(model, token_ids)
        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "fingerprint_sha256": identity["fingerprint_sha256"],
            "identity": identity,
            "trainable_state": state,
            "trainable_state_sha256": state_sha256(state),
            "added_token_base_rows": added_token_rows,
            "added_token_base_rows_sha256": state_sha256(added_token_rows),
            "optimizer_state": optimizer.state_dict(),
            "completed_updates": completed_updates,
            "next_material_index": completed_updates * MATERIALS_PER_UPDATE,
            "world_size": world_size,
            "rng_states": rng_states,
        }
        atomic_torch_save(path, checkpoint)
        write_json(Path(path).parent / "progress.json", {
            "status": "checkpointed", "completed_updates": completed_updates,
            "next_material_index": checkpoint["next_material_index"],
            "checkpoint_bytes": Path(path).stat().st_size,
            "trainable_state_sha256": checkpoint["trainable_state_sha256"],
        })
    if world_size > 1:
        dist.barrier()


def restore_checkpoint(path, model, optimizer, identity, rank, world_size,
                       train_generator, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["schema"] == CHECKPOINT_SCHEMA
    assert checkpoint["fingerprint_sha256"] == identity["fingerprint_sha256"]
    assert checkpoint["world_size"] == world_size
    assert checkpoint["next_material_index"] == checkpoint["completed_updates"] * MATERIALS_PER_UPDATE
    assert state_sha256(checkpoint["added_token_base_rows"]) == checkpoint["added_token_base_rows_sha256"]
    restore_control_token_rows(model, checkpoint["added_token_base_rows"])
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    saved = checkpoint["trainable_state"]
    assert set(params) == set(saved), "checkpoint trainable parameter set changed"
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved[name].to(device=param.device, dtype=param.dtype))
    assert trainable_sha256(model) == checkpoint["trainable_state_sha256"]
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    rng = checkpoint["rng_states"][rank]
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch_cpu"])
    if rng["torch_cuda"] is not None:
        torch.cuda.set_rng_state(rng["torch_cuda"])
    train_generator.set_state(rng["training_generator"])
    return checkpoint


def assert_preflight_receipt(path, identity):
    receipt = json.loads(Path(path).read_text())
    assert receipt["status"] == "preflight_complete"
    assert receipt["fingerprint_sha256"] == identity["fingerprint_sha256"]


def prepare_output(out, resume):
    out = Path(out)
    if resume:
        assert out.is_dir()
    else:
        out.mkdir(parents=True, exist_ok=False)
    return out


def initialize_run_output(out, resume, rank):
    out = Path(out).resolve()
    assert out.is_relative_to(ROOT.resolve())
    if rank == 0:
        prepare_output(out, resume)
    return out


def validate_run_output(out, resume, rank):
    out = Path(out).resolve()
    assert out.is_relative_to(ROOT.resolve())
    if rank == 0:
        assert out.is_dir() if resume else not out.exists()
    return out


def trim_update_log(path, completed_updates):
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_text()
    lines = raw.splitlines()
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    kept = [line for line in lines if json.loads(line)["update"] <= completed_updates]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write("".join(line + "\n" for line in kept))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def update_record(model, ddp_model, optimizer, ordered, update, rank, world_size,
                  train_generator, tok):
    lr = learning_rate(update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    positions = local_material_positions(update, rank, world_size)
    local_count = len(positions)
    losses, text_losses, crystal_losses = [], [], []
    for local_index, material_index in enumerate(positions):
        sync = local_index == local_count - 1 or world_size == 1
        context = nullcontext() if sync else ddp_model.no_sync()
        with context:
            with autocast():
                loss, stats = ddp_model(tasks(ordered[material_index]),
                                        tok, "ar", train_generator)
                assert torch.isfinite(loss), "nonfinite AR loss"
                (loss / local_count).backward()
        losses.append(loss.detach())
        text_losses.append(stats["text"])
        crystal_losses.append(stats["crystal"])
    parameters = [param for param in model.parameters() if param.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP_NORM)
    assert torch.isfinite(grad_norm), "nonfinite trainable gradient norm"
    optimizer.step()
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.stack((
        torch.stack(losses).mean().to(torch.float64),
        torch.tensor(sum(text_losses) / local_count, device=device, dtype=torch.float64),
        torch.tensor(sum(crystal_losses) / local_count, device=device, dtype=torch.float64),
        grad_norm.detach().to(torch.float64),
    ))
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values[:3] /= world_size
        values[3] /= world_size
    torch.cuda.synchronize()
    entry = {
        "update": update,
        "learning_rate": lr,
        "loss": float(values[0]),
        "text_loss": float(values[1]),
        "crystal_loss": float(values[2]),
        "grad_norm": float(values[3]),
        "finite_gradients": True,
        "first_material_index": (update - 1) * MATERIALS_PER_UPDATE,
        "next_material_index": update * MATERIALS_PER_UPDATE,
        "seconds": None,
    }
    return entry


def make_run_state(model_path, tok, train_rows, val_rows, world_size, device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = Qwen3ARBaseline.pretrained(tok, str(model_path)).to(device).train()
    train_generator = torch.Generator().manual_seed(TRAINING_GENERATOR_SEED)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=False)
        bare_model = model.module
    else:
        bare_model = model
    ordered = epoch_order(train_rows)
    order_ids = [row["material_id"] for row in ordered]
    eval_ids = BASELINE.select_eval_ids(val_rows)
    assert json_sha256(order_ids) == EXPECTED_ORDER_SHA256
    assert json_sha256(eval_ids) == EXPECTED_EVAL_IDS_SHA256
    return model, bare_model, optimizer, train_generator, ordered, order_ids, eval_ids


def record_preflight_gradient(ddp_model, model, tok, row, token_ids, world_size):
    ddp_model.train()
    ddp_model.zero_grad(set_to_none=True)
    with autocast():
        loss, stats = ddp_model(tasks(row), tok, "ar",
                                torch.Generator().manual_seed(TRAINING_GENERATOR_SEED))
    assert torch.isfinite(loss)
    loss.backward()
    gradient_report = check_new_token_gradients(model, token_ids)
    gradient_report.update({"loss": float(loss.detach()), "text_loss": stats["text"],
                            "crystal_loss": stats["crystal"], "material_id": row["material_id"]})
    ddp_model.zero_grad(set_to_none=True)
    if world_size > 1:
        dist.barrier()
    return gradient_report


def run_training(args, mode):
    model_path = args.model_path.resolve(strict=True)
    data_dir = args.data.resolve(strict=True)
    assert model_path.is_dir() and data_dir.is_dir()
    assert model_path.is_relative_to(ROOT.resolve()) and data_dir.is_relative_to(ROOT.resolve())
    assert args.model_revision == REVISION, f"expected pinned revision {REVISION}"
    versions = runtime_versions()
    rank, world_size, device = init_distributed()
    out = validate_run_output(args.out, args.resume, rank)
    if world_size > 1:
        dist.barrier()
    torch.set_num_threads(4)
    tok = load_tokenizer(model_path)
    manifest, data_hashes, train_rows, val_rows = read_frozen_data(data_dir)
    lengths = BASELINE.validate_lengths(train_rows + val_rows, tok)
    assert lengths["rows_checked"] == manifest["total"] == 8172
    longest_row, longest_length = longest_training_row(train_rows, tok)
    model, bare_model, optimizer, train_generator, ordered, order_ids, eval_ids = make_run_state(
        model_path, tok, train_rows, val_rows, world_size, device)
    identity = None
    if rank == 0:
        identity = finalize_identity(
            experiment_identity(model_path, args.model_revision, bare_model.base.config.to_dict(),
                                data_hashes, order_ids, eval_ids), data_dir, model_path, tok)
    if world_size > 1:
        identities = [identity]
        dist.broadcast_object_list(identities, src=0)
        identity = identities[0]
    if mode == "train" and args.resume is None:
        assert args.preflight_receipt, "formal training requires a matching preflight receipt"
        assert_preflight_receipt(args.preflight_receipt, identity)

    if args.resume is not None:
        assert args.resume.resolve().is_relative_to(ROOT.resolve())
    if args.preflight_receipt is not None:
        assert args.preflight_receipt.resolve().is_relative_to(ROOT.resolve())
    out = initialize_run_output(out, args.resume, rank)
    checkpoint_path = out / "checkpoint.pt"
    if world_size > 1:
        dist.barrier()
    if rank == 0 and args.resume is None:
        BASELINE.write_json(out / "evaluation-ids.json", {
            "seed": SEED, "material_ids": eval_ids, "sha256": json_sha256(eval_ids)})
        BASELINE.write_json(out / "sample-order.json", {
            "seed": SEED, "material_ids": order_ids, "sha256": json_sha256(order_ids)})
        BASELINE.write_json(out / "identity.json", identity)
        BASELINE.write_json(out / "setup.json", {
            "status": "preflight_started" if mode == "preflight" else "training_started",
            "mode": mode, "model": MODEL, "revision": args.model_revision,
            "model_path": str(model_path), "model_identity_sha256": identity["model_identity_sha256"],
            "initial_trainable_state_sha256": trainable_sha256(bare_model),
            "auxiliary_trainable_parameters": {
                "time.weight": {"shape": list(bare_model.time.weight.shape),
                                 "dtype": str(bare_model.time.weight.dtype),
                                 "initialization": "zeros"}},
            "base_config_sha256": identity["model_config_sha256"],
            "data": data_hashes, "source_sha256": identity["source_sha256"],
            "fingerprint_sha256": identity["fingerprint_sha256"],
            "seed": SEED, "training_generator_seed": TRAINING_GENERATOR_SEED,
            "evaluation_generator_seed_base": EVALUATION_GENERATOR_SEED,
            "train_materials": TRAIN_MATERIALS, "val_materials": VAL_MATERIALS,
            "microbatch_tasks": MICRO_BATCH_TASKS, "accumulation_steps": MATERIALS_PER_UPDATE,
            "effective_task_examples": TASK_EXAMPLES_PER_UPDATE, "updates": UPDATES,
            "warmup_updates": math.ceil(0.05 * UPDATES), "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "precision": "BF16 frozen base; FP32 LoRA and trainable tokens",
            "gradient_checkpointing": True, "world_size": world_size,
            "checkpoint_interval_updates": CHECKPOINT_EVERY, "checkpoint_retention": 1,
            "training_stream": "custom User:/Assistant: plain text; no HF chat template or think filtering",
            "lengths": lengths, "longest_preflight_material_id": longest_row["material_id"],
            "longest_preflight_length": longest_length, "software": versions,
        })
    if world_size > 1:
        dist.barrier()

    if args.resume:
        checkpoint = restore_checkpoint(args.resume, bare_model, optimizer, identity,
                                        rank, world_size, train_generator, device)
        completed = checkpoint["completed_updates"]
    else:
        completed = 0

    token_ids = [tok.convert_tokens_to_ids(value) for value in (START, END, MASK, PAD)]
    gradient_report = None
    if mode == "preflight":
        gradient_report = record_preflight_gradient(model, bare_model, tok, longest_row,
                                                    token_ids, world_size)
        target_update = PREFLIGHT_UPDATES
    else:
        target_update = args.stop_after_update if args.stop_after_update is not None else UPDATES
        assert completed <= target_update <= UPDATES and target_update >= 1

    if args.resume and rank == 0:
        trim_update_log(out / "updates.jsonl", completed)
    update_log = (out / "updates.jsonl").open("a" if args.resume else "w") if rank == 0 else None
    torch.cuda.reset_peak_memory_stats(device)
    training_start = time.monotonic()
    for update in range(completed + 1, target_update + 1):
        update_start = time.monotonic()
        entry = update_record(bare_model, model, optimizer, ordered, update, rank, world_size,
                              train_generator, tok)
        entry["seconds"] = time.monotonic() - update_start
        if rank == 0:
            update_log.write(json.dumps(entry, allow_nan=False) + "\n")
            update_log.flush()
            print(json.dumps(entry, allow_nan=False), flush=True)
        should_save = update % CHECKPOINT_EVERY == 0 or update == target_update
        if should_save:
            save_checkpoint(checkpoint_path, bare_model, optimizer, identity, update,
                            world_size, rank, train_generator)
    if update_log:
        update_log.close()
    training_seconds = time.monotonic() - training_start

    max_memory = torch.tensor(torch.cuda.max_memory_allocated(device), device=device, dtype=torch.long)
    if world_size > 1:
        dist.all_reduce(max_memory, op=dist.ReduceOp.MAX)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        del model, bare_model, optimizer
        torch.cuda.empty_cache()
        return

    if mode == "preflight":
        reload_report = None
        if rank == 0:
            before_hash = trainable_sha256(bare_model)
            bare_model.eval()
            prefix = tasks(longest_row)[0]["prefix"]
            with torch.inference_mode(), autocast():
                before_logits = bare_model.logits(
                    torch.tensor(tok.encode(prefix, add_special_tokens=False), device=device),
                    "ar", positions=[len(tok.encode(prefix, add_special_tokens=False)) - 1])[0].float()
            del model, bare_model, optimizer
            torch.cuda.empty_cache()
            random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            reloaded = Qwen3ARBaseline.pretrained(tok, str(model_path)).to(device).eval()
            reloaded_optimizer = torch.optim.AdamW(
                (param for param in reloaded.parameters() if param.requires_grad),
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            restore_checkpoint(checkpoint_path, reloaded, reloaded_optimizer, identity,
                               rank, world_size, train_generator, device)
            assert trainable_sha256(reloaded) == before_hash
            with torch.inference_mode(), autocast():
                after_logits = reloaded.logits(
                    torch.tensor(tok.encode(prefix, add_special_tokens=False), device=device),
                    "ar", positions=[len(tok.encode(prefix, add_special_tokens=False)) - 1])[0].float()
            max_logit_delta = float((after_logits - before_logits).abs().max())
            torch.testing.assert_close(after_logits, before_logits, rtol=2e-3, atol=2e-3)
            reload_report = {"adapter_state_sha256_before_save": before_hash,
                             "adapter_state_sha256_after_reload": trainable_sha256(reloaded),
                             "control_logits_equivalent": True,
                             "max_control_logit_delta": max_logit_delta,
                             "optimizer_state_reloaded": True,
                             "checkpoint_bytes": checkpoint_path.stat().st_size}
        if rank == 0:
            summary = {
                "status": "preflight_complete", "learning_trial": False,
                "property_evaluation_status": "not_run", "model": MODEL,
                "revision": args.model_revision,
                "fingerprint_sha256": identity["fingerprint_sha256"],
                "model_identity_sha256": identity["model_identity_sha256"],
                "data": data_hashes, "source_sha256": identity["source_sha256"],
                "seed": SEED, "updates": PREFLIGHT_UPDATES,
                "next_material_index": PREFLIGHT_UPDATES * MATERIALS_PER_UPDATE,
                "world_size": world_size, "longest_material_id": longest_row["material_id"],
                "longest_example_tokens": longest_length, "gradient_check": gradient_report,
                "save_reload": reload_report, "training_seconds": training_seconds,
                "peak_gpu_memory_mb": round(int(max_memory) / 2**20, 2), "software": versions,
            }
            write_json(out / "summary.json", summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
    elif target_update == UPDATES:
        if rank == 0:
            summary = {
                "status": "one_epoch_training_complete", "learning_trial": True,
                "property_evaluation_status": "not_run", "mode": "pure_ar",
                "model": MODEL, "revision": args.model_revision,
                "fingerprint_sha256": identity["fingerprint_sha256"],
                "model_identity_sha256": identity["model_identity_sha256"],
                "seed": SEED, "updates": UPDATES, "completed_updates": UPDATES,
                "next_material_index": TRAIN_MATERIALS,
                "microbatch_tasks": MICRO_BATCH_TASKS,
                "accumulation_steps": MATERIALS_PER_UPDATE,
                "effective_task_examples": TASK_EXAMPLES_PER_UPDATE,
                "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                "initial_trainable_state_sha256": json.loads((out / "setup.json").read_text())[
                    "initial_trainable_state_sha256"],
                "final_trainable_state_sha256": trainable_sha256(bare_model),
                "data": data_hashes, "source_sha256": identity["source_sha256"],
                "sample_order_sha256": json_sha256(order_ids),
                "evaluation_ids": eval_ids, "evaluation_ids_sha256": json_sha256(eval_ids),
                "lengths": lengths, "training_seconds": training_seconds,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "peak_gpu_memory_mb": round(int(max_memory) / 2**20, 2),
                "world_size": world_size, "gpu": torch.cuda.get_device_name(device),
                "evaluation": "pending single-GPU evaluate mode", "software": versions,
            }
            write_json(out / "summary.json", summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
        return
    else:
        if rank == 0:
            write_json(out / "summary.json", {
                "status": "paused", "learning_trial": False,
                "property_evaluation_status": "not_run", "model": MODEL,
                "revision": args.model_revision,
                "fingerprint_sha256": identity["fingerprint_sha256"],
                "completed_updates": target_update,
                "next_material_index": target_update * MATERIALS_PER_UPDATE,
                "checkpoint": str(checkpoint_path), "data": data_hashes,
                "source_sha256": identity["source_sha256"], "software": versions,
            })
def run_evaluation(args):
    model_path = args.model_path.resolve(strict=True)
    data_dir = args.data.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    assert model_path.is_relative_to(ROOT.resolve()) and data_dir.is_relative_to(ROOT.resolve())
    assert checkpoint_path.is_relative_to(ROOT.resolve())
    assert args.model_revision == REVISION
    versions = runtime_versions()
    rank, world_size, device = init_distributed()
    assert world_size == 1 and rank == 0, "32-sample evaluation uses one GPU"
    tok = load_tokenizer(model_path)
    manifest, data_hashes, train_rows, val_rows = read_frozen_data(data_dir)
    model = Qwen3ARBaseline.pretrained(tok, str(model_path)).to(device).eval()
    ordered = epoch_order(train_rows)
    eval_ids = BASELINE.select_eval_ids(val_rows)
    identity = finalize_identity(
        experiment_identity(model_path, args.model_revision, model.base.config.to_dict(),
                            data_hashes, [row["material_id"] for row in ordered], eval_ids),
        data_dir, model_path, tok)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["fingerprint_sha256"] == identity["fingerprint_sha256"]
    assert state_sha256(saved["added_token_base_rows"]) == saved["added_token_base_rows_sha256"]
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    assert set(params) == set(saved["trainable_state"])
    restore_control_token_rows(model, saved["added_token_base_rows"])
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved["trainable_state"][name].to(device=param.device, dtype=param.dtype))
    assert trainable_sha256(model) == saved["trainable_state_sha256"]
    out = prepare_output(args.out.resolve(), False)
    assert out.is_relative_to(ROOT.resolve())
    write_json(out / "identity.json", identity)
    write_json(out / "evaluation-ids.json", {
        "seed": SEED, "material_ids": eval_ids, "sha256": json_sha256(eval_ids)})
    val_counts, evaluation_seconds = BASELINE.evaluate(model, tok, val_rows, eval_ids, out)
    write_json(out / "summary.json", {
        "status": "evaluation_complete", "property_evaluation_status": "not_run",
        "model": MODEL, "revision": args.model_revision,
        "checkpoint": str(checkpoint_path), "fingerprint_sha256": identity["fingerprint_sha256"],
        "data": data_hashes, "evaluation_ids": eval_ids,
        "evaluation_ids_sha256": json_sha256(eval_ids), "evaluation_seconds": evaluation_seconds,
        "val": val_counts, "software": versions,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "train", "evaluate"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", default=REVISION)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    if args.mode == "evaluate":
        assert args.checkpoint is not None
        run_evaluation(args)
    else:
        assert args.checkpoint is None
        assert args.resume is None or args.mode == "train"
        assert args.preflight_receipt is None or args.mode == "train"
        assert args.stop_after_update is None or args.mode == "train"
        assert args.mode != "preflight" or args.resume is None
        run_training(args, args.mode)


if __name__ == "__main__":
    main()
