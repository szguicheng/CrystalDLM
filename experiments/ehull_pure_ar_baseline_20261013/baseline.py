"""Run one fixed pure-AR epoch on the frozen 7,272-material ehull subset."""
import argparse
from decimal import Decimal
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OLD_EXPERIMENT = ROOT / "experiments/interleaved_ar_diffusion_20261002"
SHARED_EXPERIMENT = ROOT / "experiments/ehull_interleaved_smoke_20261013"
os.environ.setdefault("HF_HOME", str(OLD_EXPERIMENT / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import transformers
from transformers import AutoTokenizer

from model import ARBaseline
from prepare import END, EOS, K, MASK, MAXLEN, MODEL, PAD, READ_QUESTION, REVISION, START, STEPS, tasks

SEED = 42
MICRO_BATCH_TASKS = 2
ACCUMULATION = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EVAL_COUNT = 32
DATA_FIELDS = ("cell_formula", "space_group_number", "cell_atom_count", "independent_orbit_count")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def epoch_order(rows, seed=SEED):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    return ordered


def select_eval_ids(rows, seed=SEED):
    assert len(rows) >= EVAL_COUNT
    return random.Random(seed).sample(sorted(row["material_id"] for row in rows), EVAL_COUNT)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tok.add_special_tokens({"additional_special_tokens": [START, END, MASK, PAD]})
    assert tok.eos_token == EOS
    return tok


def read_rows(path, split):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            raw = json.loads(line, parse_float=Decimal)
            assert raw["split"] == split
            rows.append({
                "material_id": raw["material_id"], "split": split,
                "plaid_text": raw["plaid_text"],
                "structure_description": raw["structure_description"],
                "property_labels": {key: raw["property_labels"][key]
                                    for key in ("energy_above_hull", "unit")},
                "core_facts": {key: raw["core_facts"][key] for key in DATA_FIELDS},
            })
    return rows


def validate_lengths(rows, tok):
    generation_max, crystal_max, reading_max = 0, 0, 0
    for row in rows:
        generation, reading = tasks(row)
        crystal_length = len(tok.encode(generation["crystal"] + END, add_special_tokens=False,
                                         truncation=False))
        generation_length = len(tok.encode(generation["prefix"] + generation["lead"],
                                           add_special_tokens=False, truncation=False)) + K
        reading_length = len(tok.encode(reading["prefix"] + reading["target"],
                                        add_special_tokens=False, truncation=False))
        assert crystal_length <= K, f"{row['material_id']} crystal exceeds K: {crystal_length}"
        assert generation_length <= MAXLEN, f"{row['material_id']} generation exceeds MAXLEN: {generation_length}"
        assert reading_length <= MAXLEN, f"{row['material_id']} reading exceeds MAXLEN: {reading_length}"
        generation_max = max(generation_max, generation_length)
        crystal_max = max(crystal_max, crystal_length)
        reading_max = max(reading_max, reading_length)
    return {"rows_checked": len(rows), "generation_max_tokens": generation_max,
            "crystal_max_tokens_including_end": crystal_max,
            "reading_max_tokens": reading_max, "truncation": False}


def learning_rate(update, total_updates):
    warmup = math.ceil(0.05 * total_updates)
    if update <= warmup:
        return LEARNING_RATE * update / warmup
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * (update - warmup) /
                                                (total_updates - warmup)))


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def check_finite_grads(model):
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), "nonfinite gradient"


def source_hashes():
    paths = {
        "baseline.py": Path(__file__),
        "model.py": Path(__file__).with_name("model.py"),
        "prepare.py": Path(__file__).with_name("prepare.py"),
        "shared_model.py": SHARED_EXPERIMENT / "model.py",
        "shared_prepare.py": SHARED_EXPERIMENT / "prepare.py",
        "strict_plaid_parser.py": OLD_EXPERIMENT / "score.py",
        "plaid_utils.py": ROOT / "work/PLaID/utils.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def initial_route_check(model, tok, row):
    model.train()
    model.zero_grad(set_to_none=True)
    with autocast():
        loss, stats = model.loss(tasks(row), tok, "ar", torch.Generator().manual_seed(SEED + 1000))
    assert torch.isfinite(loss), "nonfinite initial AR route loss"
    loss.backward()
    check_finite_grads(model)
    result = {"loss": float(loss.detach()), "text_loss": stats["text"],
              "crystal_loss": stats["crystal"], "finite_gradients": True}
    model.zero_grad(set_to_none=True)
    return result


def failure(error):
    result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if hasattr(error, "raw_text"):
        result["raw"] = error.raw_text
        result["tokens"] = error.raw_tokens
    return result


def short_audit(audit):
    return {key: audit[key] for key in ("valid", "reasons", "decoded_atoms", "detected_sg")
            if key in audit}


def evaluate(model, tok, val_rows, eval_ids, out):
    sys.path.insert(0, str(OLD_EXPERIMENT))
    from score import audit

    by_id = {row["material_id"]: row for row in val_rows}
    counts = {"val_generation_attempts": len(eval_ids), "lead_start_failures": 0,
              "lead_eos_before_start": 0, "lead_limit_failures": 0,
              "crystal_attempts": 0, "crystal_block_failures": 0,
              "crystal_end_failures": 0, "crystal_pad_boundary_failures": 0,
              "parse_valid_crystals": 0, "parse_invalid_crystals": 0, "tail_read_attempts": 0,
              "tail_read_completed": 0, "tail_read_eos_failures": 0}
    started = time.monotonic()
    with (out / "evaluation.jsonl").open("w") as log:
        for index, material_id in enumerate(eval_ids):
            row = by_id[material_id]
            generation, _ = tasks(row)
            result = {"material_id": material_id, "crystal_generator_seed": SEED + 2000 + index}
            try:
                with autocast():
                    lead = model.text(generation["prefix"], tok, "ar", trigger=True, limit=512)
                result["lead"] = {"ok": True, "raw": lead}
            except Exception as error:
                counts["lead_start_failures"] += 1
                raw_tokens = getattr(error, "raw_tokens", [])
                if raw_tokens and raw_tokens[-1] == tok.eos_token_id:
                    counts["lead_eos_before_start"] += 1
                else:
                    counts["lead_limit_failures"] += 1
                result["lead"] = failure(error)
                log.write(json.dumps(result, allow_nan=False) + "\n"); log.flush()
                continue

            counts["crystal_attempts"] += 1
            try:
                with autocast():
                    crystal, tokens = model.crystal(
                        generation["prefix"] + lead, tok, "ar",
                        torch.Generator(device="cuda").manual_seed(SEED + 2000 + index))
                result["crystal"] = {"ok": True, "raw": crystal, "tokens": tokens}
            except Exception as error:
                counts["crystal_block_failures"] += 1
                raw_tokens = getattr(error, "raw_tokens", [])
                end_positions = [i for i, value in enumerate(raw_tokens) if value == model.end_id]
                if len(end_positions) != 1:
                    counts["crystal_end_failures"] += 1
                elif model.pad_id in raw_tokens[:end_positions[0]] or any(
                        value != model.pad_id for value in raw_tokens[end_positions[0] + 1:]):
                    counts["crystal_pad_boundary_failures"] += 1
                result["crystal"] = failure(error)
                log.write(json.dumps(result, allow_nan=False) + "\n"); log.flush()
                continue

            checked = audit(crystal)
            result["crystal_audit"] = short_audit(checked)
            if checked["valid"]:
                counts["parse_valid_crystals"] += 1
                read_prefix = (generation["prefix"] + lead + crystal + END + "\nUser: "
                               + READ_QUESTION + "\nAssistant:")
                counts["tail_read_attempts"] += 1
                try:
                    with autocast():
                        raw = model.text(read_prefix, tok, "ar", trigger=False, limit=512)
                    result["tail_read"] = {"ok": True, "raw": raw}
                    counts["tail_read_completed"] += 1
                except Exception as error:
                    if getattr(error, "raw_tokens", []) and error.raw_tokens[-1] != tok.eos_token_id:
                        counts["tail_read_eos_failures"] += 1
                    result["tail_read"] = failure(error)
            else:
                counts["parse_invalid_crystals"] += 1
            log.write(json.dumps(result, allow_nan=False) + "\n"); log.flush()
    torch.cuda.synchronize()
    return counts, time.monotonic() - started


def run(data, out):
    assert torch.cuda.is_available(), "Pure-AR training requires an allocated GPU"
    assert os.environ.get("SLURM_JOB_ID"), "Training must run inside a Slurm allocation"
    torch.set_num_threads(4)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    train_path, val_path = data / "train-triples.jsonl.gz", data / "val-triples.jsonl.gz"
    for name, path in ((train_path.name, train_path), (val_path.name, val_path)):
        assert sha256_file(path) == manifest["artifacts"][name]["sha256"], f"{name} differs from manifest"
    train_rows, val_rows = read_rows(train_path, "train"), read_rows(val_path, "val")
    assert len(train_rows) == manifest["counts"]["train"] == 7272
    assert len(val_rows) == manifest["counts"]["val"] == 900
    train_ids = [row["material_id"] for row in train_rows]
    val_ids = [row["material_id"] for row in val_rows]
    assert len(set(train_ids)) == len(train_ids) and set(train_ids).isdisjoint(val_ids)

    tok = load_tokenizer()
    lengths = validate_lengths(train_rows + val_rows, tok)
    assert lengths["rows_checked"] == manifest["total"] == 8172
    ordered = epoch_order(train_rows)
    order_ids = [row["material_id"] for row in ordered]
    assert len(order_ids) == 7272 and len(set(order_ids)) == 7272 and set(order_ids) == set(train_ids)
    eval_ids = select_eval_ids(val_rows)
    updates = len(ordered) // ACCUMULATION
    assert len(ordered) % ACCUMULATION == 0 and updates == 909

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "evaluation-ids.json", {"seed": SEED, "material_ids": eval_ids,
                                               "sha256": json_sha256(eval_ids)})
    write_json(out / "sample-order.json", {"seed": SEED, "material_ids": order_ids,
                                            "sha256": json_sha256(order_ids)})
    data_hashes = {"manifest_sha256": sha256_file(manifest_path),
                   "train_sha256": sha256_file(train_path), "val_sha256": sha256_file(val_path),
                   "manifest_sources": manifest["sources"]}
    src_hashes = source_hashes()

    # Re-seed immediately before construction so the baseline and later hybrid arm match exactly.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = ARBaseline.pretrained(tok).cuda().train()
    initial_hash = state_sha256(model.state_dict())
    warmup_check = initial_route_check(model, tok, train_rows[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_rng = torch.Generator().manual_seed(SEED + 1000)
    base_config = model.base.config.to_dict()
    config = {
        "mode": "ar", "prediction_offset": 0, "seed": SEED,
        "training_generator_seed": SEED + 1000,
        "evaluation_generator_seed_base": SEED + 2000,
        "model": MODEL, "revision": REVISION,
        "capacity": K, "steps": STEPS, "max_length": MAXLEN,
        "token_ids": {token: tok.convert_tokens_to_ids(token) for token in (START, END, MASK, PAD)},
        "base_config": base_config, "initial_weights_sha256": initial_hash,
        "data_hashes": data_hashes, "source_sha256": src_hashes,
        "train_order_sha256": json_sha256(order_ids), "eval_ids_sha256": json_sha256(eval_ids),
    }
    config_hash = json_sha256(base_config)
    setup = {
        "status": "training_started", "seed": SEED, "mode": "pure_ar",
        "model": MODEL, "revision": REVISION, "prediction_offset": 0,
        "train_materials": len(train_rows), "val_materials": len(val_rows),
        "microbatch_tasks": MICRO_BATCH_TASKS, "accumulation_steps": ACCUMULATION,
        "effective_material_pairs": ACCUMULATION, "effective_task_examples": MICRO_BATCH_TASKS * ACCUMULATION,
        "updates": updates, "warmup_updates": math.ceil(0.05 * updates),
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "precision": "fp32 weights with bf16 autocast", "noise": "none; all text and crystal targets causal",
        "training_generator_seed": SEED + 1000,
        "evaluation_generator_seed_base": SEED + 2000,
        "initial_weights_sha256": initial_hash, "initial_route_gradient_check": warmup_check,
        "data": data_hashes, "source_sha256": src_hashes,
        "sample_order_sha256": json_sha256(order_ids),
        "exposure": {"unique_materials": len(set(order_ids)), "epoch_materials": len(order_ids),
                     "each_material_once": True},
        "evaluation_ids_sha256": json_sha256(eval_ids), "evaluation_ids_saved_before_training": True,
        "lengths": lengths,
        "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    write_json(out / "setup.json", setup)

    torch.cuda.reset_peak_memory_stats()
    training_start = time.monotonic()
    with (out / "updates.jsonl").open("w") as log:
        for update in range(1, updates + 1):
            update_start = time.monotonic()
            lr = learning_rate(update, updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses, text_losses, crystal_losses = [], [], []
            first = (update - 1) * ACCUMULATION
            for index in range(first, first + ACCUMULATION):
                examples = tasks(ordered[index])
                assert len(examples) == MICRO_BATCH_TASKS and [item["task"] for item in examples] == ["G", "Q"]
                with autocast():
                    loss, stats = model.loss(examples, tok, "ar", train_rng)
                assert torch.isfinite(loss), "nonfinite AR objective"
                (loss / ACCUMULATION).backward()
                losses.append(loss.detach())
                text_losses.append(stats["text"])
                crystal_losses.append(stats["crystal"])
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(grad_norm), "nonfinite gradient norm"
            optimizer.step()
            torch.cuda.synchronize()
            entry = {"update": update, "learning_rate": lr,
                     "loss": float(torch.stack(losses).mean()),
                     "text_loss": sum(text_losses) / len(text_losses),
                     "crystal_loss": sum(crystal_losses) / len(crystal_losses),
                     "grad_norm": float(grad_norm), "finite_gradients": True,
                     "first_material_index": first, "next_material_index": first + ACCUMULATION,
                     "seconds": time.monotonic() - update_start}
            log.write(json.dumps(entry, allow_nan=False) + "\n"); log.flush()
            print(json.dumps(entry, allow_nan=False), flush=True)

    training_seconds = time.monotonic() - training_start
    config["base_config_sha256"] = config_hash
    checkpoint = out / "checkpoint.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
        "completed_updates": updates, "next_material_index": len(ordered),
        "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(), "training_generator_state": train_rng.get_state(),
    }, temporary)
    temporary.replace(checkpoint)

    model.eval()
    val_counts, evaluation_seconds = evaluate(model, tok, val_rows, eval_ids, out)
    weights_hash = state_sha256(model.state_dict())
    summary = {
        "status": "one_epoch_complete",
        "learning_trial": True, "property_evaluation_status": "not_run",
        "seed": SEED, "updates": updates, "mode": "pure_ar", "prediction_offset": 0,
        "training_generator_seed": SEED + 1000,
        "evaluation_generator_seed_base": SEED + 2000,
        "microbatch_tasks": MICRO_BATCH_TASKS, "accumulation_steps": ACCUMULATION,
        "effective_material_pairs": ACCUMULATION, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "model": MODEL, "revision": REVISION,
        "initial_weights_sha256": initial_hash, "final_weights_sha256": weights_hash,
        "base_config_sha256": config_hash, "data": data_hashes, "source_sha256": src_hashes,
        "sample_order_sha256": json_sha256(order_ids), "exposure": setup["exposure"],
        "evaluation_ids": eval_ids, "evaluation_ids_sha256": json_sha256(eval_ids),
        "route_gradient_check": warmup_check, "lengths": lengths,
        "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "completed_updates": updates, "next_material_index": len(ordered),
        "finite_gradient_norms": True, "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "gpu": torch.cuda.get_device_name(), "val": val_counts,
        "software": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "transformers": transformers.__version__},
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
