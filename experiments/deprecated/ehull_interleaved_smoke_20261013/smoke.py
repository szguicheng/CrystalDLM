"""Run the bounded GPU smoke check for the shared AR/PLaID diffusion model."""
import argparse
from decimal import Decimal
import gc
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
os.environ.setdefault("HF_HOME", str(OLD_EXPERIMENT / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer

from model import Hybrid
from prepare import END, EOS, K, MASK, MAXLEN, MODEL, PAD, PREDICTION_OFFSET, READ_QUESTION, REVISION, START, STEPS, tasks

DATA_FIELDS = ("cell_formula", "space_group_number", "cell_atom_count", "independent_orbit_count")
UPDATES, SEED, MICRO_BATCH, LEARNING_RATE = 20, 42, 2, 2e-5


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
                "material_id": raw["material_id"],
                "split": split,
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
        crystal_length = len(tok.encode(generation["crystal"] + END,
                                         add_special_tokens=False, truncation=False))
        generation_length = len(tok.encode(generation["prefix"] + generation["lead"],
                                           add_special_tokens=False, truncation=False)) + K
        reading_length = len(tok.encode(reading["prefix"] + reading["target"],
                                        add_special_tokens=False, truncation=False))
        assert crystal_length <= K, f"{row['material_id']} crystal block exceeds capacity: {crystal_length}"
        assert generation_length <= MAXLEN, f"{row['material_id']} generation exceeds context: {generation_length}"
        assert reading_length <= MAXLEN, f"{row['material_id']} reading exceeds context: {reading_length}"
        generation_max = max(generation_max, generation_length)
        crystal_max = max(crystal_max, crystal_length)
        reading_max = max(reading_max, reading_length)
    return {"rows_checked": len(rows), "generation_max_tokens": generation_max,
            "crystal_max_tokens_including_end": crystal_max,
            "reading_max_tokens": reading_max, "truncation": False}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=str).encode()).hexdigest()


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


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def short_audit(audit):
    return {key: audit[key] for key in ("valid", "reasons", "decoded_atoms", "detected_sg") if key in audit}


def failure(error):
    result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if hasattr(error, "raw_text"):
        result["raw"] = error.raw_text
        result["tokens"] = error.raw_tokens
    return result


def evaluate(model, tok, val_rows, out):
    sys.path.insert(0, str(OLD_EXPERIMENT))
    from score import audit

    counts = {"val_generation_attempts": 3, "lead_trigger_failures": 0,
              "crystal_block_failures": 0, "parse_valid_crystals": 0,
              "tail_read_attempts": 0, "tail_read_completed": 0}
    started = time.monotonic()
    with (out / "evaluation.jsonl").open("w") as log:
        for index, row in enumerate(val_rows[:3]):
            generation, _ = tasks(row)
            result = {"material_id": row["material_id"]}
            try:
                with autocast():
                    lead = model.text(generation["prefix"], tok, "hybrid", trigger=True, limit=512)
                result["lead"] = {"ok": True, "raw": lead}
            except Exception as error:
                counts["lead_trigger_failures"] += 1
                result["lead"] = failure(error)
                log.write(json.dumps(result, allow_nan=False) + "\n"); log.flush()
                continue

            try:
                with autocast():
                    crystal, tokens = model.crystal(generation["prefix"] + lead, tok, "hybrid",
                                                   torch.Generator(device="cuda").manual_seed(SEED + 2000 + index))
                result["crystal"] = {"ok": True, "raw": crystal, "tokens": tokens}
            except Exception as error:
                counts["crystal_block_failures"] += 1
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
                        result["tail_read"] = {"ok": True, "raw": model.text(
                            read_prefix, tok, "hybrid", trigger=False, limit=512)}
                    counts["tail_read_completed"] += 1
                except Exception as error:
                    result["tail_read"] = failure(error)
            log.write(json.dumps(result, allow_nan=False) + "\n"); log.flush()
    torch.cuda.synchronize()
    return counts, time.monotonic() - started


def run(data, out, updates=UPDATES):
    assert torch.cuda.is_available(), "Smoke training requires an allocated GPU"
    assert os.environ.get("SLURM_JOB_ID"), "Smoke training must run inside a Slurm allocation"
    assert updates == UPDATES, f"This smoke run is fixed at {UPDATES} updates"
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    input_paths = {"train-triples.jsonl.gz": data / "train-triples.jsonl.gz",
                   "val-triples.jsonl.gz": data / "val-triples.jsonl.gz"}
    for name, path in input_paths.items():
        assert sha256_file(path) == manifest["artifacts"][name]["sha256"], f"{name} differs from manifest"
    train_rows = read_rows(data / "train-triples.jsonl.gz", "train")
    val_rows = read_rows(data / "val-triples.jsonl.gz", "val")
    assert len(train_rows) == manifest["counts"]["train"]
    assert len(val_rows) == manifest["counts"]["val"]
    assert len(train_rows) + len(val_rows) == manifest["total"] == 8172
    tok = load_tokenizer()
    lengths = validate_lengths(train_rows + val_rows, tok)
    assert lengths["rows_checked"] == manifest["total"] == 8172

    out.mkdir(parents=True, exist_ok=False)
    model = Hybrid.pretrained(tok).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    corruption_rng = torch.Generator().manual_seed(SEED + 1000)
    shuffled = list(train_rows)
    random.Random(SEED).shuffle(shuffled)
    torch.cuda.reset_peak_memory_stats()
    training_start = time.monotonic()
    with (out / "updates.jsonl").open("w") as log:
        for step, row in enumerate(shuffled[:UPDATES], start=1):
            torch.cuda.synchronize()
            step_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            examples = tasks(row)
            assert len(examples) == MICRO_BATCH and [example["task"] for example in examples] == ["G", "Q"]
            with autocast():
                loss, stats = model.loss(examples, tok, "hybrid", corruption_rng)
            assert torch.isfinite(loss), "nonfinite joint objective"
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.grad is not None), "nonfinite gradient"
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(grad_norm), "nonfinite gradient norm"
            optimizer.step()
            torch.cuda.synchronize()
            entry = {"step": step, "loss": float(loss.detach()), "text_loss": stats["text"],
                     "crystal_loss": stats["crystal"], "grad_norm": float(grad_norm),
                     "finite_gradients": True,
                     "seconds": time.monotonic() - step_start}
            assert all(math.isfinite(entry[key]) for key in ("loss", "text_loss", "crystal_loss", "grad_norm"))
            log.write(json.dumps(entry, allow_nan=False) + "\n"); log.flush()
            print(json.dumps(entry, allow_nan=False), flush=True)

    training_seconds = time.monotonic() - training_start
    config = {
        "mode": "hybrid", "prediction_offset": PREDICTION_OFFSET,
        "model": MODEL, "revision": REVISION, "capacity": K, "steps": STEPS,
        "max_length": MAXLEN,
        "token_ids": {token: tok.convert_tokens_to_ids(token) for token in (START, END, MASK, PAD)},
        "base_config": model.base.config.to_dict(),
    }
    config_hash = json_sha256(config["base_config"])
    weights_hash = state_sha256(model.state_dict())
    checkpoint = out / "checkpoint.pt"
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "config": config, "completed_updates": updates}, temporary)
    temporary.replace(checkpoint)

    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    restored = Hybrid.pretrained(tok).cuda()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored_config = restored.restore_checkpoint(saved, expected_mode="hybrid", expected_offset=0)
    assert restored_config["model"] == MODEL and restored_config["revision"] == REVISION
    restored_config_hash = json_sha256(restored.base.config.to_dict())
    assert restored_config_hash == config_hash
    restored_weights_hash = state_sha256(restored.state_dict())
    assert restored_weights_hash == weights_hash
    del saved
    gc.collect()
    restored.eval()

    evaluation_counts, evaluation_seconds = evaluate(restored, tok, val_rows, out)
    summary = {
        "status": "smoke_complete", "seed": SEED, "updates": updates,
        "microbatch": MICRO_BATCH, "learning_rate": LEARNING_RATE,
        "model": MODEL, "revision": REVISION, "mode": "hybrid", "prediction_offset": 0,
        "data": {"manifest_sha256": sha256_file(manifest_path),
                 "train_sha256": sha256_file(data / "train-triples.jsonl.gz"),
                 "val_sha256": sha256_file(data / "val-triples.jsonl.gz"),
                 "rows": lengths},
        "checkpoint_roundtrip": {"weights_exact": True, "weights_sha256": weights_hash,
                                  "model_config_exact": True, "base_config_sha256": config_hash,
                                  "bytes": checkpoint.stat().st_size},
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "finite_gradients": True,
        "last_update": entry,
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "gpu": torch.cuda.get_device_name(),
        "val": evaluation_counts,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    args = parser.parse_args()
    run(args.data.resolve(), args.out.resolve(), args.updates)


if __name__ == "__main__":
    main()
