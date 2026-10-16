"""Frozen-checkpoint prefix diagnostics for relational-CNL header generation."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import Hybrid, crystal_prefix_times, reverse_step
import prepare
import run as runner


CHECKPOINT_SHA256 = "79359e4c1936782f925b35a9fbd9b9545add49e29dff8398057aa9868c14d4e0"
FINGERPRINT = "474810720624206b4623a586f7ffc9d099ae84ff0faabf579b259d7d09249350"
RECORDS_SHA256 = "ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f"
BLOCK_SIZES = (32, 16, 8)
SELECTIONS = ("multinomial", "greedy")
PREFIX_TOKENS = 128
DEFAULT_OUT = HERE / "probe_001"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


@torch.inference_mode()
def sample_prefix(model, tok, prefix, block_size, selection, token_rng, release_rng,
                  prefix_tokens=PREFIX_TOKENS):
    """Generate a fixed token prefix with independent token and release RNGs."""
    assert type(block_size) is int and block_size > 0 and prefix_tokens % block_size == 0
    assert selection in SELECTIONS
    device = model.time.weight.device
    context = tok.encode(prefix, add_special_tokens=False)
    assert context[-1] == model.start_id
    assert len(context) + prefix_tokens <= prepare.MAXLEN
    banned = list(set(tok.all_special_ids) - {model.end_id, model.pad_id})
    context_ids = torch.tensor(context, dtype=torch.long, device=device)
    prefix_times = crystal_prefix_times(context_ids, model.start_id, model.end_id, model.pad_id)
    raw, time_trace, release_trace = [], [], []
    calls_before = model.forward_count
    started = time.monotonic()

    for block_index, start in enumerate(range(0, prefix_tokens, block_size)):
        current = torch.full((block_size,), model.mask_id, dtype=torch.long, device=device)
        block_times = []
        for k in range(prepare.STEPS, 0, -1):
            positions = (current == model.mask_id).nonzero().flatten()
            if not len(positions):
                break
            x = torch.tensor(context + raw, dtype=torch.long, device=device)
            span = (len(x), len(x) + block_size)
            queries = [span[0] + i for i in positions.tolist()]
            times = torch.cat((prefix_times,
                               torch.zeros(len(raw), dtype=torch.long, device=device),
                               torch.full((block_size,), k, dtype=torch.long, device=device)))
            logits = model.logits(torch.cat((x, current)), "hybrid", span, k, queries,
                                  times=times).float()
            if banned:
                logits[:, banned] = -torch.inf
            if selection == "multinomial":
                values = torch.multinomial((logits / 0.8).softmax(-1), 1,
                                           generator=token_rng).flatten()
            else:
                values = logits.argmax(-1)
            predicted = current.clone()
            predicted[positions] = values
            was_masked = current == model.mask_id
            current = reverse_step(current, predicted, k, model.mask_id, generator=release_rng)
            released = (was_masked & (current != model.mask_id)).nonzero().flatten().tolist()
            block_times.append(k)
            release_trace.append({
                "block": block_index,
                "k": k,
                "active_before": int(was_masked.sum()),
                "released_positions": [start + i for i in released],
            })
        raw.extend(current.tolist())
        time_trace.append(block_times)

    text = tok.decode(raw, skip_special_tokens=False)
    release_bytes = json.dumps(release_trace, sort_keys=True, separators=(",", ":")).encode()
    time_bytes = json.dumps(time_trace, separators=(",", ":")).encode()
    return {
        "token_ids": raw,
        "text": text,
        "time_trace": time_trace,
        "time_trace_sha256": sha256(time_bytes),
        "release_trace_sha256": sha256(release_bytes),
        "released_positions": sum(len(item["released_positions"]) for item in release_trace),
        "model_calls": model.forward_count - calls_before,
        "elapsed_seconds": time.monotonic() - started,
    }


def header_metrics(text):
    """Check only the first relational-CNL cell header, without decoding a document."""
    first_line = None
    newline_present = False
    for raw_line in text.splitlines(keepends=True):
        content = raw_line.rstrip("\r\n")
        if content.strip():
            first_line = content.strip()
            newline_present = raw_line.endswith(("\n", "\r"))
            break
    match = prepare.relational_codec._CELL.fullmatch(first_line) if first_line is not None else None
    value_strings = list(match.groups()) if match else None
    values = [float(value) for value in value_strings] if value_strings else None
    finite = bool(values is not None and all(math.isfinite(value) for value in values))
    positive_edges = bool(finite and all(value > 0 for value in values[:3]))
    valid_angles = bool(finite and all(0 < value < 180 for value in values[3:]))
    cell_geometry_valid = None
    if positive_edges and valid_angles:
        alpha, beta, gamma = map(math.radians, values[3:])
        gram = (1 + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
                - math.cos(alpha) ** 2 - math.cos(beta) ** 2 - math.cos(gamma) ** 2)
        cell_geometry_valid = gram > 0
    return {
        "first_nonempty_line_terminated_by_newline": newline_present,
        "header_syntax_valid": match is not None,
        "complete_header_syntax_valid": match is not None and newline_present,
        "header_value_strings": value_strings,
        "header_values": ([value if math.isfinite(value) else None for value in values]
                           if values is not None else None),
        "header_numbers_finite": finite,
        "positive_edge_lengths": positive_edges,
        "angles_in_range": valid_angles,
        "cell_geometry_valid": cell_geometry_valid,
    }


def require_gpu_allocation():
    assert torch.cuda.is_available(), "Header probe requires a Slurm CUDA allocation"
    assert os.environ.get("SLURM_JOB_ID"), "Refusing inference outside Slurm"
    assert "hpclogin" not in socket.gethostname().lower(), "Refusing inference on the login node"
    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    assert "A100" in name and memory_gib >= 79, (
        f"Requires one A100 80GB allocation; found {name} ({memory_gib:.1f} GiB)"
    )
    return {"name": name, "total_memory_GiB": memory_gib}


def case_seed(case_id, block_size, stream):
    digest = hashlib.sha256(f"header_probe_20261006:{case_id}:B{block_size}:{stream}".encode()).hexdigest()
    return int(digest[:8], 16)


def _tokenizer_hashes():
    return {
        path.name: runner.sha_file(path)
        for path in sorted(Path(prepare.TOKENIZER_PATH).iterdir()) if path.is_file()
    }


def _verified_inputs():
    data = Path(prepare.DATA_DIR)
    manifest = prepare.load_manifest()
    status = json.loads((ROOT / "pilot_001/diffusion_b32/status.json").read_text())
    receipt = json.loads((ROOT / "pilot_001/diffusion_b32/checkpoint_receipt.json").read_text())
    checkpoint = ROOT / "pilot_001/diffusion_b32/checkpoint.pt"
    records_hash = runner.sha_file(data / "records.jsonl")
    manifest_hash = runner.sha_file(data / "manifest.json")
    assert records_hash == RECORDS_SHA256 == status["dataset_records_sha256"]
    assert manifest_hash == status["dataset_manifest_sha256"]
    assert runner.sha_file(checkpoint) == CHECKPOINT_SHA256 == receipt["checkpoint_sha256"]
    assert receipt["final_fingerprint"] == FINGERPRINT
    assert receipt["reload_fingerprint"] == FINGERPRINT and receipt["strict_state_load"] is True
    assert status["final_fingerprint"] == FINGERPRINT
    parent_sources = {name: runner.sha_file(ROOT / name) for name in status["code_hashes"]}
    assert parent_sources == status["code_hashes"], "pilot runtime code differs from its frozen receipt"
    codec_pin = runner.verify_codec_pin(manifest)
    tokenizer_hashes = _tokenizer_hashes()
    rows = prepare.load_records(splits="train", path=data / "records.jsonl")
    assert len(rows) == 8 and all(row["split"] == "train" for row in rows)
    assert [row["id"] for row in rows] == manifest["cohort"]["train_ids"]
    return {
        "manifest": manifest,
        "status": status,
        "receipt": receipt,
        "checkpoint": checkpoint,
        "rows": rows,
        "codec_pin": codec_pin,
        "parent_sources": parent_sources,
        "data_hashes": {"manifest_sha256": manifest_hash, "records_sha256": records_hash},
        "tokenizer_hashes": tokenizer_hashes,
    }


def _load_frozen_model(tok, checkpoint):
    model = Hybrid.pretrained(tok, prediction_offset=0, block_size=32).cuda().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = model.restore_checkpoint(saved, expected_mode="hybrid", expected_offset=0,
                                      expected_block_size=32)
    del saved
    gc.collect()
    before = runner.fingerprint(model)
    assert config["final_fingerprint"] == FINGERPRINT and before == FINGERPRINT
    assert model.prediction_offset == 0 and model.block_size == 32
    return model, before


def run(out=DEFAULT_OUT):
    device = require_gpu_allocation()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite probe output: {out}")
    inputs = _verified_inputs()
    tok = prepare.tokenizer()
    header_reference_lengths = {
        row["id"]: len(tok.encode(row["c1"].split("\n", 1)[0] + "\n", add_special_tokens=False))
        for row in inputs["rows"]
    }
    assert max(header_reference_lengths.values()) <= PREFIX_TOKENS, (
        f"Reference header exceeds {PREFIX_TOKENS} tokens: {header_reference_lengths}"
    )
    model, fingerprint_before = _load_frozen_model(tok, inputs["checkpoint"])
    print(json.dumps({"status": "checkpoint_restored", "fingerprint_before": fingerprint_before,
                      "checkpoint_sha256": CHECKPOINT_SHA256}), flush=True)
    source_before = {
        "parent_code_hashes": inputs["parent_sources"],
        "probe_py_sha256": runner.sha_file(__file__),
        "codec_runtime_hashes": inputs["codec_pin"]["runtime_hashes"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)

    summaries = {
        str(block_size): {
            selection: {"n": 0, "header_syntax_valid": 0, "header_numbers_finite": 0,
                        "first_nonempty_line_terminated_by_newline": 0,
                        "complete_header_syntax_valid": 0,
                        "positive_edge_lengths": 0, "angles_in_range": 0,
                        "cell_geometry_valid": 0, "cell_geometry_checked": 0,
                        "failures": 0, "model_calls": 0, "elapsed_seconds": 0.0}
            for selection in SELECTIONS
        }
        for block_size in BLOCK_SIZES
    }
    paired_release_hashes = {}
    with (out / "prefixes.jsonl").open("w") as log:
        for case_index, row in enumerate(inputs["rows"], start=1):
            generation = prepare.tasks(row, "train", "minus", 0)[0]
            prefix = generation["prefix"] + generation["lead"]
            for block_size in BLOCK_SIZES:
                token_seed = case_seed(row["id"], block_size, "token")
                release_seed = case_seed(row["id"], block_size, "release")
                pair_key = (row["id"], block_size)
                paired_release_hashes[pair_key] = []
                for selection in SELECTIONS:
                    token_rng = torch.Generator(device="cuda").manual_seed(token_seed)
                    release_rng = torch.Generator(device="cuda").manual_seed(release_seed)
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    try:
                        calls_before = model.forward_count
                        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                            sample = sample_prefix(model, tok, prefix, block_size, selection,
                                                   token_rng, release_rng, PREFIX_TOKENS)
                        torch.cuda.synchronize()
                        metrics = header_metrics(sample["text"])
                        error = None
                    except Exception as exception:
                        torch.cuda.synchronize()
                        sample = {"token_ids": [], "text": "", "time_trace": [],
                                  "time_trace_sha256": None, "release_trace_sha256": None,
                                  "released_positions": 0,
                                  "model_calls": model.forward_count - calls_before}
                        metrics = header_metrics("")
                        error = f"{type(exception).__name__}: {exception}"
                    elapsed = time.monotonic() - started
                    sample_row = {
                        "id": row["id"], "split": "train", "block_size": block_size,
                        "selection": selection, "attempt": 1, "sampling_retries": 0,
                        "token_seed": token_seed, "release_seed": release_seed,
                        "token_ids": sample["token_ids"], "generated_token_count": len(sample["token_ids"]),
                        "text": sample["text"], "time_trace": sample["time_trace"],
                        "time_trace_sha256": sample["time_trace_sha256"],
                        "release_trace_sha256": sample["release_trace_sha256"],
                        "released_positions": sample["released_positions"],
                        "model_calls": sample["model_calls"], "elapsed_seconds": elapsed,
                        "header": metrics, "error": error,
                    }
                    log.write(json.dumps(sample_row, ensure_ascii=False, allow_nan=False) + "\n")
                    log.flush()
                    paired_release_hashes[pair_key].append(sample_row["release_trace_sha256"])
                    summary = summaries[str(block_size)][selection]
                    summary["n"] += 1
                    summary["failures"] += error is not None
                    for key in ("header_syntax_valid", "first_nonempty_line_terminated_by_newline",
                                "complete_header_syntax_valid", "header_numbers_finite",
                                "positive_edge_lengths", "angles_in_range"):
                        summary[key] += bool(metrics[key])
                    if metrics["cell_geometry_valid"] is not None:
                        summary["cell_geometry_checked"] += 1
                        summary["cell_geometry_valid"] += metrics["cell_geometry_valid"]
                    summary["model_calls"] += sample["model_calls"]
                    summary["elapsed_seconds"] += elapsed
                release_hashes = paired_release_hashes[pair_key]
                assert release_hashes[0] is not None and release_hashes[0] == release_hashes[1], (
                    f"Release schedule differs across token policies for {pair_key}"
                )
            print(json.dumps({"status": "case_complete", "case": case_index,
                              "cases_total": len(inputs["rows"]), "id": row["id"],
                              "attempts_complete": case_index * len(BLOCK_SIZES) * len(SELECTIONS)}),
                  flush=True)

    for by_selection in summaries.values():
        for summary in by_selection.values():
            assert summary["n"] == 8
            summary["denominator"] = 8
            summary["sampling_retries"] = 0
            summary["header_syntax_rate"] = summary["header_syntax_valid"] / 8
            summary["header_line_termination_rate"] = (
                summary["first_nonempty_line_terminated_by_newline"] / 8
            )
            summary["complete_header_syntax_rate"] = summary["complete_header_syntax_valid"] / 8

    fingerprint_after = runner.fingerprint(model)
    assert fingerprint_after == fingerprint_before == FINGERPRINT
    assert model.block_size == 32 and model.prediction_offset == 0
    checkpoint_after = runner.sha_file(inputs["checkpoint"])
    parent_sources_after = {
        name: runner.sha_file(ROOT / name) for name in inputs["status"]["code_hashes"]
    }
    assert parent_sources_after == inputs["parent_sources"]
    assert runner.sha_file(__file__) == source_before["probe_py_sha256"]
    assert checkpoint_after == CHECKPOINT_SHA256
    data_after = {
        "manifest_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "manifest.json"),
        "records_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "records.jsonl"),
    }
    assert data_after == inputs["data_hashes"]
    codec_pin_after = runner.verify_codec_pin(inputs["manifest"])
    assert codec_pin_after == inputs["codec_pin"]
    tokenizer_after = _tokenizer_hashes()
    assert tokenizer_after == inputs["tokenizer_hashes"]
    print(json.dumps({"status": "inference_complete", "fingerprint_before": fingerprint_before,
                      "fingerprint_after": fingerprint_after}), flush=True)
    write_json(out / "summary.json", {
        "status": "complete", "diagnostic": "frozen_checkpoint_prefix_only",
        "device": device, "checkpoint_sha256": CHECKPOINT_SHA256,
        "fingerprint_before": fingerprint_before, "fingerprint_after": fingerprint_after,
        "model_block_size": model.block_size, "prediction_offset": model.prediction_offset,
        "inference_block_sizes": list(BLOCK_SIZES), "selection_policies": list(SELECTIONS),
        "generated_tokens_per_attempt": PREFIX_TOKENS,
        "cases": [row["id"] for row in inputs["rows"]], "split": "train_seen8_only",
        "heldout_evaluated": False, "sampling_retries": 0,
        "header_reference_lengths": header_reference_lengths,
        "paired_release_hashes_match": True,
        "groups": summaries,
        "hashes_before": {
            **source_before, "data": inputs["data_hashes"],
            "tokenizer": inputs["tokenizer_hashes"],
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "hashes_after": {
            "parent_code_hashes": parent_sources_after,
            "probe_py_sha256": runner.sha_file(__file__),
            "codec_runtime_hashes": codec_pin_after["runtime_hashes"],
            "data": data_after,
            "tokenizer": tokenizer_after,
            "checkpoint_sha256": checkpoint_after,
        },
        "codec_pin": inputs["codec_pin"],
    })
    del model
    torch.cuda.empty_cache()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "complete", "output": str(run(args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
