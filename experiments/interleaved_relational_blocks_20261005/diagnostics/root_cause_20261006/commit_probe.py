"""Compare fixed-budget B32 commit policies on the frozen seen8 checkpoint."""
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.header_probe_20261006 import probe as header_probe
from model import crystal_prefix_times, reverse_step
import prepare
import run as runner


CHECKPOINT_SHA256 = "79359e4c1936782f925b35a9fbd9b9545add49e29dff8398057aa9868c14d4e0"
BLOCK_SIZE = 32
PREFIX_TOKENS = 128
METHODS = ("legacy_random", "uniform_one", "confidence_one", "left_to_right_one")
ONE_COMMIT_METHODS = METHODS[1:]
DEFAULT_OUT = HERE / "commit_probe_001"


def top1_confidences(logits):
    """Return max-softmax probabilities, not logits, for each query position."""
    maximum = logits.max(dim=-1).values
    return (maximum - torch.logsumexp(logits, dim=-1)).exp()


@torch.inference_mode()
def sample_prefix(model, tok, prefix, method, release_rng, prefix_tokens=PREFIX_TOKENS):
    assert method in METHODS and prefix_tokens % BLOCK_SIZE == 0
    device = model.time.weight.device
    context = tok.encode(prefix, add_special_tokens=False)
    assert context[-1] == model.start_id
    assert len(context) + prefix_tokens <= prepare.MAXLEN
    banned = list(set(tok.all_special_ids) - {model.end_id, model.pad_id})
    context_ids = torch.tensor(context, dtype=torch.long, device=device)
    prefix_times = crystal_prefix_times(context_ids, model.start_id, model.end_id, model.pad_id)
    raw, blocks, release_trace = [], [], []
    calls_before = model.forward_count
    started = time.monotonic()

    for block_index, start in enumerate(range(0, prefix_tokens, BLOCK_SIZE)):
        current = torch.full((BLOCK_SIZE,), model.mask_id, dtype=torch.long, device=device)
        steps = []
        for k in range(prepare.STEPS, 0, -1):
            positions = (current == model.mask_id).nonzero().flatten()
            if not len(positions):
                break
            x = torch.tensor(context + raw, dtype=torch.long, device=device)
            span = (len(x), len(x) + BLOCK_SIZE)
            queries = [span[0] + i for i in positions.tolist()]
            times = torch.cat((prefix_times,
                               torch.zeros(len(raw), dtype=torch.long, device=device),
                               torch.full((BLOCK_SIZE,), k, dtype=torch.long, device=device)))
            logits = model.logits(torch.cat((x, current)), "hybrid", span, k, queries,
                                  times=times).float()
            if banned:
                logits[:, banned] = -torch.inf
            candidates = logits.argmax(-1)
            confidence = top1_confidences(logits)
            active_before = int(len(positions))

            if method == "legacy_random":
                predicted = current.clone()
                predicted[positions] = candidates
                was_masked = current == model.mask_id
                current = reverse_step(current, predicted, k, model.mask_id, generator=release_rng)
                committed = (was_masked & (current != model.mask_id)).nonzero().flatten()
            else:
                if method == "uniform_one":
                    row = int(torch.randperm(len(positions), device=device,
                                             generator=release_rng)[0])
                elif method == "confidence_one":
                    row = int(confidence.argmax())
                else:
                    row = 0
                committed = positions[row:row + 1]
                current[committed] = candidates[row:row + 1]

            row_by_position = {int(position): i for i, position in enumerate(positions.tolist())}
            committed_positions = [start + int(position) for position in committed.tolist()]
            committed_rows = [row_by_position[int(position)] for position in committed.tolist()]
            active_positions = [start + int(position) for position in positions.tolist()]
            step = {
                "k": k,
                "active_before": active_before,
                "active_after": int((current == model.mask_id).sum()),
                "active_positions": active_positions,
                "candidate_token_ids": [int(value) for value in candidates.tolist()],
                "prediction_confidences": [float(value) for value in confidence.tolist()],
                "committed_positions": committed_positions,
                "committed_token_ids": [int(current[position].item()) for position in committed.tolist()],
                "committed_probabilities": [float(confidence[row].item()) for row in committed_rows],
            }
            steps.append(step)
            release_trace.append({
                "block": block_index, "k": k, "active_before": active_before,
                "released_positions": [start + int(position) for position in committed.tolist()],
            })

        assert not (current == model.mask_id).any()
        raw.extend(current.tolist())
        blocks.append({"block": block_index, "start": start, "steps": steps})

    text = tok.decode(raw, skip_special_tokens=False)
    release_bytes = json.dumps(release_trace, sort_keys=True, separators=(",", ":")).encode()
    time_trace = [[step["k"] for step in block["steps"]] for block in blocks]
    time_bytes = json.dumps(time_trace, separators=(",", ":")).encode()
    return {
        "token_ids": raw,
        "text": text,
        "blocks": blocks,
        "time_trace": time_trace,
        "time_trace_sha256": header_probe.sha256(time_bytes),
        "release_trace_sha256": header_probe.sha256(release_bytes),
        "released_positions": sum(len(item["released_positions"]) for item in release_trace),
        "model_calls": model.forward_count - calls_before,
        "elapsed_seconds": time.monotonic() - started,
    }


def _token_class(token_id, model):
    return "end" if token_id == model.end_id else "pad" if token_id == model.pad_id else "body"


def _position_scores(logits, labels, queries, model):
    labels = torch.as_tensor(labels, dtype=torch.long, device=logits.device)
    probabilities = top1_confidences(logits)
    predictions = logits.argmax(-1)
    nlls = F.cross_entropy(logits, labels, reduction="none")
    return [
        {
            "position": index,
            "query_position": int(queries[index]),
            "label": int(labels[index]),
            "label_class": _token_class(int(labels[index]), model),
            "top1": int(predictions[index]),
            "top1_probability": float(probabilities[index]),
            "target_nll": float(nlls[index]),
            "correct": bool(predictions[index] == labels[index]),
        }
        for index in range(len(labels))
    ]


def _aggregate_scores(rows):
    classes = {}
    for label_class in ("body", "end", "pad"):
        selected = [row for row in rows if row["label_class"] == label_class]
        classes[label_class] = {
            "count": len(selected),
            "correct": sum(row["correct"] for row in selected),
            "nll_sum": sum(row["target_nll"] for row in selected),
        }
        classes[label_class]["accuracy"] = (
            classes[label_class]["correct"] / len(selected) if selected else None
        )
        classes[label_class]["mean_nll"] = (
            classes[label_class]["nll_sum"] / len(selected) if selected else None
        )
    return {
        "nll_count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None,
        "nll_sum": sum(row["target_nll"] for row in rows),
        "mean_nll": sum(row["target_nll"] for row in rows) / len(rows) if rows else None,
        "class_correct": classes,
    }


@torch.inference_mode()
def score_gold_block(model, context, gold_ids, noise_step, mask_seed=None):
    device = model.time.weight.device
    labels = torch.tensor(gold_ids[:BLOCK_SIZE], dtype=torch.long, device=device)
    assert len(labels) == BLOCK_SIZE and noise_step in (16, 32)
    if noise_step == BLOCK_SIZE:
        masked = torch.ones(BLOCK_SIZE, dtype=torch.bool, device=device)
    else:
        rng = torch.Generator(device=device).manual_seed(mask_seed)
        masked = torch.rand(BLOCK_SIZE, device=device, generator=rng) < noise_step / BLOCK_SIZE
    local_positions = masked.nonzero().flatten()
    queries = [len(context) + int(index) for index in local_positions.tolist()]
    corrupted = labels.clone()
    corrupted[masked] = model.mask_id
    ids = torch.cat((torch.tensor(context, dtype=torch.long, device=device), corrupted))
    prefix_times = crystal_prefix_times(torch.tensor(context, dtype=torch.long, device=device),
                                        model.start_id, model.end_id, model.pad_id)
    times = torch.cat((prefix_times, torch.full((BLOCK_SIZE,), noise_step,
                                                dtype=torch.long, device=device)))
    logits = model.logits(ids, "hybrid", (len(context), len(context) + BLOCK_SIZE),
                          noise_step, queries, times=times).float()
    rows = _position_scores(logits, labels[masked], queries, model)
    for row, local_position in zip(rows, local_positions.tolist()):
        row["position"] = int(local_position)
    return {
        "noise_step": noise_step,
        "mask_seed": mask_seed,
        "mask_seed_source": ("header_probe.case_seed:id:B32:gold_mask_k16"
                             if mask_seed is not None else None),
        "gold_target_count": BLOCK_SIZE,
        "masked_count": int(masked.sum()),
        "scored_count": len(rows),
        "scoring_scope": "masked target slots only",
        "positions": rows,
        **_aggregate_scores(rows),
    }


@torch.inference_mode()
def score_clean_ar(model, context, gold_ids):
    device = model.time.weight.device
    labels = torch.tensor(gold_ids[:BLOCK_SIZE], dtype=torch.long, device=device)
    assert len(labels) == BLOCK_SIZE
    start = len(context)
    ids = torch.cat((torch.tensor(context, dtype=torch.long, device=device), labels))
    queries = list(range(start - 1, start + BLOCK_SIZE - 1))
    logits = model.logits(ids, "ar", positions=queries).float()
    rows = _position_scores(logits, labels, queries, model)
    for index, row in enumerate(rows):
        row["position"] = index
    return {
        "target": "first 32 source crystal tokens",
        "queries": queries,
        "scoring_scope": "causal next-token teacher forcing on first 32 tokens only",
        "positions": rows,
        **_aggregate_scores(rows),
    }


def _mean_headers(rows):
    metrics = (
        "header_syntax_valid", "first_nonempty_line_terminated_by_newline",
        "complete_header_syntax_valid", "header_numbers_finite", "positive_edge_lengths",
        "angles_in_range", "cell_geometry_valid",
    )
    result = {"n": len(rows)}
    for name in metrics:
        values = [row["header"][name] for row in rows if row["header"][name] is not None]
        result[name] = {
            "count": sum(bool(value) for value in values),
            "denominator": len(values),
            "mean": sum(bool(value) for value in values) / len(values) if values else None,
        }
    return result


def _commit_times(sample):
    return {
        position: (block["block"], step["k"])
        for block in sample["blocks"] for step in block["steps"]
        for position in step["committed_positions"]
    }


def _step_release_counts(sample):
    return [len(step["committed_positions"])
            for block in sample["blocks"] for step in block["steps"]]


def _compare_one_commit_methods(samples):
    comparisons = {}
    for left, right in (("uniform_one", "confidence_one"),
                        ("uniform_one", "left_to_right_one"),
                        ("confidence_one", "left_to_right_one")):
        left_times, right_times = _commit_times(samples[left]), _commit_times(samples[right])
        left_counts, right_counts = _step_release_counts(samples[left]), _step_release_counts(samples[right])
        key = f"{left}_vs_{right}"
        comparisons[key] = {
            "exact_commit_time_matches": sum(left_times[pos] == right_times[pos] for pos in range(PREFIX_TOKENS)),
            "positions": PREFIX_TOKENS,
            "exact_release_count_step_matches": sum(a == b for a, b in zip(left_counts, right_counts)),
            "steps": min(len(left_counts), len(right_counts)),
        }
    return comparisons


def _read_legacy_reference():
    path = header_probe.DEFAULT_OUT / "prefixes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    selected = [row for row in rows if row["block_size"] == BLOCK_SIZE and row["selection"] == "greedy"]
    assert len(selected) == 8 and len({row["id"] for row in selected}) == 8
    assert sum(row["header"]["header_syntax_valid"] for row in selected) == 3
    return path, {row["id"]: row for row in selected}


def run(out=DEFAULT_OUT):
    device = header_probe.require_gpu_allocation()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite commit probe output: {out}")
    inputs = header_probe._verified_inputs()
    tok = prepare.tokenizer()
    rows = inputs["rows"]
    prefix_lengths = {
        row["id"]: len(tok.encode(prepare.tasks(row, "train", "minus", 0)[0]["prefix"]
                                   + prepare.tasks(row, "train", "minus", 0)[0]["lead"],
                                   add_special_tokens=False))
        for row in rows
    }
    assert max(prefix_lengths.values()) + PREFIX_TOKENS <= prepare.MAXLEN
    model, fingerprint_before = header_probe._load_frozen_model(tok, inputs["checkpoint"])
    legacy_path, legacy_reference = _read_legacy_reference()
    print(json.dumps({"status": "checkpoint_restored", "fingerprint_before": fingerprint_before,
                      "checkpoint_sha256": CHECKPOINT_SHA256}), flush=True)
    source_before = {
        "parent_code_hashes": inputs["parent_sources"],
        "commit_probe_py_sha256": runner.sha_file(__file__),
        "header_probe_py_sha256": runner.sha_file(header_probe.__file__),
        "codec_runtime_hashes": inputs["codec_pin"]["runtime_hashes"],
        "legacy_probe_records_sha256": runner.sha_file(legacy_path),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=False)
    generation_rows, teacher_rows = [], []
    summaries = {method: [] for method in METHODS}
    teacher_groups = {"k32_allmask": [], "k16_bernoulli": [], "clean_ar": []}

    with (out / "generation.jsonl").open("w") as log:
        for case_index, row in enumerate(rows, start=1):
            task = prepare.tasks(row, "train", "minus", 0)[0]
            conditioning_text = task["prefix"] + task["lead"]
            context_ids = tok.encode(conditioning_text, add_special_tokens=False)
            token_seed = header_probe.case_seed(row["id"], BLOCK_SIZE, "token")
            release_seed = header_probe.case_seed(row["id"], BLOCK_SIZE, "release")
            case_samples = {}
            for method in METHODS:
                release_rng = torch.Generator(device="cuda").manual_seed(release_seed)
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sample = sample_prefix(model, tok, conditioning_text, method, release_rng)
                torch.cuda.synchronize()
                header = header_probe.header_metrics(sample["text"])
                old = legacy_reference[row["id"]] if method == "legacy_random" else None
                if old is not None:
                    assert old["token_seed"] == token_seed and old["release_seed"] == release_seed
                    assert sample["token_ids"] == old["token_ids"]
                    assert sample["text"] == old["text"]
                    assert sample["time_trace_sha256"] == old["time_trace_sha256"]
                    assert sample["release_trace_sha256"] == old["release_trace_sha256"]
                    assert header == old["header"]
                record = {
                    "id": row["id"], "split": "train_seen8", "method": method,
                    "conditioning_text": conditioning_text, "conditioning_token_ids": context_ids,
                    "gold_reference_in_generation_context": False,
                    "block_size": BLOCK_SIZE, "generated_token_count": len(sample["token_ids"]),
                    "token_ids": sample["token_ids"], "text": sample["text"],
                    "blocks": sample["blocks"], "time_trace": sample["time_trace"],
                    "time_trace_sha256": sample["time_trace_sha256"],
                    "release_trace_sha256": sample["release_trace_sha256"],
                    "released_positions": sample["released_positions"],
                    "token_seed": token_seed, "token_seed_used": False,
                    "release_seed": release_seed,
                    "release_seed_used": method in ("legacy_random", "uniform_one"),
                    "seed_source": "header_probe.case_seed(case_id, 32, stream)",
                    "model_calls": sample["model_calls"], "elapsed_seconds": time.monotonic() - started,
                    "header": header, "legacy_exact_match": old is not None,
                }
                log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                case_samples[method] = sample
                summaries[method].append(record)
                generation_rows.append(record)

            gold_ids = tok.encode(row["c1"], add_special_tokens=False) + [model.end_id]
            mask_seed = header_probe.case_seed(row["id"], BLOCK_SIZE, "gold_mask_k16")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                gold_k32 = score_gold_block(model, context_ids, gold_ids, 32)
                gold_k16 = score_gold_block(model, context_ids, gold_ids, 16, mask_seed)
                clean_ar = score_clean_ar(model, context_ids, gold_ids)
            teacher_record = {
                "id": row["id"], "split": "train_seen8", "target": "G",
                "gold_target_first_32_token_ids": gold_ids[:BLOCK_SIZE],
                "scope": "first gold block token diagnostics only; no full-structure claim",
                "k32_allmask": gold_k32, "k16_bernoulli": gold_k16,
                "clean_ar": clean_ar,
            }
            teacher_rows.append(teacher_record)
            teacher_groups["k32_allmask"].extend(gold_k32["positions"])
            teacher_groups["k16_bernoulli"].extend(gold_k16["positions"])
            teacher_groups["clean_ar"].extend(clean_ar["positions"])
            if case_index % 1 == 0:
                print(json.dumps({"status": "case_complete", "case": case_index,
                                  "cases_total": len(rows), "id": row["id"]}), flush=True)

    with (out / "teacher_forced.jsonl").open("w") as log:
        for record in teacher_rows:
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    fingerprint_after = runner.fingerprint(model)
    assert fingerprint_after == fingerprint_before and model.block_size == BLOCK_SIZE
    assert model.prediction_offset == 0
    assert runner.sha_file(inputs["checkpoint"]) == CHECKPOINT_SHA256
    parent_sources_after = {name: runner.sha_file(ROOT / name) for name in inputs["status"]["code_hashes"]}
    assert parent_sources_after == inputs["parent_sources"]
    assert runner.sha_file(__file__) == source_before["commit_probe_py_sha256"]
    assert runner.sha_file(header_probe.__file__) == source_before["header_probe_py_sha256"]
    data_after = {
        "manifest_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "manifest.json"),
        "records_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "records.jsonl"),
    }
    assert data_after == inputs["data_hashes"]
    codec_pin_after = runner.verify_codec_pin(inputs["manifest"])
    assert codec_pin_after == inputs["codec_pin"]
    tokenizer_after = header_probe._tokenizer_hashes()
    assert tokenizer_after == inputs["tokenizer_hashes"]

    pairwise = {}
    for record in teacher_rows:
        by_method = {row["method"]: row for row in generation_rows if row["id"] == record["id"]}
        for key, value in _compare_one_commit_methods({
            method: {"blocks": by_method[method]["blocks"]} for method in ONE_COMMIT_METHODS
        }).items():
            current = pairwise.setdefault(key, {name: 0 for name in value})
            for name, amount in value.items():
                current[name] += amount

    legacy_rows = summaries["legacy_random"]
    exact_legacy = sum(item["legacy_exact_match"] for item in legacy_rows)
    assert exact_legacy == 8
    assert sum(item["header"]["header_syntax_valid"] for item in legacy_rows) == 3
    header_metrics = {
        method: _mean_headers(items) for method, items in summaries.items()
    }
    for rows_for_metric in teacher_groups.values():
        assert all(math.isfinite(row["top1_probability"]) and math.isfinite(row["target_nll"])
                   for row in rows_for_metric)
    runner.write_json(out / "summary.json", {
        "status": "complete", "diagnostic": "B32 commit policy root-cause probe",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fingerprint_before": fingerprint_before, "fingerprint_after": fingerprint_after,
        "model_block_size": model.block_size, "prediction_offset": model.prediction_offset,
        "methods": list(METHODS), "token_selection": "greedy argmax for every method",
        "generated_tokens_per_case": PREFIX_TOKENS, "blocks_per_case": 4,
        "split": "train_seen8_only", "cases": [row["id"] for row in rows],
        "heldout_evaluated": False, "optimizer_updates": 0,
        "legacy_reproduction": {
            "source": str(legacy_path), "cases": 8, "exact_token_sequence_matches": exact_legacy,
            "prior_header_syntax_valid": 3, "denominator": 8,
        },
        "mean_header_metrics_all_8": header_metrics,
        "mean_generation_costs_all_8": {
            method: {
                "n": len(items),
                "mean_model_calls": sum(item["model_calls"] for item in items) / len(items),
                "mean_elapsed_seconds": sum(item["elapsed_seconds"] for item in items) / len(items),
            }
            for method, items in summaries.items()
        },
        "fixed_commit_policy_pairwise_matches": pairwise,
        "teacher_forced_first_block_only": {
            "target": "G", "scope": "first 32 target tokens; no full-structure claims",
            **{name: _aggregate_scores(values) for name, values in teacher_groups.items()},
        },
        "hashes_before": source_before,
        "hashes_after": {
            "parent_code_hashes": parent_sources_after,
            "commit_probe_py_sha256": runner.sha_file(__file__),
            "header_probe_py_sha256": runner.sha_file(header_probe.__file__),
            "codec_runtime_hashes": codec_pin_after["runtime_hashes"],
            "data": data_after, "tokenizer": tokenizer_after,
            "checkpoint_sha256": runner.sha_file(inputs["checkpoint"]),
        },
        "codec_pin": inputs["codec_pin"], "device": device,
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
