#!/usr/bin/env python3
"""Read-only denoising probe for pilot_002's shifted-hybrid checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "diagnostic001"
CHECKPOINT = ROOT / "pilot_002" / "hybrid_shift" / "checkpoint.pt"
STATUS = ROOT / "pilot_002" / "hybrid_shift" / "status.json"
CHECKPOINT_RECEIPT = ROOT / "pilot_002" / "hybrid_shift" / "checkpoint_receipt.json"
RUN_SOURCE = ROOT / "q2_coverage_preparation" / "source_snapshot" / "run.py"
EXPECTED_CHECKPOINT_SHA256 = "3e1b0e30b5d7edc147e11b1b975959835cad60587f1aa5aafc8ca54b0cdf25a2"
EXPECTED_FINAL_FINGERPRINT = "01f8ce739f46b7d4ebfa10810940f891f1fea0da37f016acd5788a4b161aabd8"
EXPECTED_RUN_SHA256 = "9f5881028177c3b527ea064198b91ae73600894801960113aec2e8ffcea8ab50"
EXPECTED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def install_unpack_observer(core, events, case_id):
    """Record raw IDs while preserving the original unpack call and outcome."""
    original = core.unpack_block

    def observe(ids, end_id, pad_id):
        raw_ids = list(ids)
        event = {
            "case_id": case_id,
            "raw_token_ids": raw_ids,
            "end_positions": [i for i, token in enumerate(raw_ids) if token == end_id],
            "pad_positions": [i for i, token in enumerate(raw_ids) if token == pad_id],
        }
        try:
            result = original(ids, end_id, pad_id)
        except Exception as error:
            event["unpack_error"] = f"{type(error).__name__}: {error}"
            events.append(event)
            raise
        event["unpack_error"] = None
        events.append(event)
        return result

    core.unpack_block = observe
    return original


def self_check_observer():
    """Exercise the observation hook with a toy unpacker; imports no ML stack."""
    class ToyCore:
        pass

    core = ToyCore()
    success_ids = [4, 7, 8, 8]
    failed_ids = [5, 7, 8]
    marker = object()
    expected_error = ValueError("sentinel unpack failure")
    calls = []

    def unpack(ids, end_id, pad_id):
        calls.append(ids)
        if ids is success_ids:
            return marker
        if ids is failed_ids:
            raise expected_error
        raise AssertionError("observer changed the input object")

    core.unpack_block = unpack
    events = []
    original = install_unpack_observer(core, events, "toy")
    try:
        assert core.unpack_block(success_ids, 7, 8) is marker
        try:
            core.unpack_block(failed_ids, 7, 8)
        except ValueError as error:
            assert error is expected_error
        else:
            raise AssertionError("the original exception was swallowed")
    finally:
        core.unpack_block = original

    assert calls == [success_ids, failed_ids]
    assert events[0]["raw_token_ids"] == success_ids
    assert events[0]["end_positions"] == [1] and events[0]["pad_positions"] == [2, 3]
    assert events[0]["unpack_error"] is None
    assert events[1]["raw_token_ids"] == failed_ids
    assert events[1]["unpack_error"] == "ValueError: sentinel unpack failure"


def _seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_seen8(records_path):
    rows = []
    with Path(records_path).open() as stream:
        for _ in range(8):
            line = stream.readline()
            if not line:
                raise ValueError("frozen records ended before eight train rows")
            rows.append(json.loads(line))
    if len(rows) != 8 or any(row["split"] != "train" for row in rows):
        raise ValueError("expected exactly the first eight frozen train rows")
    return rows


def _class_metrics(mask, labels, predictions, target_nll, target_probability, top1_probability):
    count = int(mask.sum().item())
    if count == 0:
        return {
            "count": 0,
            "top1_accuracy": None,
            "mean_target_nll": None,
            "mean_target_probability": None,
            "mean_top1_probability": None,
        }
    return {
        "count": count,
        "top1_accuracy": float((predictions[mask] == labels[mask]).float().mean().item()),
        "mean_target_nll": float(target_nll[mask].mean().item()),
        "mean_target_probability": float(target_probability[mask].mean().item()),
        "mean_top1_probability": float(top1_probability[mask].mean().item()),
    }


def _teacher_forced(torch, autocast, model, model_core, tok, task, k, mask_seed, device, strict_unpack, steps, capacity):
    example = dict(task, noise_step=k)
    mask_rng = torch.Generator().manual_seed(mask_seed)
    packed = model.pack(example, tok, "hybrid", mask_rng)
    group = next(group for group in packed["groups"] if group["kind"] == "crystal")
    prefix_length = len(tok.encode(task["prefix"], add_special_tokens=False))
    lead_length = len(tok.encode(task["lead"], add_special_tokens=False))
    start = prefix_length + lead_length
    query_to_block = {
        query: index
        for index, query in enumerate(
            model_core.prediction_positions(start, range(capacity), "hybrid", model.prediction_offset)
        )
    }
    block_indices = [query_to_block[int(position)] for position in group["positions"]]
    if k == steps and len(block_indices) != capacity:
        raise AssertionError("k=32 must mask all K positions, including END and PAD")

    positions = torch.as_tensor(group["positions"], dtype=torch.long, device=device)
    labels = torch.as_tensor(group["labels"], dtype=torch.long, device=device)
    if len(group["positions"]):
        ids = packed["ids"].to(device)
        allowed = packed["allowed"].to(device)
        times = packed["times"].to(device)
        with torch.inference_mode(), autocast():
            hidden = model.hidden(ids[None], allowed, times[None])[0]
            logits = model.base.lm_head(hidden[positions]).float()
            top1_ids = logits.argmax(dim=-1)
            log_partition = torch.logsumexp(logits, dim=-1)
            target_nll = log_partition - logits.gather(1, labels[:, None]).squeeze(1)
            top1_probability = (logits.max(dim=-1).values - log_partition).exp()
            target_probability = (-target_nll).exp()
    else:
        top1_ids = torch.empty(0, dtype=torch.long, device=device)
        target_nll = top1_probability = target_probability = torch.empty(0, device=device)

    oracle_body_length = len(tok.encode(task["crystal"], add_special_tokens=False))
    block_indices_tensor = torch.as_tensor(block_indices, dtype=torch.long, device=device)
    is_body = block_indices_tensor < oracle_body_length
    is_end = block_indices_tensor == oracle_body_length
    is_pad = block_indices_tensor > oracle_body_length
    for index, label in zip(block_indices, group["labels"]):
        expected = model.end_id if index == oracle_body_length else model.pad_id if index > oracle_body_length else None
        if expected is not None and label != expected:
            raise AssertionError("teacher-forced target END/PAD grouping disagrees with pack()")
        if expected is None and label in (model.end_id, model.pad_id):
            raise AssertionError("teacher-forced body grouping contains END/PAD")
    metrics = {
        "body": _class_metrics(is_body, labels, top1_ids, target_nll, target_probability, top1_probability),
        "END": _class_metrics(is_end, labels, top1_ids, target_nll, target_probability, top1_probability),
        "PAD": _class_metrics(is_pad, labels, top1_ids, target_nll, target_probability, top1_probability),
    }

    predictions = top1_ids.tolist()
    masked_block = packed["ids"][start:start + capacity].tolist()
    for index, prediction in zip(block_indices, predictions):
        masked_block[index] = prediction
    end_top1_positions = [index for index, prediction in zip(block_indices, predictions) if prediction == model.end_id]
    try:
        strict_unpack(masked_block, model.end_id, model.pad_id)
        boundary_error = None
    except Exception as error:
        boundary_error = f"{type(error).__name__}: {error}"

    return {
        "k": k,
        "mask_seed": mask_seed,
        "masked_count": len(block_indices),
        "oracle_body_tokens_for_grouping_only": oracle_body_length,
        "prefix_tokens": prefix_length,
        "lead_tokens": lead_length,
        "class_metrics": metrics,
        "masked_block_positions": block_indices,
        "masked_target_token_ids": group["labels"],
        "masked_top1_token_ids": predictions,
        "masked_argmax_end_positions": end_top1_positions,
        "masked_argmax_end_count": len(end_top1_positions),
        "teacher_forced_top1_boundary_valid": boundary_error is None,
        "teacher_forced_top1_boundary_error": boundary_error,
    }


def run_probe():
    import torch

    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("run the probe inside a Slurm CUDA allocation")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("the Slurm environment must use the pinned offline HF cache")
    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    device_properties = torch.cuda.get_device_properties(device_index)
    if "A100" not in device_name or device_properties.total_memory < 79 * 2**30:
        raise RuntimeError(f"expected an 80-GiB A100, found {device_name}")
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite the unique output directory: {OUT}")

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import model as model_core
    import prepare

    spec = importlib.util.spec_from_file_location("pilot002_frozen_run", RUN_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen runner helper: {RUN_SOURCE}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    status = json.loads(STATUS.read_text())
    receipt = json.loads(CHECKPOINT_RECEIPT.read_text())
    if status["status"] != "evaluated" or status["mode"] != "hybrid" or status["prediction_offset"] != 1:
        raise ValueError("hybrid status does not match the requested checkpoint")
    if status["model_revision"] != EXPECTED_REVISION or prepare.REVISION != EXPECTED_REVISION:
        raise ValueError("pretrained model revision changed")
    if prepare.K != 2560 or prepare.STEPS != 32:
        raise ValueError("K or diffusion step count changed")

    checkpoint_sha256 = runner.sha_file(CHECKPOINT)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256 or checkpoint_sha256 != receipt["checkpoint_sha256"]:
        raise ValueError("hybrid checkpoint SHA does not match the verified receipt")
    if status["final_fingerprint"] != EXPECTED_FINAL_FINGERPRINT:
        raise ValueError("hybrid status fingerprint does not match the requested checkpoint")
    if (
        status["checkpoint_reload_mode"] != "hybrid"
        or status["checkpoint_reload_offset"] != 1
        or status["checkpoint_reload_fingerprint"] != EXPECTED_FINAL_FINGERPRINT
    ):
        raise ValueError("hybrid status reload receipt does not match mode, offset and fingerprint")
    if not status["checkpoint_reload_verified"] or not receipt["reload_verified"] or not receipt["strict_state_load"]:
        raise ValueError("hybrid checkpoint receipt does not confirm a strict reload")
    if (
        receipt["mode"] != "hybrid"
        or receipt["prediction_offset"] != 1
        or receipt["final_fingerprint"] != EXPECTED_FINAL_FINGERPRINT
        or receipt["reload_fingerprint"] != EXPECTED_FINAL_FINGERPRINT
    ):
        raise ValueError("hybrid checkpoint receipt does not match mode, offset and fingerprint")
    if runner.sha_file(RUN_SOURCE) != EXPECTED_RUN_SHA256 or status["code_hashes"]["run.py"] != EXPECTED_RUN_SHA256:
        raise ValueError("frozen runner source differs from the checkpoint receipt")

    core_hashes = {
        name: runner.sha_file(ROOT / name)
        for name in ("model.py", "prepare.py", "score.py")
    }
    if any(core_hashes[name] != status["code_hashes"][name] for name in core_hashes):
        raise ValueError("current model/prepare/score core differs from the checkpoint receipt")

    data_hashes = runner.data_source_hashes(prepare.DATA_DIR)
    if data_hashes["manifest_sha256"] != status["dataset_manifest_sha256"]:
        raise ValueError("frozen dataset manifest SHA differs from the checkpoint receipt")
    if data_hashes["records_sha256"] != status["dataset_records_sha256"]:
        raise ValueError("frozen dataset records SHA differs from the checkpoint receipt")
    rows = _read_seen8(prepare.DATA_DIR / "records.jsonl")
    expected_train_ids = set(status["actual_exposure_counts"]["per_material"])
    if {row["id"] for row in rows} != expected_train_ids:
        raise ValueError("the first eight records are not the checkpoint's seen8 cohort")

    tok = prepare.TOK
    model = model_core.Hybrid.pretrained(tok, prediction_offset=1).cuda().eval()
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    loaded_config = model.restore_checkpoint(saved, expected_mode="hybrid", expected_offset=1)
    del saved
    loaded_fingerprint = runner.fingerprint(model)
    if loaded_config["mode"] != "hybrid" or loaded_config["prediction_offset"] != 1:
        raise ValueError("strict checkpoint config is not hybrid offset 1")
    if loaded_config["final_fingerprint"] != EXPECTED_FINAL_FINGERPRINT or loaded_fingerprint != EXPECTED_FINAL_FINGERPRINT:
        raise ValueError("strictly reloaded checkpoint fingerprint mismatch")

    results = []
    mode = "hybrid"
    for row in rows:
        direction = "minus" if _seed(row["id"]) % 2 == 0 else "plus"
        task = prepare.tasks(row, "train", direction, 0)[0]
        sampling_seed = _seed(row["id"] + "20261004")
        rng = torch.Generator(device="cuda").manual_seed(sampling_seed)
        events = []
        original_unpack = install_unpack_observer(model_core, events, row["id"])
        try:
            generated = runner.attempt(lambda: runner.generate(model, tok, mode, task["prefix"], rng))
        finally:
            model_core.unpack_block = original_unpack
        if len(events) > 1 or (generated["ok"] and len(events) != 1):
            raise AssertionError(f"expected at most one original unpack call for {row['id']}")
        observation = events[0] if events else None
        if generated["ok"] and generated["value"]["tokens"] != observation["raw_token_ids"]:
            raise AssertionError(f"observer changed the sampled token sequence for {row['id']}")
        results.append({
            "id": row["id"],
            "split": "train",
            "direction": direction,
            "sampling_seed": sampling_seed,
            "mode": mode,
            "temperature": 0.8,
            "steps": prepare.STEPS,
            "free_prefix_tokens": len(tok.encode(task["prefix"], add_special_tokens=False)),
            "free_generated_lead_tokens": len(tok.encode(generated["value"]["lead"], add_special_tokens=False)) if generated["ok"] else None,
            "free_raw_block_tokens": len(observation["raw_token_ids"]) if observation is not None else None,
            "free_generation_ok": generated["ok"],
            "free_generation_error": generated.get("error"),
            "lead_sha256": hashlib.sha256(generated["value"]["lead"].encode()).hexdigest() if generated["ok"] else None,
            "crystal_text_sha256": hashlib.sha256(generated["value"]["crystal"].encode()).hexdigest() if generated["ok"] else None,
            "raw_unpacked_block_observation": observation,
        })

    strict_unpack = model_core.unpack_block
    mask_seeds = {}
    for result, row in zip(results, rows):
        direction = result["direction"]
        task = prepare.tasks(row, "train", direction, 0)[0]
        metrics = []
        for k in (32, 16, 1):
            mask_seed = _seed(f"{row['id']}20261004:teacher-mask:k{k}")
            mask_seeds[f"{row['id']}:k{k}"] = mask_seed
            metrics.append(_teacher_forced(
                torch, runner.autocast, model, model_core, tok, task, k, mask_seed, torch.device("cuda"),
                strict_unpack, prepare.STEPS, prepare.K,
            ))
        result["teacher_forced"] = metrics

    final_fingerprint = runner.fingerprint(model)
    if final_fingerprint != loaded_fingerprint:
        raise AssertionError("probe changed model weights")

    OUT.mkdir(parents=False, exist_ok=False)
    with (OUT / "results.jsonl").open("x") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    runner.write_json(OUT / "summary.json", {
        "status": "complete",
        "job_id": os.environ["SLURM_JOB_ID"],
        "device": device_name,
        "device_total_GiB": device_properties.total_memory / 2**30,
        "model": status["model"],
        "revision": status["model_revision"],
        "mode": "hybrid",
        "prediction_offset": 1,
        "temperature": 0.8,
        "steps": prepare.STEPS,
        "random_release": "unchanged Hybrid.crystal/reverse_step path",
        "samples_per_id": 1,
        "train_rows_loaded": 8,
        "heldout_rows_decoded_or_evaluated": 0,
        "optimizer_steps": 0,
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "sha256": checkpoint_sha256,
            "final_fingerprint": loaded_fingerprint,
            "strict_reload": True,
        },
        "data_sha256": data_hashes,
        "seen8_ids": [row["id"] for row in rows],
        "core_sha256": core_hashes,
        "frozen_runner_sha256": runner.sha_file(RUN_SOURCE),
        "current_run_py_sha256_not_used_for_sampling": runner.sha_file(ROOT / "run.py"),
        "diagnostic_code_sha256": runner.sha_file(Path(__file__)),
        "mask_seeds": mask_seeds,
        "oracle_target_length_used_for_free_sampling": False,
        "metric_semantics": {
            "teacher_forced_prefix": "gold prefix and lead from the fixed seen8 G training example; labels come from its gold CNL block",
            "teacher_forced_logits": "raw full-vocabulary logits under the existing BF16 autocast context, without free-sampler special-token filtering",
            "free_sampling": "existing Hybrid sampler and its original special-token filtering/temperature/reverse-release behavior",
            "comparison": "teacher-forced top1 and NLL are conditional denoising metrics and are not directly comparable to free-generation accuracy",
        },
        "raw_capture": "observer records all token IDs and END/PAD positions immediately before calling the original unpack_block; it returns/raises unchanged",
        "weight_fingerprint_after": final_fingerprint,
    })
    print(json.dumps({"status": "complete", "output": str(OUT), "rows": len(results)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="exercise the observational hook without importing/loading the ML stack")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check_observer()
        print("observer self-check passed")
        return 0
    run_probe()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
