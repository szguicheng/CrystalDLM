"""Sample the fixed 700-request benchmark with the trained AR/diffusion model."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
AR_EVALUATION_PATH = ROOT / "experiments/structure_conditioned_ar8b_20261017/evaluate.py"
REFERENCE_RUNNER_PATH = ROOT / "experiments/structure_request_eval_20261017/run.py"
REFERENCE_SNAPSHOT = ROOT / "experiments/structure_conditioned_ar8b_20261017/reference_source_snapshots/structure_request_eval_run.py"
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["data"]["directory"]
MODEL = ROOT / "models/Qwen3-8B-b968826d9c46"
TRAIN_RUN = EXPERIMENT / "run_001"
DEFAULT_CHECKPOINT = TRAIN_RUN / "checkpoint.pt"
EHULL_BOUND = CONFIG["evaluation"]["ehull_upper_bound_eV_per_atom"]
DIFFUSION_STEPS = CONFIG["diffusion"]["steps"]
CRYSTAL_CAPACITY = CONFIG["diffusion"]["crystal_capacity"]
PREDICTION_OFFSET = CONFIG["diffusion"]["prediction_offset"]
TRAINING_PROMPT_SCOPE = (
    "G tasks request the actual space-group number and an ehull upper bound; "
    "Q targets retain the frozen factual reading targets. Text is causal AR and "
    "the crystal block is trained in hybrid mode with bidirectional absorbing-MASK diffusion.")
EVALUATION_SCOPE = (
    "fixed matched space-group and ehull-upper-bound requests; 700 hybrid samples; "
    "strict structure audit followed by separate GRACE scoring")
_TRAINING = None


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AR_EVALUATION = _load_module("structure_conditioned_ar8b_evaluation_for_hybrid", AR_EVALUATION_PATH)
RUNNER = AR_EVALUATION.RUNNER
RUNNER.EXPERIMENT = EXPERIMENT.resolve()
RUNNER.SHARED_RUN = TRAIN_RUN.resolve()
RUNNER.DEFAULT_DATA = DATA.resolve()
RUNNER.DEFAULT_MODEL = MODEL.resolve()
RUNNER.TRAINING_PROMPT_SCOPE = TRAINING_PROMPT_SCOPE
RUNNER.EVALUATION_SCOPE = EVALUATION_SCOPE
REQUESTS = RUNNER.REQUESTS


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_sources():
    return {
        "evaluate.py": _sha256_file(EXPERIMENT / "evaluate.py"),
        "matched_ar_evaluate.py": _sha256_file(AR_EVALUATION_PATH),
        "reference_runner.py": _sha256_file(REFERENCE_RUNNER_PATH),
        "reference_runner_pre_edit.py": _sha256_file(REFERENCE_SNAPSHOT),
    }


def _load_training_module():
    global _TRAINING
    if _TRAINING is None:
        sys.path.insert(0, str(EXPERIMENT))
        _TRAINING = _load_module(
            "structure_conditioned_hybrid8b_train_for_evaluation", EXPERIMENT / "train.py")
    return _TRAINING


def _load_model(args):
    train = _load_training_module()
    torch = train.torch
    model_path = args.model_path.resolve(strict=True)
    data_dir = args.data.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    for path in (model_path, data_dir, checkpoint_path):
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError(f"project artifact must stay under {ROOT}: {path}")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to one explicitly selected free GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("generation requires a selected CUDA device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("each shard process must expose exactly one selected GPU")
    torch.set_num_threads(4)
    device = torch.device("cuda", torch.cuda.current_device())
    tok = train.load_tokenizer(model_path)
    _, data_hashes, train_rows, val_rows = train.read_frozen_data(data_dir)
    model = train.Qwen3ARBaseline.pretrained(
        tok, str(model_path), gradient_checkpointing=True).to(device).eval()
    model.base.config.use_cache = False
    order_ids = [row["material_id"] for row in train.epoch_order(train_rows)]
    eval_ids = train.BASELINE.select_eval_ids(val_rows)
    identity = train.finalize_identity(
        train.experiment_identity(model_path, train.REVISION, model.base.config.to_dict(),
                                  data_hashes, order_ids, eval_ids),
        data_dir, model_path, tok)
    recorded_identity = json.loads((TRAIN_RUN / "identity.json").read_text(encoding="utf-8"))
    RUNNER.validate_manifest(identity, recorded_identity)

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = json.loads((TRAIN_RUN / "summary.json").read_text(encoding="utf-8"))
    if saved["schema"] != train.CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if saved["fingerprint_sha256"] != identity["fingerprint_sha256"]:
        raise ValueError("checkpoint identity does not match the hybrid training receipt")
    RUNNER.validate_manifest(saved["identity"], identity)
    if saved["completed_updates"] != train.UPDATES or saved["next_material_index"] != train.TRAIN_MATERIALS:
        raise ValueError("checkpoint is not the completed one-epoch hybrid run")
    if (summary.get("status") != "one_epoch_training_complete"
            or summary.get("mode") != "hybrid" or not summary.get("learning_trial")):
        raise ValueError("training receipt does not establish a completed hybrid learning trial")
    if train.state_sha256(saved["added_token_base_rows"]) != saved["added_token_base_rows_sha256"]:
        raise ValueError("checkpoint control-token rows failed their hash check")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(params) != set(saved["trainable_state"]):
        raise ValueError("checkpoint trainable parameter set differs from initialized hybrid model")
    train.restore_control_token_rows(model, saved["added_token_base_rows"])
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved["trainable_state"][name].to(device=param.device, dtype=param.dtype))
    trainable_hash = train.trainable_sha256(model)
    if trainable_hash != saved["trainable_state_sha256"]:
        raise ValueError("restored hybrid trainable state failed its hash check")
    if trainable_hash != summary["final_trainable_state_sha256"]:
        raise ValueError("checkpoint trainable state differs from the completed training receipt")
    del saved
    return train, model, tok, identity, trainable_hash, train.runtime_versions()


def _assert_prefix_cache_parity(model, tok, prompt):
    """Compare cached and full AR logits only on the clean prefix ending at START."""
    import torch

    ids = tok.encode(prompt, add_special_tokens=False)
    sampler = RUNNER.CachedCausalSampler(model, tok)
    contexts = []

    def check(label):
        device = model.time.weight.device
        input_ids = torch.tensor(ids, device=device)
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            reference = model.logits(input_ids, "ar", positions=[len(ids) - 1])[0].float()
        contexts.append({"context": label,
                         **RUNNER._logit_parity_metrics(reference, sampler.logits)})

    sampler.prefill(ids)
    check("prompt")
    for index, token_id in enumerate(tok.encode(" A representative lead", add_special_tokens=False)[:3]):
        sampler.append(token_id)
        ids.append(token_id)
        check(f"text_prefix_token_{index}")
    sampler.append(model.start_id)
    ids.append(model.start_id)
    check("clean_prefix_ending_at_start")
    passed = all(item["distribution_criteria_passed"] for item in contexts)
    return {
        "passed": passed,
        "mode": "prefix-only AR cache parity",
        "interpretation": "The comparison ends at START, before any current-crystal token; hybrid denoising logits are checked by the diffusion sampler contract.",
        "acceptance": {"top1_same": True, "kl_reference_to_cached_max": RUNNER.PARITY_MAX_KL,
                       "max_abs_max": RUNNER.PARITY_MAX_ABS,
                       "mean_abs_max": RUNNER.PARITY_MAX_MEAN_ABS},
        "contexts": contexts,
    }


def _sample_rows(logits, generator, banned_ids, temperature, top_p):
    import torch

    scores = logits.float() / temperature
    scores[:, list(banned_ids)] = -torch.inf
    if top_p < 1:
        sorted_scores, sorted_ids = torch.sort(scores, descending=True, dim=-1)
        sorted_probs = sorted_scores.softmax(-1)
        remove = sorted_probs.cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove_original = torch.zeros_like(remove).scatter_(1, sorted_ids, remove)
        scores[remove_original] = -torch.inf
    probs = scores.softmax(-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


def _apply_releases(current, positions, predictions, uniform_draws, mask_id, k):
    """Release selected MASKs with probability 1/k and retain every known token."""
    updated = list(current)
    for position, prediction, draw in zip(positions, predictions, uniform_draws):
        if updated[position] == mask_id and draw < 1.0 / k:
            updated[position] = prediction
    return updated


def _diffuse_crystal(model, prefix_ids, capacity, steps, generator, banned_ids,
                     temperature, top_p):
    import torch

    device = model.time.weight.device
    current = [model.mask_id] * capacity
    start = len(prefix_ids)
    span = (start, start + capacity)
    forward_steps = 0
    for k in range(steps, 0, -1):
        masked_positions = [index for index, token_id in enumerate(current)
                            if token_id == model.mask_id]
        if not masked_positions:
            continue
        ids = torch.tensor(prefix_ids + current, dtype=torch.long, device=device)
        positions = [start + index for index in masked_positions]
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.logits(ids, "hybrid", span=span, k=k, positions=positions)
            predictions = _sample_rows(logits, generator, banned_ids, temperature, top_p).tolist()
            draws = torch.rand(len(positions), device=device, generator=generator).cpu().tolist()
        current = _apply_releases(current, masked_positions, predictions, draws,
                                  model.mask_id, k)
        forward_steps += 1
    return current, forward_steps


def _sample_attempt(item, model, tok, score, train, checkpoint_id, state_hash, software, out):
    import torch

    started = time.monotonic()
    generator = torch.Generator(device=model.time.weight.device).manual_seed(item["seed"])
    sampler = RUNNER.CachedCausalSampler(model, tok)
    prompt_tokens = tok.encode(item["prompt"], add_special_tokens=False)
    text_started = time.monotonic()
    sampler.prefill(prompt_tokens)
    text_forwards = 1
    lead_tokens = []
    banned_text = {model.mask_id, model.pad_id, model.end_id}
    lead_error = None
    for _ in range(RUNNER.MAX_LEAD_TOKENS):
        token_id = sampler.sample(generator, banned_text)
        lead_tokens.append(token_id)
        if token_id == model.start_id:
            sampler.append(token_id)
            text_forwards += 1
            break
        if token_id == tok.eos_token_id:
            lead_error = "AR response ended before the crystal start token"
            break
        if sampler.length >= train.BASELINE.MAXLEN:
            lead_error = "text context capacity exceeded"
            break
        sampler.append(token_id)
        text_forwards += 1
    else:
        lead_error = f"AR did not emit the crystal start token within {RUNNER.MAX_LEAD_TOKENS} tokens"
    text_seconds = time.monotonic() - text_started
    lead_text = tok.decode(lead_tokens, skip_special_tokens=False)
    receipt = {
        "status": "complete",
        "mode": "hybrid",
        "diffusion_steps": train.STEPS,
        "request_id": item["request_id"],
        "space_group_number_requested": item["space_group_number"],
        "attempt_id": item["attempt_id"],
        "flat_index": item["flat_index"],
        "plan_index": item["plan_index"],
        "seed": item["seed"],
        "prompt": item["prompt"],
        "sampling": {"lead": "stochastic top-p", "crystal": "32-step absorbing-MASK diffusion",
                     "temperature": RUNNER.TEMPERATURE, "top_p": RUNNER.TOP_P,
                     "max_lead_tokens": RUNNER.MAX_LEAD_TOKENS, "crystal_capacity": train.K,
                     "diffusion_steps": train.STEPS, "prediction_offset": PREDICTION_OFFSET},
        "checkpoint_fingerprint_sha256": checkpoint_id,
        "trainable_state_sha256": state_hash,
        "software": software,
        "lead_status": "failed" if lead_error else "ok",
        "lead_error": lead_error,
        "lead_raw_text": lead_text,
        "lead_token_ids": lead_tokens,
        "crystal_status": "skipped" if lead_error else "pending",
        "structural_valid": False,
        "requested_sg_match": False,
        "audit": None,
        "actual_space_group_number": None,
        "cif": None,
        "timing_seconds": {"text_generation": text_seconds},
        "forward_step_counts": {"text": text_forwards, "crystal": 0, "crystal_schedule": 0},
    }
    if lead_error:
        receipt["crystal_error"] = "crystal start was not reached"
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    prefix_ids = prompt_tokens + lead_tokens
    if len(prefix_ids) + train.K > train.BASELINE.MAXLEN:
        raise RuntimeError("clean prefix plus diffusion span exceeds model context capacity")
    banned_crystal = set(tok.all_special_ids) - {model.end_id, model.pad_id}
    crystal_started = time.monotonic()
    raw_tokens, crystal_forwards = _diffuse_crystal(
        model, prefix_ids, train.K, train.STEPS, generator, banned_crystal,
        RUNNER.TEMPERATURE, RUNNER.TOP_P)
    crystal_seconds = time.monotonic() - crystal_started
    raw_text = tok.decode(raw_tokens, skip_special_tokens=False)
    receipt["crystal_raw_token_ids"] = raw_tokens
    receipt["crystal_raw_text"] = raw_text
    receipt["timing_seconds"]["crystal_generation"] = crystal_seconds
    receipt["forward_step_counts"].update(
        {"crystal": crystal_forwards, "crystal_schedule": train.STEPS})
    parse_started = time.monotonic()
    try:
        crystal_text = RUNNER._boundary_content(raw_tokens, model, tok)
        receipt["crystal_text"] = crystal_text
        receipt["crystal_status"] = "ok"
    except RUNNER.GenerationFailure as error:
        receipt["crystal_status"] = "failed"
        receipt["crystal_error"] = str(error)
        receipt["audit"] = {"valid": False, "reasons": [str(error)]}
        receipt["timing_seconds"]["parse"] = time.monotonic() - parse_started
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    audit = score.audit(crystal_text)
    receipt["audit"] = audit
    receipt["structural_valid"] = audit.get("valid") is True
    cif_text = None
    try:
        structure = score.parse_fn_wyckoff(crystal_text, to_cif=False)
        cif_text = structure.to(fmt="cif")
        receipt["actual_space_group_number"] = int(score.SpacegroupAnalyzer(
            structure, symprec=.01, angle_tolerance=5).get_space_group_number())
    except Exception as error:
        receipt["decode_error"] = f"{type(error).__name__}: {error}"
    if cif_text is not None:
        cif_path = out / "attempts" / item["request_id"] / f"attempt-{item['attempt_id']:03d}.cif"
        receipt["cif"] = RUNNER._write_cif(cif_path, cif_text)
    receipt["requested_sg_match"] = RUNNER.requested_sg_matches(
        receipt["actual_space_group_number"], item["space_group_number"])
    receipt["timing_seconds"]["parse"] = time.monotonic() - parse_started
    receipt["timing_seconds"]["total"] = time.monotonic() - started
    return receipt


def _attempt_path(shard_dir, item):
    return shard_dir / "attempts" / item["request_id"] / f"attempt-{item['attempt_id']:03d}.json"


def _validate_receipt(receipt, item, checkpoint_id, state_hash):
    RUNNER._validate_receipt(receipt, item, checkpoint_id, state_hash)
    if receipt.get("mode") != "hybrid" or receipt.get("diffusion_steps") != DIFFUSION_STEPS:
        raise ValueError("existing attempt receipt is not from this hybrid diffusion protocol")


def _augment_summary(summary, receipts, steps):
    summary.update({
        "mode": "hybrid",
        "diffusion_steps": steps,
        "prediction_offset": PREDICTION_OFFSET,
        "random_seed_parameters": {"attempt_plan_seed": RUNNER.SEED,
                                   "per_attempt_seeds": "fixed matched-AR plan"},
        "strict_valid_denominator": summary["raw_attempt_denominator"],
        "forward_step_counts": {
            "text": sum(item.get("forward_step_counts", {}).get("text", 0) for item in receipts),
            "crystal": sum(item.get("forward_step_counts", {}).get("crystal", 0) for item in receipts),
            "crystal_schedule": sum(item.get("forward_step_counts", {}).get("crystal_schedule", 0)
                                     for item in receipts),
        },
    })
    for field in ("text_generation", "crystal_generation", "parse", "total"):
        timed = [item["timing_seconds"][field] for item in receipts
                 if field in item.get("timing_seconds", {})]
        if timed:
            summary.setdefault("mean_timing_seconds", {})[field] = sum(timed) / len(timed)
            summary.setdefault("timing_attempt_counts", {})[field] = len(timed)
    return summary


def summarize_run(out):
    out = Path(out).resolve(strict=True)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("mode") != "hybrid"
            or manifest.get("diffusion_steps") != DIFFUSION_STEPS
            or manifest.get("prediction_offset") != PREDICTION_OFFSET):
        raise ValueError("run manifest is not from this hybrid diffusion protocol")
    summary = RUNNER.summarize_run(out)
    plan = {(item["request_id"], item["attempt_id"]): item for item in manifest["attempt_plan"]}
    receipts = []
    for path in sorted(out.glob("shard-*/attempts/*/attempt-*.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        item = plan[(receipt["request_id"], receipt["attempt_id"])]
        _validate_receipt(receipt, item, manifest["checkpoint_fingerprint_sha256"],
                          manifest["trainable_state_sha256"])
        receipts.append(receipt)
    _augment_summary(summary, receipts, DIFFUSION_STEPS)
    summary["prefix_cache_parity_by_shard"] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out.glob("shard-*/prefix_cache_parity.json"))]
    RUNNER._write_json(out / "summary.json", summary)
    return summary


def run(args):
    plan = RUNNER.attempt_plan(args.limit)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be within 0..num-shards-1")
    shard_plan = [item for item in plan if item["plan_index"] % args.num_shards == args.shard_index]
    if args.mode == "plan":
        print(json.dumps({"attempts": len(plan), "seed": RUNNER.SEED,
                          "seeds_sha256": RUNNER.json_sha256([item["seed"] for item in plan]),
                          "requests": REQUESTS, "mode": "hybrid",
                          "diffusion_steps": DIFFUSION_STEPS,
                          "crystal_capacity": CRYSTAL_CAPACITY}, indent=2))
        return
    out = args.out.resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")
    if args.mode == "summarize":
        print(json.dumps(summarize_run(out), indent=2, ensure_ascii=False))
        return

    train, model, tok, identity, state_hash, software = _load_model(args)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    parity = _assert_prefix_cache_parity(model, tok, REQUESTS[0]["prompt"])
    RUNNER._write_json(shard_dir / "prefix_cache_parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError(f"prefix-only cached logits parity failed; see {shard_dir / 'prefix_cache_parity.json'}")
    run_manifest = {
        "schema": 1,
        "experiment": CONFIG["experiment"],
        "mode": "hybrid",
        "diffusion_steps": train.STEPS,
        "prediction_offset": PREDICTION_OFFSET,
        "random_seed_parameters": {"attempt_plan_seed": RUNNER.SEED,
                                   "per_attempt_seeds": "same fixed plan as matched AR"},
        "model": identity["model"],
        "revision": identity["revision"],
        "checkpoint_fingerprint_sha256": identity["fingerprint_sha256"],
        "trainable_state_sha256": state_hash,
        "training_data_hashes": identity["data_hashes"],
        "training_source_sha256": identity["source_sha256"],
        "evaluation_source_sha256": _evaluation_sources(),
        "requests": REQUESTS,
        "attempts_per_request": RUNNER.ATTEMPTS_PER_REQUEST,
        "attempt_plan": plan,
        "attempt_plan_sha256": RUNNER.json_sha256(plan),
        "limit": args.limit,
        "num_shards": args.num_shards,
        "sampling": {"lead": "stochastic top-p", "crystal": "absorbing-MASK diffusion",
                     "seed": RUNNER.SEED, "temperature": RUNNER.TEMPERATURE,
                     "top_p": RUNNER.TOP_P, "max_lead_tokens": RUNNER.MAX_LEAD_TOKENS,
                     "crystal_capacity": train.K, "diffusion_steps": train.STEPS,
                     "prediction_offset": PREDICTION_OFFSET,
                     "step_schedule": "32 down to 1",
                     "release_probability": "1/k for each still-masked position",
                     "raw_tokens": "no END/PAD boundary repair"},
        "training_prompt_scope": TRAINING_PROMPT_SCOPE,
        "evaluation_scope": EVALUATION_SCOPE,
        "ehull_target": f"<={EHULL_BOUND:g} eV/atom (prompt condition only; property evaluation not run)",
        "cached_logits_parity_contract": {
            "required": True,
            "scope": "causal clean prefix through START only",
            "does_not_compare": "causal crystal logits with bidirectional hybrid denoising logits",
            "strict_elementwise_tolerance_is_diagnostic_only": {
                "rtol": RUNNER.PARITY_STRICT_RTOL, "atol": RUNNER.PARITY_STRICT_ATOL},
            "bf16_distribution_acceptance": {
                "top1_same": True, "kl_reference_to_cached_max": RUNNER.PARITY_MAX_KL,
                "max_abs_max": RUNNER.PARITY_MAX_ABS,
                "mean_abs_max": RUNNER.PARITY_MAX_MEAN_ABS},
            "fp32_tiny_model_route_test_required": True},
        "forward_step_definition": {
            "text": "cached prompt prefill plus one forward per appended lead token, including START",
            "crystal": "one hybrid forward per diffusion step while MASK positions remain",
            "schedule_steps_per_generated_crystal": train.STEPS},
    }
    RUNNER._ensure_manifest(out / "run_manifest.json", run_manifest, args.resume, shared=True)
    shard_manifest = {"shard_index": args.shard_index, "num_shards": args.num_shards,
                      "attempt_plan_indices": [item["plan_index"] for item in shard_plan],
                      "prefix_cache_parity_passed": parity["passed"],
                      "run_manifest_sha256": RUNNER.json_sha256(run_manifest)}
    RUNNER._ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume)
    score = RUNNER._load_score_module()
    receipts = []
    summary_path = shard_dir / "summary.json"
    for item in shard_plan:
        receipt_path = _attempt_path(shard_dir, item)
        if receipt_path.exists():
            if not args.resume:
                raise FileExistsError(f"{receipt_path} exists; pass --resume")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            _validate_receipt(receipt, item, identity["fingerprint_sha256"], state_hash)
        else:
            receipt = _sample_attempt(item, model, tok, score, train,
                                      identity["fingerprint_sha256"], state_hash,
                                      software, shard_dir)
            RUNNER._write_json(receipt_path, receipt)
        receipts.append(receipt)
        summary = _augment_summary(RUNNER.summarize(shard_plan, receipts), receipts, train.STEPS)
        summary.update({"run_manifest_sha256": RUNNER.json_sha256(run_manifest),
                        "shard": shard_manifest,
                        "prefix_cache_parity": parity,
                        "temperature": RUNNER.TEMPERATURE, "top_p": RUNNER.TOP_P,
                        "ehull_evaluation_status": "not_run"})
        RUNNER._write_json(summary_path, summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "plan", "summarize"), default="sample")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model-path", type=Path, default=MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int,
                        help="limit attempts per request; --limit 1 gives seven preflight attempts")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
