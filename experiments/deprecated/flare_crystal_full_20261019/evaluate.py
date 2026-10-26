"""Generate, structurally audit, read back, and score the fixed 700 requests."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
AR_EVALUATOR_PATH = ROOT / "experiments/structure_conditioned_ar8b_20261017/evaluate.py"
REFERENCE_RUNNER_PATH = ROOT / "experiments/structure_request_eval_20261017/run.py"
REFERENCE_SCORE_PATH = ROOT / "experiments/interleaved_ar_diffusion_20261002/score.py"
PILOT_MODEL_PATH = ROOT / "experiments/flare_crystal_qwen3_8b_20261019/model.py"
DEFAULT_CHECKPOINTS = {arm: EXPERIMENT / arm / "checkpoint.pt" for arm in ("ar", "flare")}
TEMPERATURE, TOP_P = .8, .95
MAX_READBACK_TOKENS = 128
READBACK_FIELDS = ("formula", "space_group", "cell_atom_count", "independent_orbit_count")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AR_EVALUATOR = _load_module("flare_full_matched_ar_evaluator", AR_EVALUATOR_PATH)
RUNNER = AR_EVALUATOR.RUNNER
RUNNER.EXPERIMENT = EXPERIMENT.resolve()
RUNNER.SHARED_RUN = EXPERIMENT.resolve()
RUNNER.TRAINING_PROMPT_SCOPE = "G requests contain space group and ehull conditions; Q targets remain frozen factual readback tasks"
RUNNER.EVALUATION_SCOPE = "700 independent requests per arm; model-generated lead and crystal; separate four-field readback and GRACE scoring"
RUNNER.REQUESTS = AR_EVALUATOR.RUNNER.REQUESTS
OBJECTIVE = _load_module("flare_full_verified_denoiser", EXPERIMENT / "model.py")
READBACK_PATH = ROOT / "experiments/evaluation_common/readback.py"
READBACK = _load_module("canonical_four_fact_readback", READBACK_PATH)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_sources():
    return {
        "evaluate.py": _sha256_file(EXPERIMENT / "evaluate.py"),
        "trainer_run.py": _sha256_file(EXPERIMENT / "run.py"),
        "flare_model.py": _sha256_file(EXPERIMENT / "model.py"),
        "verified_pilot_model.py": _sha256_file(PILOT_MODEL_PATH),
        "matched_ar_evaluate.py": _sha256_file(AR_EVALUATOR_PATH),
        "reference_runner.py": _sha256_file(REFERENCE_RUNNER_PATH),
        "structure_score.py": _sha256_file(REFERENCE_SCORE_PATH),
        "readback_grader.py": _sha256_file(READBACK_PATH),
    }


def _request_fingerprint(item):
    return RUNNER.json_sha256({key: item[key] for key in
                               ("request_id", "attempt_id", "prompt", "space_group_number", "seed")})


def _commit_highest_confidence(current, positions, sampled_ids, confidences):
    """Commit the one sampled position with the highest model probability."""
    chosen = max(range(len(positions)), key=lambda i: (float(confidences[i]), -i))
    updated = list(current)
    updated[positions[chosen]] = int(sampled_ids[chosen])
    return updated, positions[chosen], int(sampled_ids[chosen])


def _truncate_at_end(token_ids, end_id):
    try:
        return list(token_ids[:token_ids.index(end_id) + 1]), True
    except ValueError:
        return list(token_ids), False


def _sample_rows(logits, generator, banned_ids, temperature=TEMPERATURE, top_p=TOP_P):
    import torch

    scores = logits.float() / temperature
    scores[:, list(banned_ids)] = -torch.inf
    confidence = scores.softmax(-1)
    sampled_scores = scores.clone()
    if top_p < 1:
        sorted_scores, sorted_ids = torch.sort(sampled_scores, descending=True, dim=-1)
        sorted_probs = sorted_scores.softmax(-1)
        remove = sorted_probs.cumsum(-1) > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sampled_scores[torch.zeros_like(remove).scatter_(1, sorted_ids, remove)] = -torch.inf
    sampled = torch.multinomial(sampled_scores.softmax(-1), 1, generator=generator).squeeze(-1)
    confidence = confidence.gather(1, sampled[:, None]).squeeze(1)
    return sampled, confidence


def _sample_flare_block(model, prefix, completed, size, train, generator, banned_ids):
    import torch

    current = [model.mask_id] * size
    device = model.time.weight.device
    for _ in range(size):
        positions = [index for index, token in enumerate(current) if token == model.mask_id]
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = OBJECTIVE.denoise_logits(model, prefix + completed, current, len(prefix))
            sampled, confidence = _sample_rows(logits[positions], generator, banned_ids)
        current, _, _ = _commit_highest_confidence(
            current, positions, sampled.tolist(), confidence.tolist())
    return current


def _sample_crystal(arm, prefix, model, tok, train, generator):
    banned = set(tok.all_special_ids) - {model.end_id}
    banned.add(model.pad_id)
    if arm == "ar":
        sampler = RUNNER.CachedCausalSampler(model, tok)
        sampler.prefill(prefix)
        raw = []
        for _ in range(train.K):
            token = sampler.sample(generator, banned)
            raw.append(token)
            if token == model.end_id:
                break
            sampler.append(token)
        return raw, len(raw)

    block_size = OBJECTIVE.BLOCK_SIZE
    raw, forwards = [], 0
    while len(raw) < train.K and model.end_id not in raw:
        block = _sample_flare_block(model, prefix, raw,
                                    min(block_size, train.K - len(raw)), train,
                                    generator, banned)
        raw.extend(block)
        forwards += len(block)
        raw, ended = _truncate_at_end(raw, model.end_id)
        if ended:
            break
    return raw, forwards


def _readback(model, tok, prefix_and_crystal, question):
    import torch

    sampler = RUNNER.CachedCausalSampler(model, tok)
    suffix = tok.encode("\nUser: " + question + "\nAssistant:", add_special_tokens=False)
    sampler.prefill(prefix_and_crystal + suffix)
    banned = set(tok.all_special_ids) - {tok.eos_token_id}
    generated = []
    for _ in range(MAX_READBACK_TOKENS):
        logits = sampler.logits.clone()
        logits[list(banned)] = -torch.inf
        token = int(logits.argmax())
        generated.append(token)
        if token == tok.eos_token_id:
            return tok.decode(generated[:-1], skip_special_tokens=False), generated, "eos"
        sampler.append(token)
    return tok.decode(generated, skip_special_tokens=False), generated, "max_tokens"


def _grade_readback(score, response, body, structure, actual_sg):
    parsed = score.parse(body)
    graded = READBACK.grade_four_facts(response, parsed['formula'], actual_sg,
                                     len(structure), parsed['orbits'])
    fields = graded['fields']
    return {
        "question": None,
        "response": response,
        "status": None,
        "fields": fields,
        "all_four_correct": all(fields.values()),
        "evaluator_version": READBACK.VERSION,
        "reference_facts": {
            "validated_cell_formula": parsed["formula"],
            "header_space_group_number": parsed["sg"],
            "decoded_space_group_number": int(actual_sg),
            "decoded_cell_atom_count": len(structure),
            "validated_independent_orbit_count": parsed["orbits"],
            "scope": "formula/counts from decoded composition; orbit count from validated Wyckoff records; header and decoded SG retained separately",
        },
    }


def _validate_receipt(receipt, item, manifest):
    if (receipt.get("request_id"), receipt.get("attempt_id"), receipt.get("seed")) != (
            item["request_id"], item["attempt_id"], item["seed"]):
        raise ValueError("existing attempt receipt does not match its fixed request and seed")
    if receipt.get("arm") != manifest["arm"]:
        raise ValueError("existing attempt receipt belongs to another arm")
    if receipt.get("checkpoint_fingerprint_sha256") != manifest["checkpoint_fingerprint_sha256"]:
        raise ValueError("existing attempt receipt belongs to another checkpoint")
    if receipt.get("trainable_state_sha256") != manifest["trainable_state_sha256"]:
        raise ValueError("existing attempt receipt has a different trained state")
    if receipt.get("evaluation_source_sha256") != manifest["evaluation_source_sha256"]:
        raise ValueError("existing attempt receipt has a different code fingerprint")
    if receipt.get("request_fingerprint_sha256") != _request_fingerprint(item):
        raise ValueError("existing attempt receipt has a different request fingerprint")


def _attempt_path(shard_dir, item):
    return shard_dir / "attempts" / item["request_id"] / f"attempt-{item['attempt_id']:03d}.json"


def _sample_attempt(item, arm, model, tok, score, train, identity, source_hash,
                    checkpoint_hash, state_hash, software, sources, shard_dir, question):
    import torch

    started = time.monotonic()
    generator = torch.Generator(device=model.time.weight.device).manual_seed(item["seed"])
    prompt_ids = tok.encode(item["prompt"], add_special_tokens=False)
    sampler = RUNNER.CachedCausalSampler(model, tok)
    sampler.prefill(prompt_ids)
    lead_ids, lead_error = [], None
    banned_text = {model.mask_id, model.pad_id, model.end_id}
    for _ in range(RUNNER.MAX_LEAD_TOKENS):
        token = sampler.sample(generator, banned_text)
        lead_ids.append(token)
        if token == model.start_id:
            sampler.append(token)
            break
        if token == tok.eos_token_id:
            lead_error = "AR response ended before the crystal start token"
            break
        if sampler.length >= train.BASELINE.MAXLEN:
            lead_error = "text context capacity exceeded"
            break
        sampler.append(token)
    else:
        lead_error = f"AR did not emit the crystal start token within {RUNNER.MAX_LEAD_TOKENS} tokens"
    receipt = {
        "status": "complete", "arm": arm, "request_id": item["request_id"],
        "space_group_number_requested": item["space_group_number"],
        "attempt_id": item["attempt_id"], "flat_index": item["flat_index"],
        "plan_index": item["plan_index"], "seed": item["seed"], "prompt": item["prompt"],
        "request_fingerprint_sha256": _request_fingerprint(item),
        "checkpoint_fingerprint_sha256": checkpoint_hash,
        "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
        "source_trainable_state_sha256": source_hash,
        "trainable_state_sha256": state_hash, "evaluation_source_sha256": sources,
        "software": software,
        "sampling": {"lead": "stochastic top-p", "crystal": "causal AR" if arm == "ar" else "B=4 block denoising; commit highest sampled-probability position",
                     "temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_lead_tokens": RUNNER.MAX_LEAD_TOKENS,
                     "crystal_capacity": train.K, "pad_banned": True},
        "lead_status": "failed" if lead_error else "ok", "lead_error": lead_error,
        "lead_token_ids": lead_ids,
        "lead_raw_text": tok.decode(lead_ids, skip_special_tokens=False),
        "crystal_status": "skipped" if lead_error else "pending",
        "crystal_raw_token_ids": [], "crystal_raw_text": "",
        "structural_valid": False, "requested_sg_match": False,
        "audit": None, "actual_space_group_number": None, "cif": None,
        "readback": {"status": "skipped", "reason": "crystal generation did not start" if lead_error else "awaiting structural audit"},
        "timing_seconds": {"total_started": time.monotonic() - started},
    }
    if lead_error:
        receipt["crystal_error"] = "crystal start was not reached"
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    prefix = prompt_ids + lead_ids
    if len(prefix) + train.K > train.BASELINE.MAXLEN:
        receipt["crystal_status"] = "failed"
        receipt["crystal_error"] = "clean prefix plus crystal capacity exceeds model context"
        receipt["readback"] = {"status": "skipped", "reason": "crystal context capacity exceeded"}
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt
    crystal_started = time.monotonic()
    raw, forwards = _sample_crystal(arm, prefix, model, tok, train, generator)
    receipt["timing_seconds"]["crystal_generation"] = time.monotonic() - crystal_started
    receipt["forward_steps"] = {"crystal": forwards}
    raw, ended = _truncate_at_end(raw, model.end_id)
    receipt["crystal_raw_token_ids"] = raw
    receipt["crystal_raw_text"] = tok.decode(raw, skip_special_tokens=False)
    try:
        body = RUNNER._boundary_content(raw, model, tok)
        receipt["crystal_text"] = body
        receipt["crystal_status"] = "ok"
    except RUNNER.GenerationFailure as error:
        receipt["crystal_status"] = "failed"
        receipt["crystal_error"] = str(error)
        receipt["audit"] = {"valid": False, "reasons": [str(error)]}
        receipt["readback"] = {"status": "skipped", "reason": "crystal block did not end with END"}
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    audit_started = time.monotonic()
    audit = score.audit(body)
    receipt["audit"] = audit
    receipt["structural_valid"] = audit.get("valid") is True
    structure = None
    try:
        structure = score.parse_fn_wyckoff(body, to_cif=False)
        receipt["actual_space_group_number"] = int(score.SpacegroupAnalyzer(
            structure, symprec=.01, angle_tolerance=5).get_space_group_number())
        cif_path = shard_dir / "attempts" / item["request_id"] / f"attempt-{item['attempt_id']:03d}.cif"
        cif = RUNNER._write_cif(cif_path, structure.to(fmt="cif"))
        cif["path"] = str(Path(cif["path"]).resolve().relative_to(shard_dir.parent.resolve()))
        receipt["cif"] = cif
    except Exception as error:
        receipt["decode_error"] = f"{type(error).__name__}: {error}"
    receipt["requested_sg_match"] = RUNNER.requested_sg_matches(
        receipt["actual_space_group_number"], item["space_group_number"])
    receipt["timing_seconds"]["audit_and_decode"] = time.monotonic() - audit_started
    if not receipt["structural_valid"]:
        receipt["readback"] = {"status": "skipped", "reason": "strict structural audit failed"}
    elif structure is None:
        receipt["readback"] = {"status": "skipped", "reason": "validated structure could not be decoded for readback"}
    else:
        readback_started = time.monotonic()
        answer, answer_ids, readback_status = _readback(
            model, tok, prefix + raw, question)
        graded = _grade_readback(score, answer, body, structure,
                                 receipt["actual_space_group_number"])
        graded.update({"question": question, "status": readback_status,
                       "token_ids": answer_ids,
                       "timing_seconds": time.monotonic() - readback_started})
        receipt["readback"] = graded
    receipt["timing_seconds"]["total"] = time.monotonic() - started
    return receipt


def _read_receipts(out, manifest):
    plan = manifest["attempt_plan"]
    receipts = []
    for item in plan:
        candidates = list(Path(out).glob(
            f"shard-*/attempts/{item['request_id']}/attempt-{item['attempt_id']:03d}.json"))
        if len(candidates) > 1:
            raise ValueError(f"duplicate attempt receipts for plan slot {item['plan_index']}")
        if candidates:
            receipt = json.loads(candidates[0].read_text(encoding="utf-8"))
            _validate_receipt(receipt, item, manifest)
            receipts.append(receipt)
    return receipts


def _augment_summary(summary, receipts):
    attempted = [r.get("readback", {}) for r in receipts if r.get("structural_valid") is True]
    summary.update({
        "readback_attempts": len(attempted),
        "readback_skipped_invalid_structures": sum(r.get("structural_valid") is not True for r in receipts),
        "readback_eos_completed": sum(item.get("status") == "eos" for item in attempted),
        "readback_all_four_correct": sum(item.get("all_four_correct") is True for item in attempted),
        "readback_field_correct_counts": {
            field: sum(item.get("fields", {}).get(field) is True for item in attempted)
            for field in READBACK_FIELDS},
        "readback_denominator": len(attempted),
        "quality_scope": "readback is separate from structural validity and GRACE property scoring",
    })
    return summary


def summarize_run(out):
    out = Path(out).resolve(strict=True)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    receipts = _read_receipts(out, manifest)
    summary = RUNNER.summarize(manifest["attempt_plan"], receipts)
    summary.update({"arm": manifest["arm"],
                    "run_manifest_sha256": RUNNER.json_sha256(manifest),
                    "requests": manifest["requests"],
                    "attempt_plan_sha256": manifest["attempt_plan_sha256"],
                    "checkpoint_fingerprint_sha256": manifest["checkpoint_fingerprint_sha256"],
                    "trainable_state_sha256": manifest["trainable_state_sha256"],
                    "ehull_evaluation_status": "pending_grace_score"})
    _augment_summary(summary, receipts)
    RUNNER._write_json(out / "summary.json", summary)
    return summary


def run(args):
    plan = RUNNER.attempt_plan(args.limit)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be within 0..num-shards-1")
    shard_plan = [item for item in plan if item["plan_index"] % args.num_shards == args.shard_index]
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")
    if args.mode == "summarize":
        if not (out / "run_manifest.json").is_file():
            raise FileNotFoundError(f"missing run manifest: {out}")
        summary = summarize_run(out)
        if summary["arm"] != args.arm:
            raise ValueError("requested arm differs from run manifest")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    trainer = _load_module("flare_full_training_runner", EXPERIMENT / "run.py")
    train, model, tok, identity, source_hash, software = trainer.load_model()
    checkpoint_path = (args.checkpoint or DEFAULT_CHECKPOINTS[args.arm]).expanduser().resolve(strict=True)
    if not checkpoint_path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"checkpoint must stay under project root: {checkpoint_path}")
    checkpoint = trainer.load_checkpoint(model, checkpoint_path, train, args.arm)
    state_hash = train.trainable_sha256(model)
    if checkpoint.get("trainable_state_sha256") != state_hash:
        raise ValueError("restored arm checkpoint does not match its recorded state hash")
    checkpoint_hash = _sha256_file(checkpoint_path)
    sources = _evaluation_sources()
    code_hash = RUNNER.json_sha256(sources)
    _, data_hashes, train_rows, _ = train.read_frozen_data(train.DEFAULT_DATA)
    if identity.get("data_hashes") and identity["data_hashes"] != data_hashes:
        raise ValueError("loaded source identity does not match the frozen data")
    question = train.tasks(train_rows[0])[1]["question"]
    run_manifest = {
        "schema": 1, "experiment": EXPERIMENT.name, "arm": args.arm,
        "model": identity["model"], "revision": identity["revision"],
        "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
        "source_trainable_state_sha256": source_hash,
        "checkpoint_fingerprint_sha256": checkpoint_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "trainable_state_sha256": state_hash,
        "training_data_hashes": data_hashes,
        "training_source_sha256": identity.get("source_sha256", {}),
        "evaluation_source_sha256": sources, "evaluation_code_fingerprint_sha256": code_hash,
        "requests": RUNNER.REQUESTS, "attempts_per_request": RUNNER.ATTEMPTS_PER_REQUEST,
        "attempt_plan": plan, "attempt_plan_sha256": RUNNER.json_sha256(plan),
        "request_set_fingerprint_sha256": RUNNER.json_sha256(RUNNER.REQUESTS),
        "limit": args.limit, "num_shards": args.num_shards,
        "readback_question": question,
        "sampling": {"lead": "stochastic top-p", "crystal": "causal AR" if args.arm == "ar" else "B=4 FLARE denoising; one highest sampled-probability commit per forward",
                     "temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_lead_tokens": RUNNER.MAX_LEAD_TOKENS,
                     "crystal_capacity": train.K, "readback": "greedy AR, max 128 tokens, stop at EOS",
                     "pad_banned": True},
        "ehull_target": "<=0.08 eV/atom (request condition only; property score is separate)",
        "evaluation_scope": RUNNER.EVALUATION_SCOPE,
    }
    RUNNER._ensure_manifest(out / "run_manifest.json", run_manifest, args.resume, shared=True)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    shard_manifest = {"arm": args.arm, "shard_index": args.shard_index,
                      "num_shards": args.num_shards,
                      "attempt_plan_indices": [item["plan_index"] for item in shard_plan],
                      "run_manifest_sha256": RUNNER.json_sha256(run_manifest),
                      "evaluation_code_fingerprint_sha256": code_hash}
    RUNNER._ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume)
    score = RUNNER._load_score_module()
    receipts = []
    for item in shard_plan:
        receipt_path = _attempt_path(shard_dir, item)
        if receipt_path.exists():
            if not args.resume:
                raise FileExistsError(f"{receipt_path} exists; pass --resume")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            _validate_receipt(receipt, item, run_manifest)
        else:
            receipt = _sample_attempt(item, args.arm, model, tok, score, train,
                                      identity, source_hash, checkpoint_hash, state_hash,
                                      software, sources, shard_dir, question)
            RUNNER._write_json(receipt_path, receipt)
        receipts.append(receipt)
        summary = RUNNER.summarize(shard_plan, receipts)
        summary.update({"arm": args.arm, "shard": shard_manifest,
                        "run_manifest_sha256": RUNNER.json_sha256(run_manifest),
                        "ehull_evaluation_status": "pending_grace_score"})
        _augment_summary(summary, receipts)
        RUNNER._write_json(shard_dir / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "summarize"), default="sample")
    parser.add_argument("--arm", choices=("ar", "flare"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--limit", type=int, help="attempts per space group; default is 100")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
