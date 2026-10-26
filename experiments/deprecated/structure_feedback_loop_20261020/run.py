"""Run paired independent-resampling and structure-feedback retries on frozen FLARE."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVALUATOR_PATH = ROOT / "experiments/flare_crystal_full_20261019/evaluate.py"
TRAINER_PATH = ROOT / "experiments/flare_crystal_full_20261019/run.py"
MAX_LEAD_TOKENS, CRYSTAL_CAPACITY, MAX_CONTEXT, MAX_ATTEMPTS = 256, 544, 4096, 3
ARMS = ("independent", "feedback")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = _load_module("structure_feedback_frozen_evaluator", EVALUATOR_PATH)
RUNNER = E.RUNNER


def retry_seed(seed, round_index):
    digest = hashlib.sha256(f"{seed}:{round_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


def _continue_arm(attempts, seed, generate, is_success, persist=lambda _attempt: None):
    if not attempts or len(attempts) > MAX_ATTEMPTS:
        raise ValueError("an arm must start with one to three persisted attempts")
    if any(is_success(attempt) for attempt in attempts[:-1]):
        raise ValueError("persisted arm continued after a successful candidate")
    for round_index in range(len(attempts), MAX_ATTEMPTS):
        if is_success(attempts[-1]):
            break
        attempt = generate(round_index, retry_seed(seed, round_index), attempts[-1])
        attempts.append(attempt)
        persist(attempt)
    return attempts


def _candidate_success(audit, actual_sg, requested_sg):
    return audit is not None and audit.get("valid") is True and actual_sg == int(requested_sg)


def _feedback_context(original_prompt_ids, previous, feedback_ids, start_id, end_id):
    context = list(original_prompt_ids)
    lead_ids = previous["lead_token_ids"]
    crystal_ids = previous["crystal_raw_token_ids"]
    context.extend(lead_ids)
    context.extend(crystal_ids)
    opened = start_id in lead_ids
    synthetic_end_added = opened and end_id not in crystal_ids
    if synthetic_end_added:
        context.append(end_id)
    context.extend(feedback_ids)
    return context, synthetic_end_added


def _denoise_layout(clean_prefix, current, start_id, end_id):
    if not clean_prefix:
        raise ValueError("denoising needs a nonempty clean prefix for its anchor")
    clean_count, block_count = len(clean_prefix), len(current)
    ids = list(clean_prefix) + [clean_prefix[-1]] + list(current)
    position_ids = list(range(clean_count)) + [clean_count - 1] + list(range(clean_count, clean_count + block_count))
    times = RUNNER.attention_time_values(clean_prefix, start_id, end_id)
    time_ids = times + [times[-1]] + [0] * block_count
    n = len(ids)
    allowed = [[False] * n for _ in range(n)]
    for query in range(clean_count):
        allowed[query][:query + 1] = [True] * (query + 1)
    for query in range(clean_count, n):
        allowed[query][:] = [True] * n
    return dict(ids=ids, position_ids=position_ids, time_ids=time_ids, allowed=allowed,
                clean_count=clean_count, block_count=block_count)


def _denoise_logits(model, clean_prefix, current):
    import torch

    plan = _denoise_layout(clean_prefix, current, model.start_id, model.end_id)
    device = model.time.weight.device
    ids = torch.tensor(plan["ids"], dtype=torch.long, device=device)[None]
    positions = torch.tensor(plan["position_ids"], dtype=torch.long, device=device)[None]
    times = torch.tensor(plan["time_ids"], dtype=torch.long, device=device)[None]
    embedded = model.base.get_input_embeddings()(ids)
    embedded = embedded + model.time(times.clamp_min(0)) * (times >= 0).unsqueeze(-1)
    permitted = torch.tensor(plan["allowed"], dtype=torch.bool, device=device)[None, None]
    additive = torch.zeros_like(permitted, dtype=embedded.dtype).masked_fill(
        ~permitted, torch.finfo(embedded.dtype).min)
    hidden = model.base.model(inputs_embeds=embedded, position_ids=positions,
        attention_mask={"full_attention": additive}, use_cache=False).last_hidden_state[0]
    begin, end = plan["clean_count"], plan["clean_count"] + plan["block_count"]
    return model.base.lm_head(hidden[begin:end]).float()


def _checker_feedback(receipt, requested_sg):
    audit = receipt.get("audit") or {}
    actual = receipt.get("actual_space_group_number")
    actual_text = str(actual) if actual is not None else "undetermined"
    facts = audit.get("facts") or {}
    fact_text = ", ".join(f"{key}={facts[key]}" for key in ("formula", "sg", "atoms", "orbits")
                           if key in facts) or "none recorded"
    reasons = audit.get("reasons") or []
    reason_text = ", ".join(map(str, reasons)) or "none recorded"
    status = "passed" if audit.get("valid") is True else "failed" if audit else "not run"
    error = receipt.get("lead_error") or receipt.get("crystal_error")
    error_text = f" Recorded generation error: {error}." if error else ""
    return (f"Checker feedback: the original request requires space group {int(requested_sg)} and "
            "energy above hull no greater than 0.08 eV/atom. The decoded space group is "
            f"{actual_text}. The strict structural audit {status}; recorded reasons: {reason_text}. "
            f"Recorded structure facts: {fact_text}.{error_text} The ehull target has not been "
            "evaluated. Write a brief revised design description, then generate a new crystal "
            "in the same format.")


def _max_positions(model):
    return int(model.base.config.max_position_embeddings)


def _assert_context_budget(context_ids, model):
    max_positions = _max_positions(model)
    assert MAX_CONTEXT <= max_positions, (
        f"the frozen model supports {max_positions} tokens, below the {MAX_CONTEXT}-token pilot budget")
    assert len(context_ids) + MAX_LEAD_TOKENS + CRYSTAL_CAPACITY <= MAX_CONTEXT, (
        "prompt plus maximum lead and crystal capacity exceeds the 4096-token pilot context")


def _sample_lead(context_ids, model, tok, generator):
    sampler = RUNNER.CachedCausalSampler(model, tok, E.TEMPERATURE, E.TOP_P)
    sampler.prefill(context_ids)
    generated, error = [], None
    banned = {model.mask_id, model.pad_id, model.end_id}
    for _ in range(MAX_LEAD_TOKENS):
        token = sampler.sample(generator, banned)
        generated.append(token)
        if token == model.start_id:
            sampler.append(token)
            break
        if token == tok.eos_token_id:
            error = "AR response ended before the crystal start token"
            break
        sampler.append(token)
    else:
        error = f"AR did not emit the crystal start token within {MAX_LEAD_TOKENS} tokens"
    return generated, error


def _sample_flare_block(model, clean_prefix, size, generator, banned_ids):
    import torch

    current = [model.mask_id] * size
    device = model.time.weight.device
    for _ in range(size):
        positions = [index for index, token in enumerate(current) if token == model.mask_id]
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = _denoise_logits(model, clean_prefix, current)
            sampled, confidence = E._sample_rows(logits[positions], generator, banned_ids)
        current, _, _ = E._commit_highest_confidence(
            current, positions, sampled.tolist(), confidence.tolist())
    return current


def _sample_crystal(prefix_ids, model, tok, generator):
    banned = set(tok.all_special_ids) - {model.end_id}
    banned.add(model.pad_id)
    generated, forwards = [], 0
    while len(generated) < CRYSTAL_CAPACITY and model.end_id not in generated:
        size = min(E.OBJECTIVE.BLOCK_SIZE, CRYSTAL_CAPACITY - len(generated))
        block = _sample_flare_block(model, prefix_ids + generated, size, generator, banned)
        generated.extend(block)
        forwards += len(block)
    raw, ended = E._truncate_at_end(generated, model.end_id)
    return generated, raw, forwards, ended


def _candidate_attempt(item, policy, round_index, seed, context_ids, feedback, synthetic_end,
                       model, tok, train, score, identity, checkpoint_hash, state_hash,
                       code_hashes, cif_root):
    import torch

    started = time.monotonic()
    prompt_ids = list(context_ids)
    receipt = {
        "request_id": item["request_id"], "group_index": item["attempt_id"],
        "plan_index": item["plan_index"], "round": round_index, "seed": seed,
        "policy": policy, "prompt": item["prompt"], "context_token_ids": prompt_ids,
        "feedback": feedback, "feedback_context_synthetic_end_added": synthetic_end,
        "checkpoint_fingerprint_sha256": checkpoint_hash,
        "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
        "trainable_state_sha256": state_hash, "evaluation_source_sha256": code_hashes,
        "sampling": {"lead": "stochastic top-p AR", "crystal": "B=4 FLARE shifted-block denoising",
                     "temperature": E.TEMPERATURE, "top_p": E.TOP_P,
                     "max_lead_tokens": MAX_LEAD_TOKENS, "crystal_capacity": CRYSTAL_CAPACITY},
        "lead_status": "not_started", "lead_error": None, "lead_token_ids": [],
        "lead_raw_text": "", "crystal_status": "not_started",
        "crystal_prefix_token_ids": [], "crystal_generated_token_ids": [],
        "crystal_generated_text": "", "crystal_raw_token_ids": [], "crystal_raw_text": "",
        "structural_valid": False, "requested_sg_match": False,
        "actual_space_group_number": None, "audit": None, "facts": None, "cif": None,
        "ehull_evaluation_status": "not_evaluated",
        "failure_categories": [], "success": False,
        "timing_seconds": {"total": 0.0},
    }
    try:
        _assert_context_budget(prompt_ids, model)
    except AssertionError as error:
        receipt.update(lead_status="failed", lead_error=str(error), crystal_status="skipped",
                       failure_categories=["context_overflow"])
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    generator = torch.Generator(device=model.time.weight.device).manual_seed(seed)
    lead_ids, lead_error = _sample_lead(prompt_ids, model, tok, generator)
    receipt["lead_token_ids"] = lead_ids
    receipt["lead_raw_text"] = tok.decode(lead_ids, skip_special_tokens=False)
    if lead_error:
        receipt.update(lead_status="failed", lead_error=lead_error, crystal_status="skipped",
                       failure_categories=["no_start"])
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    receipt["lead_status"] = "ok"
    crystal_prefix = prompt_ids + lead_ids
    receipt["crystal_prefix_token_ids"] = crystal_prefix
    assert len(crystal_prefix) + CRYSTAL_CAPACITY <= _max_positions(model)
    generated, raw, forwards, ended = _sample_crystal(crystal_prefix, model, tok, generator)
    receipt.update(crystal_status="ok" if ended else "failed",
                   crystal_generated_token_ids=generated,
                   crystal_generated_text=tok.decode(generated, skip_special_tokens=False),
                   crystal_raw_token_ids=raw,
                   crystal_raw_text=tok.decode(raw, skip_special_tokens=False),
                   forward_steps={"crystal": forwards})
    if not ended:
        receipt["crystal_error"] = "crystal block did not emit END within 544 token positions"
        receipt["failure_categories"] = ["no_end"]
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    try:
        body = RUNNER._boundary_content(raw, model, tok)
    except RUNNER.GenerationFailure as error:
        receipt["crystal_status"] = "failed"
        receipt["crystal_error"] = str(error)
        receipt["audit"] = {"valid": False, "reasons": [str(error)]}
        receipt["failure_categories"] = ["boundary_invalid"]
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    audit = score.audit(body)
    actual_sg = audit.get("detected_sg")
    structure = None
    if actual_sg is None or audit.get("valid") is True:
        try:
            structure = score.parse_fn_wyckoff(body, to_cif=False)
            actual_sg = int(score.SpacegroupAnalyzer(
                structure, symprec=.01, angle_tolerance=5).get_space_group_number())
        except Exception as error:
            if audit.get("valid") is True:
                raise
            receipt["decoded_space_group_error"] = f"{type(error).__name__}: {error}"
    matched = RUNNER.requested_sg_matches(actual_sg, item["space_group_number"])
    valid = audit.get("valid") is True
    receipt.update(audit=audit, facts=audit.get("facts"), structural_valid=valid,
                   actual_space_group_number=actual_sg, requested_sg_match=matched,
                   success=_candidate_success(audit, actual_sg, item["space_group_number"]))
    if not valid:
        receipt["failure_categories"].append("strict_invalid")
        receipt["failure_categories"].extend(str(reason) for reason in audit.get("reasons", []))
    if actual_sg is not None and not matched:
        receipt["failure_categories"].append("wrong_requested_sg")
    if valid:
        if structure is None:
            structure = score.parse_fn_wyckoff(body, to_cif=False)
        cif_path = cif_root / f"round-{round_index:02d}-{policy}.cif"
        cif = RUNNER._write_cif(cif_path, structure.to(fmt="cif"))
        cif["path"] = str(Path(cif["path"]).resolve().relative_to(cif_root.parents[3].resolve()))
        receipt["cif"] = cif
    receipt["timing_seconds"]["total"] = time.monotonic() - started
    return receipt


def _seed_manifest(item):
    return {"shared_initial": item["seed"], "paired_retry_seeds": {
        str(round_index): retry_seed(item["seed"], round_index)
        for round_index in (1, 2)}}


def _group_path(shard_dir, item):
    return (shard_dir / "groups" / item["request_id"] /
            f"group-{item['attempt_id']:03d}.json")


def _validate_group(group, item, identity_hash, checkpoint_hash, state_hash, code_hashes):
    if (group.get("request_id"), group.get("group_index"), group.get("plan_index"),
            group.get("original_seed"), group.get("plan_item"),
            group.get("seed_manifest"), group.get("run_identity_sha256")) != (
            item["request_id"], item["attempt_id"], item["plan_index"], item["seed"],
            item, _seed_manifest(item), identity_hash):
        raise ValueError("existing group receipt does not match the frozen plan, seeds, or run identity")
    if (group.get("checkpoint_fingerprint_sha256") != checkpoint_hash or
            group.get("trainable_state_sha256") != state_hash or
            group.get("evaluation_source_sha256") != code_hashes):
        raise ValueError("existing group receipt belongs to another checkpoint or code revision")
    arms = group["arms"]
    for arm in ARMS:
        attempts = arms[arm]["attempts"]
        if not 1 <= len(attempts) <= MAX_ATTEMPTS or arms[arm]["attempts_used"] != len(attempts):
            raise ValueError("existing group receipt has an invalid attempt count")
        for round_index, attempt in enumerate(attempts):
            expected_seed = item["seed"] if round_index == 0 else retry_seed(item["seed"], round_index)
            if attempt.get("round") != round_index or attempt.get("seed") != expected_seed:
                raise ValueError("existing group receipt has a different paired attempt seed")
            if attempt.get("checkpoint_fingerprint_sha256") != group["checkpoint_fingerprint_sha256"]:
                raise ValueError("existing attempt receipt belongs to another checkpoint")
            if attempt.get("evaluation_source_sha256") != group["evaluation_source_sha256"]:
                raise ValueError("existing attempt receipt has a different code identity")
            if (attempt.get("trainable_state_sha256") != state_hash or
                    attempt.get("request_id") != item["request_id"] or
                    attempt.get("plan_index") != item["plan_index"]):
                raise ValueError("existing attempt receipt differs from its request or frozen state")
            expected_policy = "shared_initial" if round_index == 0 else arm
            if attempt.get("policy") != expected_policy:
                raise ValueError("existing attempt receipt belongs to another retry policy")
            if round_index and attempts[round_index - 1]["success"]:
                raise ValueError("existing arm continued after a successful candidate")
        if arms[arm]["success"] is not (attempts[-1]["success"] if attempts[-1]["success"] else
                                         False if len(attempts) == MAX_ATTEMPTS else None):
            raise ValueError("existing arm terminal status does not match its persisted attempts")
    if arms["independent"]["attempts"][0] != arms["feedback"]["attempts"][0]:
        raise ValueError("the shared initial candidate differs between policies")
    if group.get("status") not in ("running", "complete"):
        raise ValueError("existing group receipt has an unknown status")
    if group["status"] == "complete" and any(arms[arm]["success"] is None for arm in ARMS):
        raise ValueError("completed group has an unfinished policy")


def _write_group(path, group):
    RUNNER._write_json(path, group)


def _run_group(item, args, out, shard_dir, model, tok, train, score,
               identity, checkpoint_hash, state_hash, code_hashes, identity_hash):
    path = _group_path(shard_dir, item)
    cif_root = out / shard_dir.name / "cifs" / item["request_id"] / f"group-{item['attempt_id']:03d}"
    prompt_ids = tok.encode(item["prompt"], add_special_tokens=False)
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"{path} exists; pass --resume to continue")
        group = json.loads(path.read_text(encoding="utf-8"))
        _validate_group(group, item, identity_hash, checkpoint_hash, state_hash, code_hashes)
        if group["status"] == "complete":
            print(json.dumps({"group": item["request_id"], "index": item["attempt_id"],
                              "status": "skipped_complete"}), flush=True)
            return
    else:
        initial = _candidate_attempt(
            item, "shared_initial", 0, item["seed"], prompt_ids, None, False,
            model, tok, train, score, identity, checkpoint_hash, state_hash, code_hashes, cif_root)
        initial_success = initial["success"]
        arms = {arm: {"attempts": [initial],
                      "success": True if initial_success else None,
                      "attempts_used": 1} for arm in ARMS}
        group = {"status": "running", "end_status": None,
                 "request_id": item["request_id"], "group_index": item["attempt_id"],
                 "plan_index": item["plan_index"], "original_seed": item["seed"],
                 "plan_item": item, "seed_manifest": _seed_manifest(item),
                 "run_identity_sha256": identity_hash,
                 "checkpoint_fingerprint_sha256": checkpoint_hash,
                 "trainable_state_sha256": state_hash,
                 "evaluation_source_sha256": code_hashes, "arms": arms}
        _write_group(path, group)

    for arm in ARMS:
        record = group["arms"][arm]
        attempts = record["attempts"]
        if record["success"] is True or len(attempts) == MAX_ATTEMPTS:
            continue

        def generate(round_index, seed, previous):
            feedback, synthetic = None, False
            if arm == "feedback":
                feedback = _checker_feedback(previous, item["space_group_number"])
                feedback_ids = tok.encode("\nUser: " + feedback + "\nAssistant:",
                                          add_special_tokens=False)
                context, synthetic = _feedback_context(
                    prompt_ids, previous, feedback_ids, model.start_id, model.end_id)
            else:
                context = prompt_ids
            return _candidate_attempt(
                item, arm, round_index, seed, context, feedback, synthetic,
                model, tok, train, score, identity, checkpoint_hash, state_hash,
                code_hashes, cif_root)

        def persist(_attempt):
            record["attempts_used"] = len(attempts)
            record["success"] = (True if attempts[-1]["success"] else
                                 False if len(attempts) == MAX_ATTEMPTS else None)
            _write_group(path, group)

        _continue_arm(attempts, item["seed"], generate,
                      lambda receipt: receipt["success"], persist)
        record["attempts_used"] = len(attempts)
        record["success"] = True if attempts[-1]["success"] else False
        _write_group(path, group)

    independent, feedback = (group["arms"][arm]["success"] for arm in ARMS)
    group["status"] = "complete"
    group["end_status"] = ("both_success" if independent and feedback else
                            "independent_only" if independent else
                            "feedback_only" if feedback else "neither_success")
    _write_group(path, group)
    print(json.dumps({"group": item["request_id"], "index": item["attempt_id"],
                      "status": group["end_status"],
                      "independent_attempts": group["arms"]["independent"]["attempts_used"],
                      "feedback_attempts": group["arms"]["feedback"]["attempts_used"]}), flush=True)


def _sampler_parity(model, tok, item, identity, state_hash, checkpoint_hash, shard_dir):
    import torch

    prefix = tok.encode(item["prompt"], add_special_tokens=False) + [model.start_id]
    current = [model.mask_id] * E.OBJECTIVE.BLOCK_SIZE
    device = model.time.weight.device
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        actual = _denoise_logits(model, prefix, current)
        expected = E.OBJECTIVE.denoise_logits(model, prefix, current, len(prefix))
    difference = (actual - expected).abs()
    atol, rtol = 2e-3, 2e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    mismatches = int((actual.argmax(-1) != expected.argmax(-1)).sum().item())
    if mismatches:
        raise AssertionError("local single-segment denoising argmax differs from frozen objective")
    result = {"status": "passed", "max_absolute_difference": float(difference.max().item()),
              "mean_absolute_difference": float(difference.mean().item()),
              "argmax_mismatches": mismatches, "atol": atol, "rtol": rtol,
              "checkpoint_fingerprint_sha256": checkpoint_hash,
              "trainable_state_sha256": state_hash,
              "source_checkpoint_identity_sha256": identity["fingerprint_sha256"]}
    path = shard_dir / "sampler_parity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key in ("checkpoint_fingerprint_sha256", "trainable_state_sha256",
                    "source_checkpoint_identity_sha256", "atol", "rtol"):
            if existing.get(key) != result[key]:
                raise ValueError("existing sampler parity receipt has a different run identity")
        if existing.get("status") != "passed":
            raise ValueError("existing sampler parity receipt did not pass")
    else:
        RUNNER._write_json(path, result)
    return result


def run(args):
    if args.groups_per_sg not in (1, 10):
        raise ValueError("only the one-group smoke or frozen ten-group pilot is authorized")
    if args.num_shards not in (1, 2) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("the pilot supports one or two valid shards")
    if args.gpu not in (0, 1) or (args.num_shards == 2 and args.gpu != args.shard_index):
        raise ValueError("use GPU 1 for one-shard smoke or match shard 0/1 to GPU 0/1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    plan = RUNNER.attempt_plan(limit=args.groups_per_sg)
    if len(RUNNER.REQUESTS) != 7:
        raise ValueError("the frozen pilot requires its seven space-group requests")
    shard_plan = [item for item in plan if item["plan_index"] % args.num_shards == args.shard_index]
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")

    trainer = _load_module("structure_feedback_frozen_trainer", TRAINER_PATH)
    train, model, tok, identity, source_hash, software = trainer.load_model()
    model.eval()
    checkpoint_path = E.DEFAULT_CHECKPOINTS["flare"].expanduser().resolve(strict=True)
    checkpoint = trainer.load_checkpoint(model, checkpoint_path, train, "flare")
    state_hash = train.trainable_sha256(model)
    if checkpoint.get("trainable_state_sha256") != state_hash:
        raise ValueError("restored frozen FLARE state differs from its checkpoint receipt")
    checkpoint_hash = E._sha256_file(checkpoint_path)
    score = RUNNER._load_score_module()
    code_hashes = {**E._evaluation_sources(), "feedback_runner.py": E._sha256_file(Path(__file__))}
    max_positions = _max_positions(model)
    if MAX_CONTEXT > max_positions:
        raise ValueError(f"pilot context budget {MAX_CONTEXT} exceeds model capacity {max_positions}")

    identity_fields = {"checkpoint_fingerprint_sha256": checkpoint_hash,
                       "trainable_state_sha256": state_hash,
                       "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
                       "evaluation_source_sha256": code_hashes,
                       "group_plan_sha256": RUNNER.json_sha256(plan)}
    identity_hash = RUNNER.json_sha256(identity_fields)
    manifest = {
        "schema": 1, "experiment": HERE.name, "arm_checkpoint": "flare",
        "model": identity["model"], "revision": identity["revision"],
        **identity_fields, "run_identity_sha256": identity_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "source_trainable_state_sha256": source_hash, "software": software,
        "group_plan": plan, "groups_per_sg": args.groups_per_sg,
        "group_plan_sha256": RUNNER.json_sha256(plan),
        "seed_manifest": [_seed_manifest(item) for item in plan],
        "num_shards": args.num_shards, "context_budget": MAX_CONTEXT,
        "sampling": {"lead_temperature": E.TEMPERATURE, "lead_top_p": E.TOP_P,
                     "crystal_temperature": E.TEMPERATURE, "crystal_top_p": E.TOP_P,
                     "max_lead_tokens": MAX_LEAD_TOKENS,
                     "crystal_capacity_including_END": CRYSTAL_CAPACITY,
                     "max_attempts_per_policy": MAX_ATTEMPTS},
        "ehull_target": "<=0.08 eV/atom (input condition only; not evaluated)",
        "property_evaluation_status": "not_evaluated",
        "evaluation_scope": "paired independent versus checker-feedback retries; strict structure and requested SG only",
    }
    RUNNER._ensure_manifest(out / "run_manifest.json", manifest, args.resume, shared=True)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    shard_manifest = {"shard_index": args.shard_index, "num_shards": args.num_shards,
                      "gpu_visible": args.gpu,
                      "group_plan_indices": [item["plan_index"] for item in shard_plan],
                      "run_identity_sha256": identity_hash}
    RUNNER._ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume, shared=True)
    parity = _sampler_parity(model, tok, shard_plan[0], identity, state_hash, checkpoint_hash, shard_dir)
    print(json.dumps({"sampler_parity": parity, "shard": args.shard_index}), flush=True)
    for item in shard_plan:
        _run_group(item, args, out, shard_dir, model, tok, train, score, identity,
                   checkpoint_hash, state_hash, code_hashes, identity_hash)
    if train.trainable_sha256(model) != state_hash:
        raise RuntimeError("frozen FLARE checkpoint weights changed during inference")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups-per-sg", type=int, choices=(1, 10), default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
