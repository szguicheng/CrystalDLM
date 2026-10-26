"""Sample fixed space-group requests from the existing Qwen3-8B pure-AR checkpoint."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015"
SG_NUMBERS = (1, 15, 38, 119, 143, 194, 216)
REQUESTS = tuple({
    "request_id": f"sg-{sg:03d}",
    "space_group_number": sg,
    "prompt": (f"User: Generate a crystal with space group number {sg} and "
               "energy above hull no greater than 0.08 eV/atom.\nAssistant:"),
} for sg in SG_NUMBERS)
ATTEMPTS_PER_REQUEST = 100
SEED = 42
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_LEAD_TOKENS = 256
PARITY_MAX_KL = 0.01
PARITY_MAX_ABS = 0.5
PARITY_MAX_MEAN_ABS = 0.1
PARITY_STRICT_RTOL = 2e-3
PARITY_STRICT_ATOL = 2e-3
SHARED_RUN = EXPERIMENT / "run_001"
DEFAULT_DATA = ROOT / "research/crystal_training_preflight_20261013/data_001"
DEFAULT_MODEL = ROOT / "models/Qwen3-8B-b968826d9c46"


class GenerationFailure(Exception):
    def __init__(self, message, token_ids=(), text=""):
        super().__init__(message)
        self.token_ids = list(token_ids)
        self.text = text


def json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def attention_time_values(token_ids, start_id, end_id):
    """Match Hybrid.logits: START gets time -1; following crystal tokens get time 0."""
    active = False
    result = []
    for token_id in token_ids:
        if token_id == start_id:
            active = True
            result.append(-1)
        elif active:
            result.append(0)
            if token_id == end_id:
                active = False
        else:
            result.append(-1)
    return result


def attempt_plan(limit=None):
    seeds = random.Random(SEED).sample(range(1, 2**31),
                                        len(REQUESTS) * ATTEMPTS_PER_REQUEST)
    attempts_per_request = ATTEMPTS_PER_REQUEST if limit is None else limit
    if not 1 <= attempts_per_request <= ATTEMPTS_PER_REQUEST:
        raise ValueError(f"limit must be in 1..{ATTEMPTS_PER_REQUEST} attempts per request")
    plan = []
    for request_index, request in enumerate(REQUESTS):
        for attempt_index in range(attempts_per_request):
            flat_index = request_index * ATTEMPTS_PER_REQUEST + attempt_index
            plan.append({**request, "attempt_id": attempt_index,
                         "flat_index": flat_index, "plan_index": len(plan),
                         "seed": seeds[flat_index]})
    return plan


def validate_manifest(actual, expected):
    if json_sha256(actual) != json_sha256(expected):
        raise ValueError("existing run manifest does not match checkpoint, prompts, or sampling protocol")


def requested_sg_matches(actual_sg, requested_sg):
    return actual_sg is not None and int(actual_sg) == int(requested_sg)


def summarize(plan, receipts):
    by_slot = {(item["request_id"], item["attempt_id"]): item for item in receipts}
    completed = [by_slot[(item["request_id"], item["attempt_id"])]
                 for item in plan if (item["request_id"], item["attempt_id"]) in by_slot]
    valid = sum(item.get("structural_valid") is True for item in completed)
    sg_match = sum(item.get("requested_sg_match") is True for item in completed)
    return {
        "status": "complete" if len(completed) == len(plan) else "incomplete",
        "scheduled_attempts": len(plan),
        "completed_attempts": len(completed),
        "pending_attempts": len(plan) - len(completed),
        "strict_valid_structures": valid,
        "requested_space_group_matches": sg_match,
        "valid_and_requested_space_group_matches": sum(
            item.get("structural_valid") is True and item.get("requested_sg_match") is True
            for item in completed),
        "lead_failures": sum(item.get("lead_status") != "ok" for item in completed),
        "crystal_failures": sum(item.get("crystal_status") != "ok" for item in completed),
        "invalid_structure_attempts": sum(item.get("structural_valid") is not True
                                           for item in completed),
        "ehull_evaluation_status": "not_run",
        "raw_attempt_denominator": len(plan),
    }


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_cif(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_training_module():
    sys.path.insert(0, str(EXPERIMENT))
    return _load_module("crystaldlm_qwen3_train_reference", EXPERIMENT / "train.py")


def _load_score_module():
    score_path = ROOT / "experiments/interleaved_ar_diffusion_20261002/score.py"
    return _load_module("crystaldlm_plaid_score_reference", score_path)


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
    recorded_identity = json.loads((SHARED_RUN / "identity.json").read_text())
    validate_manifest(identity, recorded_identity)

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = json.loads((SHARED_RUN / "summary.json").read_text())
    if saved["schema"] != train.CHECKPOINT_SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if saved["fingerprint_sha256"] != identity["fingerprint_sha256"]:
        raise ValueError("checkpoint identity does not match the verified training receipt")
    validate_manifest(saved["identity"], identity)
    if saved["completed_updates"] != train.UPDATES or saved["next_material_index"] != train.TRAIN_MATERIALS:
        raise ValueError("checkpoint is not the completed one-epoch pure-AR run")
    if summary.get("status") != "one_epoch_training_complete" or not summary.get("learning_trial"):
        raise ValueError("training receipt does not establish a completed learning trial")
    if train.state_sha256(saved["added_token_base_rows"]) != saved["added_token_base_rows_sha256"]:
        raise ValueError("checkpoint control-token rows failed their hash check")
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if set(params) != set(saved["trainable_state"]):
        raise ValueError("checkpoint trainable parameter set differs from initialized model")
    train.restore_control_token_rows(model, saved["added_token_base_rows"])
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(saved["trainable_state"][name].to(device=param.device, dtype=param.dtype))
    trainable_hash = train.trainable_sha256(model)
    if trainable_hash != saved["trainable_state_sha256"]:
        raise ValueError("restored trainable state failed its hash check")
    if trainable_hash != summary["final_trainable_state_sha256"]:
        raise ValueError("checkpoint trainable state differs from the completed training receipt")
    del saved
    return train, model, tok, identity, trainable_hash, train.runtime_versions()


def _additive_mask(allowed, dtype, torch):
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(
        ~allowed, torch.finfo(dtype).min)


class CachedCausalSampler:
    """Qwen3 cached logits with Hybrid.logits' causal mask and learned time embeddings."""
    def __init__(self, model, tok, temperature=TEMPERATURE, top_p=TOP_P):
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and top_p must be in (0, 1]")
        self.model = model
        self.tok = tok
        import torch
        self.torch = torch
        self.device = model.time.weight.device
        self.temperature = temperature
        self.top_p = top_p
        self.past_key_values = None
        self.length = 0
        self.active_crystal = False
        self.logits = None

    def _embed(self, token_ids, time_values):
        torch = self.torch
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).view(1, -1)
        times = torch.as_tensor(time_values, dtype=torch.long, device=self.device).view(1, -1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embedded = self.model.base.get_input_embeddings()(ids)
            return embedded + self.model.time(times.clamp_min(0)) * (times >= 0).unsqueeze(-1)

    def prefill(self, token_ids):
        torch = self.torch
        if not token_ids:
            raise ValueError("cannot prefill an empty prompt")
        times = attention_time_values(token_ids, self.model.start_id, self.model.end_id)
        embedded = self._embed(token_ids, times)
        length = len(token_ids)
        allowed = torch.ones((1, 1, length, length), dtype=torch.bool, device=self.device).tril()
        attention_mask = {"full_attention": _additive_mask(allowed, embedded.dtype, torch)}
        cache_position = torch.arange(length, dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.base.model(inputs_embeds=embedded, attention_mask=attention_mask,
                                           past_key_values=None, use_cache=True,
                                           cache_position=cache_position)
            self.logits = self.model.base.lm_head(output.last_hidden_state[:, -1]).float()[0]
        self.past_key_values = output.past_key_values
        self.length = length
        for token_id in token_ids:
            if token_id == self.model.start_id:
                self.active_crystal = True
            elif self.active_crystal and token_id == self.model.end_id:
                self.active_crystal = False
        return self.logits

    def append(self, token_id):
        torch = self.torch
        if token_id == self.model.start_id:
            time_value = -1
        else:
            time_value = 0 if self.active_crystal else -1
        embedded = self._embed([token_id], [time_value])
        allowed = torch.ones((1, 1, 1, self.length + 1), dtype=torch.bool, device=self.device)
        attention_mask = {"full_attention": _additive_mask(allowed, embedded.dtype, torch)}
        cache_position = torch.tensor([self.length], dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.base.model(inputs_embeds=embedded, attention_mask=attention_mask,
                                           past_key_values=self.past_key_values, use_cache=True,
                                           cache_position=cache_position)
            self.logits = self.model.base.lm_head(output.last_hidden_state[:, -1]).float()[0]
        self.past_key_values = output.past_key_values
        self.length += 1
        if token_id == self.model.start_id:
            self.active_crystal = True
        elif self.active_crystal and token_id == self.model.end_id:
            self.active_crystal = False
        return self.logits

    def sample(self, generator, banned_ids):
        torch = self.torch
        with torch.inference_mode():
            scores = self.logits.clone() / self.temperature
            scores[list(banned_ids)] = -torch.inf
            if self.top_p < 1:
                sorted_scores, sorted_ids = torch.sort(scores, descending=True)
                sorted_probs = sorted_scores.softmax(-1)
                remove = sorted_probs.cumsum(-1) > self.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                scores[sorted_ids[remove]] = -torch.inf
            probs = scores.softmax(-1)
            return int(torch.multinomial(probs, 1, generator=generator))


def _logit_parity_metrics(reference, cached):
    import torch
    reference = reference.float().flatten()
    cached = cached.float().flatten()
    absolute = (cached - reference).abs()
    log_reference = torch.log_softmax(reference, dim=-1)
    log_cached = torch.log_softmax(cached, dim=-1)
    probabilities = log_reference.exp()
    kl = float((probabilities * (log_reference - log_cached)).sum())
    within_tolerance = torch.isclose(cached, reference, rtol=PARITY_STRICT_RTOL,
                                    atol=PARITY_STRICT_ATOL)
    top1_reference = int(reference.argmax())
    top1_cached = int(cached.argmax())
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms_abs": float(absolute.square().mean().sqrt()),
        "kl_reference_to_cached": kl,
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(
            reference.unsqueeze(0), cached.unsqueeze(0)).item()),
        "top1_reference": top1_reference,
        "top1_cached": top1_cached,
        "top1_same": top1_reference == top1_cached,
        "strict_close_fraction": float(within_tolerance.float().mean()),
        "strict_close_rtol": PARITY_STRICT_RTOL,
        "strict_close_atol": PARITY_STRICT_ATOL,
        "distribution_criteria_passed": (
            top1_reference == top1_cached and kl <= PARITY_MAX_KL
            and max_abs <= PARITY_MAX_ABS and mean_abs <= PARITY_MAX_MEAN_ABS),
    }


def _assert_parity(model, tok, prompt):
    import torch
    ids = tok.encode(prompt, add_special_tokens=False)
    sampler = CachedCausalSampler(model, tok)
    contexts = []

    def check(label):
        input_ids = torch.tensor(ids, device=model.time.weight.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference = model.logits(input_ids, "ar", positions=[len(ids) - 1])[0].float()
        contexts.append({"context": label,
                         **_logit_parity_metrics(reference, sampler.logits)})

    sampler.prefill(ids)
    check("prompt_before_start")
    for token_id, label in ((model.start_id, "after_start"), (0, "inside_crystal_token_0"),
                            (model.end_id, "at_crystal_end"), (1, "after_crystal_end")):
        sampler.append(token_id)
        ids.append(token_id)
        check(label)
    passed = all(item["distribution_criteria_passed"] for item in contexts)
    report = {
        "passed": passed,
        "interpretation": "BF16 cached and full forwards may differ elementwise from rounding; FP32 tiny-model route coverage is checked separately.",
        "acceptance": {"top1_same": True, "kl_reference_to_cached_max": PARITY_MAX_KL,
                       "max_abs_max": PARITY_MAX_ABS,
                       "mean_abs_max": PARITY_MAX_MEAN_ABS},
        "strict_elementwise_tolerance": {"rtol": PARITY_STRICT_RTOL,
                                         "atol": PARITY_STRICT_ATOL},
        "aggregate": {
            "max_abs": max(item["max_abs"] for item in contexts),
            "max_mean_abs": max(item["mean_abs"] for item in contexts),
            "max_kl_reference_to_cached": max(
                item["kl_reference_to_cached"] for item in contexts),
            "all_top1_same": all(item["top1_same"] for item in contexts),
            "all_distribution_criteria_passed": passed,
        },
        "contexts": contexts,
    }
    print(json.dumps({"cached_logits_parity_diagnostic": report}, allow_nan=False), flush=True)
    return report


def _boundary_content(raw, model, tok):
    end_positions = [index for index, value in enumerate(raw) if value == model.end_id]
    if len(end_positions) != 1:
        raise GenerationFailure("crystal block requires exactly one END", raw,
                                tok.decode(raw, skip_special_tokens=False))
    end = end_positions[0]
    if model.pad_id in raw[:end] or any(value != model.pad_id for value in raw[end + 1:]):
        raise GenerationFailure("invalid crystal PAD boundary", raw,
                                tok.decode(raw, skip_special_tokens=False))
    return tok.decode(raw[:end], skip_special_tokens=False)


def _sample_attempt(item, model, tok, score, train, checkpoint_id, state_hash, software, out):
    import torch
    started = time.monotonic()
    generator = torch.Generator(device=model.time.weight.device).manual_seed(item["seed"])
    sampler = CachedCausalSampler(model, tok)
    prompt_tokens = tok.encode(item["prompt"], add_special_tokens=False)
    sampler.prefill(prompt_tokens)
    lead_tokens = []
    lead_started = time.monotonic()
    banned_text = {model.mask_id, model.pad_id, model.end_id}
    lead_error = None
    for _ in range(MAX_LEAD_TOKENS):
        token_id = sampler.sample(generator, banned_text)
        lead_tokens.append(token_id)
        if token_id == model.start_id:
            sampler.append(token_id)
            break
        if token_id == tok.eos_token_id:
            lead_error = "AR response ended before the crystal start token"
            break
        if sampler.length >= train.BASELINE.MAXLEN:
            lead_error = "text context capacity exceeded"
            break
        sampler.append(token_id)
    else:
        lead_error = f"AR did not emit the crystal start token within {MAX_LEAD_TOKENS} tokens"
    lead_seconds = time.monotonic() - lead_started
    lead_text = tok.decode(lead_tokens, skip_special_tokens=False)
    receipt = {
        "status": "complete",
        "request_id": item["request_id"],
        "space_group_number_requested": item["space_group_number"],
        "attempt_id": item["attempt_id"],
        "flat_index": item["flat_index"],
        "plan_index": item["plan_index"],
        "seed": item["seed"],
        "prompt": item["prompt"],
        "sampling": {"lead": "stochastic top-p", "crystal": "stochastic top-p",
                     "temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_lead_tokens": MAX_LEAD_TOKENS, "crystal_capacity": train.K},
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
        "timing_seconds": {"lead_generation": lead_seconds},
    }
    if lead_error:
        receipt["crystal_error"] = "crystal start was not reached"
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    if sampler.length + train.K > train.BASELINE.MAXLEN:
        raise RuntimeError("prompt plus generated lead exceeds model context capacity")
    raw_tokens = []
    banned_crystal = set(tok.all_special_ids) - {model.end_id, model.pad_id}
    crystal_started = time.monotonic()
    for _ in range(train.K):
        token_id = sampler.sample(generator, banned_crystal)
        raw_tokens.append(token_id)
        if token_id == model.end_id:
            raw_tokens.extend([model.pad_id] * (train.K - len(raw_tokens)))
            break
        sampler.append(token_id)
    crystal_seconds = time.monotonic() - crystal_started
    raw_text = tok.decode(raw_tokens, skip_special_tokens=False)
    receipt["crystal_raw_token_ids"] = raw_tokens
    receipt["crystal_raw_text"] = raw_text
    receipt["timing_seconds"]["crystal_generation"] = crystal_seconds
    try:
        crystal_text = _boundary_content(raw_tokens, model, tok)
        receipt["crystal_text"] = crystal_text
        receipt["crystal_status"] = "ok"
    except GenerationFailure as error:
        receipt["crystal_status"] = "failed"
        receipt["crystal_error"] = str(error)
        receipt["crystal_raw_token_ids"] = error.token_ids
        receipt["crystal_raw_text"] = error.text
        receipt["audit"] = {"valid": False, "reasons": [str(error)]}
        receipt["timing_seconds"]["total"] = time.monotonic() - started
        return receipt

    audit_started = time.monotonic()
    audit = score.audit(crystal_text)
    receipt["audit"] = audit
    receipt["structural_valid"] = audit.get("valid") is True
    cif_text = None
    try:
        structure = score.parse_fn_wyckoff(crystal_text, to_cif=False)
        cif_text = structure.to(fmt="cif")
        actual_sg = int(score.SpacegroupAnalyzer(
            structure, symprec=.01, angle_tolerance=5).get_space_group_number())
        receipt["actual_space_group_number"] = actual_sg
    except Exception as error:
        receipt["decode_error"] = f"{type(error).__name__}: {error}"
    if cif_text is not None:
        cif_path = out / "attempts" / item["request_id"] / f"attempt-{item['attempt_id']:03d}.cif"
        receipt["cif"] = _write_cif(cif_path, cif_text)
    receipt["requested_sg_match"] = requested_sg_matches(
        receipt["actual_space_group_number"], item["space_group_number"])
    receipt["timing_seconds"]["structure_audit_and_decode"] = time.monotonic() - audit_started
    receipt["timing_seconds"]["total"] = time.monotonic() - started
    return receipt


def _attempt_path(shard_dir, item):
    return shard_dir / "attempts" / item["request_id"] / f"attempt-{item['attempt_id']:03d}.json"


def _validate_receipt(receipt, item, checkpoint_id, state_hash):
    if (receipt.get("request_id"), receipt.get("attempt_id"), receipt.get("seed")) != (
            item["request_id"], item["attempt_id"], item["seed"]):
        raise ValueError("existing attempt receipt does not match its fixed request and seed")
    if receipt.get("checkpoint_fingerprint_sha256") != checkpoint_id:
        raise ValueError("existing attempt receipt belongs to another checkpoint")
    if receipt.get("trainable_state_sha256") != state_hash:
        raise ValueError("existing attempt receipt has a different trainable-state hash")


def _ensure_manifest(path, expected, resume, shared=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        validate_manifest(json.loads(path.read_text(encoding="utf-8")), expected)
        if not resume and not shared:
            raise FileExistsError(f"{path} already exists; pass --resume to continue this run")
        return
    if resume:
        raise FileNotFoundError(f"cannot resume without {path}")
    _write_json(path, expected)


def summarize_run(out):
    out = Path(out).resolve(strict=True)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    plan = manifest["attempt_plan"]
    receipts = []
    for item in plan:
        candidates = list(out.glob(
            f"shard-*/attempts/{item['request_id']}/attempt-{item['attempt_id']:03d}.json"))
        if len(candidates) > 1:
            raise ValueError(f"duplicate attempt receipts for plan slot {item['plan_index']}")
        if candidates:
            receipt = json.loads(candidates[0].read_text(encoding="utf-8"))
            _validate_receipt(receipt, item, manifest["checkpoint_fingerprint_sha256"],
                              manifest["trainable_state_sha256"])
            receipts.append(receipt)
    summary = summarize(plan, receipts)
    summary.update({"run_manifest_sha256": json_sha256(manifest),
                    "requests": manifest["requests"],
                    "attempt_plan_sha256": manifest["attempt_plan_sha256"],
                    "sampling": manifest["sampling"],
                    "checkpoint_fingerprint_sha256": manifest["checkpoint_fingerprint_sha256"],
                    "trainable_state_sha256": manifest["trainable_state_sha256"],
                    "cached_logits_parity_by_shard": [
                        json.loads(path.read_text(encoding="utf-8"))
                        for path in sorted(out.glob("shard-*/cache_parity.json"))],
                    "ehull_evaluation_status": "not_run"})
    _write_json(out / "summary.json", summary)
    return summary


def run(args):
    plan = attempt_plan(args.limit)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be within 0..num-shards-1")
    shard_plan = [item for item in plan if item["plan_index"] % args.num_shards == args.shard_index]
    if args.mode == "plan":
        print(json.dumps({"attempts": len(plan), "seeds_sha256": json_sha256(
            [item["seed"] for item in plan]), "requests": REQUESTS}, indent=2))
        return
    out = args.out.resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")
    if args.mode == "summarize":
        print(json.dumps(summarize_run(out), indent=2, ensure_ascii=False))
        return

    train, model, tok, identity, state_hash, software = _load_model(args)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    parity = _assert_parity(model, tok, REQUESTS[0]["prompt"])
    _write_json(shard_dir / "cache_parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError(f"cached logits parity failed; see {shard_dir / 'cache_parity.json'}")
    run_manifest = {
        "schema": 1,
        "model": identity["model"],
        "revision": identity["revision"],
        "checkpoint_fingerprint_sha256": identity["fingerprint_sha256"],
        "trainable_state_sha256": state_hash,
        "requests": REQUESTS,
        "attempts_per_request": ATTEMPTS_PER_REQUEST,
        "attempt_plan": plan,
        "attempt_plan_sha256": json_sha256(plan),
        "limit": args.limit,
        "num_shards": args.num_shards,
        "sampling": {"lead": "stochastic top-p", "crystal": "stochastic top-p",
                     "temperature": TEMPERATURE, "top_p": TOP_P,
                     "max_lead_tokens": MAX_LEAD_TOKENS, "crystal_capacity": train.K},
        "training_prompt_scope": "ehull-only requests with reference-structure descriptions; no requested space-group condition",
        "evaluation_scope": "prompt-transfer check for space-group requests; no model comparison",
        "ehull_target": "<=0.08 eV/atom (prompt condition only; property evaluation not run)",
        "cached_logits_parity_contract": {
            "required": True,
            "contexts": ["prompt_before_start", "after_start", "inside_crystal_token_0",
                         "at_crystal_end", "after_crystal_end"],
            "strict_elementwise_tolerance_is_diagnostic_only": {
                "rtol": PARITY_STRICT_RTOL, "atol": PARITY_STRICT_ATOL},
            "bf16_distribution_acceptance": {
                "top1_same": True, "kl_reference_to_cached_max": PARITY_MAX_KL,
                "max_abs_max": PARITY_MAX_ABS,
                "mean_abs_max": PARITY_MAX_MEAN_ABS},
            "interpretation": "BF16 elementwise roundoff is reported; distribution criteria gate sampling.",
            "fp32_tiny_model_route_test_required": True},
    }
    _ensure_manifest(out / "run_manifest.json", run_manifest, args.resume, shared=True)
    shard_manifest = {"shard_index": args.shard_index, "num_shards": args.num_shards,
                      "attempt_plan_indices": [item["plan_index"] for item in shard_plan],
                      "cached_logits_parity_passed": parity["passed"],
                      "run_manifest_sha256": json_sha256(run_manifest)}
    _ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume)
    score = _load_score_module()
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
            _write_json(receipt_path, receipt)
        receipts.append(receipt)
        summary = summarize(shard_plan, receipts)
        summary.update({"run_manifest_sha256": json_sha256(run_manifest),
                        "shard": shard_manifest,
                        "cached_logits_parity": parity,
                        "temperature": TEMPERATURE, "top_p": TOP_P,
                        "ehull_evaluation_status": "not_run"})
        _write_json(summary_path, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "plan", "summarize"), default="sample")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=SHARED_RUN / "checkpoint.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int,
                        help="limit attempts per request; --limit 1 gives seven total preflight attempts")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
