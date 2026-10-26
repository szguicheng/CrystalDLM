"""Run the fixed 700-request structural-condition evaluation for this checkpoint."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
REFERENCE_RUNNER_PATH = ROOT / "experiments/structure_request_eval_20261017/run.py"
REFERENCE_SNAPSHOT = EXPERIMENT / "reference_source_snapshots/structure_request_eval_run.py"
CONFIG = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
DATA = ROOT / CONFIG["data"]["directory"]
MODEL = ROOT / "models/Qwen3-8B-b968826d9c46"
TRAIN_RUN = EXPERIMENT / "run_001"
DEFAULT_CHECKPOINT = TRAIN_RUN / "checkpoint.pt"
EHULL_BOUND = CONFIG["evaluation"]["ehull_upper_bound_eV_per_atom"]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module("structure_conditioned_ar8b_evaluation_helpers", REFERENCE_RUNNER_PATH)
RUNNER.EXPERIMENT = EXPERIMENT.resolve()
RUNNER.SHARED_RUN = TRAIN_RUN.resolve()
RUNNER.DEFAULT_DATA = DATA.resolve()
RUNNER.DEFAULT_MODEL = MODEL.resolve()
RUNNER.TRAINING_PROMPT_SCOPE = (
    "G tasks request actual space-group number and an ehull upper bound; "
    "Q targets retain the frozen factual reading targets")
RUNNER.EVALUATION_SCOPE = (
    "fixed space-group and ehull-upper-bound prompts; 700 pure-AR samples; "
    "strict structure audit followed by separate GRACE scoring")
RUNNER.REQUESTS = tuple({
    "request_id": f"sg-{sg:03d}",
    "space_group_number": sg,
    "prompt": (f"User: Generate a crystal with space group number {sg} and "
               f"energy above hull no greater than {EHULL_BOUND:g} eV/atom.\nAssistant:"),
} for sg in CONFIG["evaluation"]["space_groups"])


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_sources():
    return {
        "evaluate.py": _sha256_file(EXPERIMENT / "evaluate.py"),
        "reference_runner.py": _sha256_file(REFERENCE_RUNNER_PATH),
        "reference_runner_pre_edit.py": _sha256_file(REFERENCE_SNAPSHOT),
    }


def run(args):
    plan = RUNNER.attempt_plan(args.limit)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be within 0..num-shards-1")
    shard_plan = [item for item in plan if item["plan_index"] % args.num_shards == args.shard_index]
    if args.mode == "plan":
        print(json.dumps({"attempts": len(plan), "seeds_sha256": RUNNER.json_sha256(
            [item["seed"] for item in plan]), "requests": RUNNER.REQUESTS}, indent=2))
        return
    out = args.out.resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")
    if args.mode == "summarize":
        print(json.dumps(RUNNER.summarize_run(out), indent=2, ensure_ascii=False))
        return

    train, model, tok, identity, state_hash, software = RUNNER._load_model(args)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    parity = RUNNER._assert_parity(model, tok, RUNNER.REQUESTS[0]["prompt"])
    RUNNER._write_json(shard_dir / "cache_parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError(f"cached logits parity failed; see {shard_dir / 'cache_parity.json'}")
    run_manifest = {
        "schema": 1,
        "experiment": CONFIG["experiment"],
        "model": identity["model"],
        "revision": identity["revision"],
        "checkpoint_fingerprint_sha256": identity["fingerprint_sha256"],
        "trainable_state_sha256": state_hash,
        "training_data_hashes": identity["data_hashes"],
        "training_source_sha256": identity["source_sha256"],
        "evaluation_source_sha256": _evaluation_sources(),
        "requests": RUNNER.REQUESTS,
        "attempts_per_request": RUNNER.ATTEMPTS_PER_REQUEST,
        "attempt_plan": plan,
        "attempt_plan_sha256": RUNNER.json_sha256(plan),
        "limit": args.limit,
        "num_shards": args.num_shards,
        "sampling": {"lead": "stochastic top-p", "crystal": "stochastic top-p",
                     "temperature": RUNNER.TEMPERATURE, "top_p": RUNNER.TOP_P,
                     "max_lead_tokens": RUNNER.MAX_LEAD_TOKENS, "crystal_capacity": train.K},
        "training_prompt_scope": RUNNER.TRAINING_PROMPT_SCOPE,
        "evaluation_scope": RUNNER.EVALUATION_SCOPE,
        "ehull_target": f"<={EHULL_BOUND:g} eV/atom (prompt condition only; property evaluation not run)",
        "cached_logits_parity_contract": {
            "required": True,
            "contexts": ["prompt_before_start", "after_start", "inside_crystal_token_0",
                         "at_crystal_end", "after_crystal_end"],
            "strict_elementwise_tolerance_is_diagnostic_only": {
                "rtol": RUNNER.PARITY_STRICT_RTOL, "atol": RUNNER.PARITY_STRICT_ATOL},
            "bf16_distribution_acceptance": {
                "top1_same": True, "kl_reference_to_cached_max": RUNNER.PARITY_MAX_KL,
                "max_abs_max": RUNNER.PARITY_MAX_ABS,
                "mean_abs_max": RUNNER.PARITY_MAX_MEAN_ABS},
            "interpretation": "BF16 elementwise roundoff is reported; distribution criteria gate sampling.",
            "fp32_tiny_model_route_test_required": True},
    }
    RUNNER._ensure_manifest(out / "run_manifest.json", run_manifest, args.resume, shared=True)
    shard_manifest = {"shard_index": args.shard_index, "num_shards": args.num_shards,
                      "attempt_plan_indices": [item["plan_index"] for item in shard_plan],
                      "cached_logits_parity_passed": parity["passed"],
                      "run_manifest_sha256": RUNNER.json_sha256(run_manifest)}
    RUNNER._ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume)
    score = RUNNER._load_score_module()
    receipts = []
    summary_path = shard_dir / "summary.json"
    for item in shard_plan:
        receipt_path = RUNNER._attempt_path(shard_dir, item)
        if receipt_path.exists():
            if not args.resume:
                raise FileExistsError(f"{receipt_path} exists; pass --resume")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            RUNNER._validate_receipt(receipt, item, identity["fingerprint_sha256"], state_hash)
        else:
            receipt = RUNNER._sample_attempt(item, model, tok, score, train,
                                             identity["fingerprint_sha256"], state_hash,
                                             software, shard_dir)
            RUNNER._write_json(receipt_path, receipt)
        receipts.append(receipt)
        summary = RUNNER.summarize(shard_plan, receipts)
        summary.update({"run_manifest_sha256": RUNNER.json_sha256(run_manifest),
                        "shard": shard_manifest,
                        "cached_logits_parity": parity,
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
