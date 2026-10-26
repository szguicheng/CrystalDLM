"""Replay the frozen 70-group comparison with the corrected feedback context."""
import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = HERE.parents[2]
LEGACY_RUN = EXPERIMENT / "run.py"
SOURCE_OUT = EXPERIMENT / "run_001"
DIAGNOSIS_MANIFEST = EXPERIMENT / "switch_diagnosis/flare/manifest.json"
CONTEXT_PATH = HERE / "context.py"
DEFAULT_OUT = HERE / "run_001"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOOP = _load_module("structure_feedback_frozen_legacy_loop", LEGACY_RUN)


def _source_groups(plan, parent_manifest):
    expected = {(item["request_id"], item["attempt_id"]): item for item in plan}
    paths = sorted(SOURCE_OUT.glob("shard-*/groups/*/group-*.json"))
    groups, hashes = {}, {}
    for path in paths:
        group = json.loads(path.read_text(encoding="utf-8"))
        key = (group["request_id"], group["group_index"])
        if key not in expected or key in groups:
            raise ValueError(f"unexpected or duplicate source group: {key}")
        if (group.get("status") != "complete" or group.get("plan_item") != expected[key] or
                group.get("run_identity_sha256") != parent_manifest["run_identity_sha256"] or
                group.get("checkpoint_fingerprint_sha256") != parent_manifest["checkpoint_fingerprint_sha256"] or
                group.get("trainable_state_sha256") != parent_manifest["trainable_state_sha256"] or
                group.get("evaluation_source_sha256") != parent_manifest["evaluation_source_sha256"]):
            raise ValueError(f"source group does not match its completed parent run: {key}")
        left = group["arms"]["independent"]["attempts"]
        right = group["arms"]["feedback"]["attempts"]
        if not left or not right or left[0] != right[0]:
            raise ValueError(f"source group does not preserve its shared initial candidate: {key}")
        relpath = str(path.resolve().relative_to(ROOT))
        digest = LOOP.E._sha256_file(path)
        groups[key] = (group, relpath, digest)
        hashes[relpath] = digest
    if set(groups) != set(expected):
        raise ValueError(f"source run must contain all {len(expected)} completed groups")
    return groups, hashes


def _attempt_seed_ok(attempt, item, round_index):
    expected = item["seed"] if round_index == 0 else LOOP.retry_seed(item["seed"], round_index)
    return attempt.get("round") == round_index and attempt.get("seed") == expected


def _validate_group(group, item, source, identity, diagnostic):
    old, source_path, source_hash = source
    if (group.get("request_id"), group.get("group_index"), group.get("plan_index"),
            group.get("original_seed"), group.get("plan_item"),
            group.get("run_identity_sha256"), group.get("reused_from"),
            group.get("evaluation_split")) != (
            item["request_id"], item["attempt_id"], item["plan_index"], item["seed"],
            item, identity["run_identity_sha256"], {"path": source_path, "sha256": source_hash},
            "diagnostic" if _case_id(item) in diagnostic else "heldout"):
        raise ValueError("existing group receipt does not match its frozen source or run identity")
    if group["arms"]["independent"] != old["arms"]["independent"]:
        raise ValueError("independent arm differs from its pinned historical receipt")
    feedback = group["arms"]["feedback"]["attempts"]
    if not feedback or feedback[0] != old["arms"]["feedback"]["attempts"][0]:
        raise ValueError("feedback arm no longer begins with the shared historical candidate")
    record = group["arms"]["feedback"]
    if not 1 <= len(feedback) <= LOOP.MAX_ATTEMPTS or record["attempts_used"] != len(feedback):
        raise ValueError("feedback arm has an invalid attempt count")
    if record["success"] is not (feedback[-1]["success"] if feedback[-1]["success"] else
                                  False if len(feedback) == LOOP.MAX_ATTEMPTS else None):
        raise ValueError("feedback terminal status differs from its receipts")
    for round_index, attempt in enumerate(feedback):
        if not _attempt_seed_ok(attempt, item, round_index):
            raise ValueError("feedback receipt has a different paired seed")
        if round_index and (attempt.get("policy") != "feedback_fixed" or
                attempt.get("checkpoint_fingerprint_sha256") != identity["checkpoint_fingerprint_sha256"] or
                attempt.get("trainable_state_sha256") != identity["trainable_state_sha256"] or
                attempt.get("evaluation_source_sha256") != identity["evaluation_source_sha256"]):
            raise ValueError("feedback retry receipt belongs to another code or checkpoint identity")
        if round_index and feedback[round_index - 1]["success"]:
            raise ValueError("feedback retries continued after success")
    if group.get("status") not in ("running", "complete"):
        raise ValueError("existing group receipt has an unknown status")
    if group["status"] == "complete" and record["success"] is None:
        raise ValueError("completed group has an unfinished feedback arm")


def _case_id(item):
    return f"{item['request_id']}-{item['attempt_id']:03d}"


def _end_status(group):
    independent, feedback = (group["arms"][arm]["success"] for arm in LOOP.ARMS)
    return ("both_success" if independent and feedback else
            "independent_only" if independent else
            "feedback_only" if feedback else "neither_success")


def _group_path(shard_dir, item):
    return shard_dir / "groups" / item["request_id"] / f"group-{item['attempt_id']:03d}.json"


def _write_group(path, group):
    LOOP.RUNNER._write_json(path, group)


def _run_group(item, source, diagnostic, args, out, shard_dir, build_context,
               model, tok, train, score, identity, code_hashes):
    old, source_path, source_hash = source
    path = _group_path(shard_dir, item)
    if path.exists():
        if not args.resume:
            raise FileExistsError(f"{path} exists; pass --resume to continue")
        group = json.loads(path.read_text(encoding="utf-8"))
        _validate_group(group, item, source, identity, diagnostic)
        if group["status"] == "complete":
            print(json.dumps({"group": _case_id(item), "status": "skipped_complete"}), flush=True)
            return
    else:
        initial = deepcopy(old["arms"]["feedback"]["attempts"][0])
        group = {
            "status": "running", "end_status": None,
            "request_id": item["request_id"], "group_index": item["attempt_id"],
            "plan_index": item["plan_index"], "original_seed": item["seed"],
            "plan_item": item, "seed_manifest": old["seed_manifest"],
            "run_identity_sha256": identity["run_identity_sha256"],
            "checkpoint_fingerprint_sha256": identity["checkpoint_fingerprint_sha256"],
            "trainable_state_sha256": identity["trainable_state_sha256"],
            "evaluation_source_sha256": code_hashes,
            "parent_run_manifest_sha256": identity["parent_run_manifest_sha256"],
            "evaluation_split": "diagnostic" if _case_id(item) in diagnostic else "heldout",
            "reused_from": {"path": source_path, "sha256": source_hash},
            "arms": {
                "independent": deepcopy(old["arms"]["independent"]),
                "feedback": {"attempts": [initial],
                             "success": True if initial["success"] else None,
                             "attempts_used": 1},
            },
        }
        _write_group(path, group)

    feedback_record = group["arms"]["feedback"]
    attempts = feedback_record["attempts"]
    if feedback_record["success"] is not True and len(attempts) < LOOP.MAX_ATTEMPTS:
        cif_root = out / shard_dir.name / "cifs" / item["request_id"] / f"group-{item['attempt_id']:03d}"

        def generate(round_index, seed, previous):
            context_ids, feedback = build_context(item, previous, tok, LOOP)
            attempt = LOOP._candidate_attempt(
                item, "feedback_fixed", round_index, seed, context_ids, feedback, False,
                model, tok, train, score, identity["model_identity"],
                identity["checkpoint_fingerprint_sha256"],
                identity["trainable_state_sha256"], code_hashes, cif_root)
            attempt["reused_from_group_sha256"] = source_hash
            attempt["context_policy"] = "fresh request with previous candidate as plain input data"
            return attempt

        def persist(_attempt):
            feedback_record["attempts_used"] = len(attempts)
            feedback_record["success"] = (True if attempts[-1]["success"] else
                                           False if len(attempts) == LOOP.MAX_ATTEMPTS else None)
            _write_group(path, group)

        LOOP._continue_arm(attempts, item["seed"], generate,
                           lambda receipt: receipt["success"], persist)
        feedback_record["attempts_used"] = len(attempts)
        feedback_record["success"] = bool(attempts[-1]["success"])

    group["status"] = "complete"
    group["end_status"] = _end_status(group)
    _write_group(path, group)
    print(json.dumps({"group": _case_id(item), "status": group["end_status"],
                      "feedback_attempts": feedback_record["attempts_used"]}), flush=True)


def run(args):
    if args.num_shards not in (1, 2) or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("the switch-fix run supports one or two valid shards")
    if args.gpu not in (0, 1) or (args.num_shards == 2 and args.gpu != args.shard_index):
        raise ValueError("use a single explicit GPU; for two shards GPU must match shard index")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    parent_manifest_path = SOURCE_OUT / "run_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_manifest_hash = LOOP.E._sha256_file(parent_manifest_path)
    plan = LOOP.RUNNER.attempt_plan(limit=10)
    if plan != parent_manifest["group_plan"]:
        raise ValueError("source plan differs from the frozen 70-group plan")
    sources, source_group_hashes = _source_groups(plan, parent_manifest)
    diagnosis = json.loads(DIAGNOSIS_MANIFEST.read_text(encoding="utf-8"))
    diagnostic = set(diagnosis["cases"])
    if len(diagnostic) != 14 or not diagnostic <= {_case_id(item) for item in plan}:
        raise ValueError("diagnostic split must identify exactly the 14 replayed source groups")
    diagnostic_hash = LOOP.E._sha256_file(DIAGNOSIS_MANIFEST)

    context = _load_module("structure_feedback_fixed_context", CONTEXT_PATH)
    trainer = LOOP._load_module("structure_feedback_fixed_trainer", LOOP.TRAINER_PATH)
    train, model, tok, model_identity, source_hash, software = trainer.load_model()
    model.eval()
    checkpoint_path = LOOP.E.DEFAULT_CHECKPOINTS["flare"].expanduser().resolve(strict=True)
    checkpoint = trainer.load_checkpoint(model, checkpoint_path, train, "flare")
    state_hash = train.trainable_sha256(model)
    checkpoint_hash = LOOP.E._sha256_file(checkpoint_path)
    if checkpoint.get("trainable_state_sha256") != state_hash:
        raise ValueError("restored frozen FLARE state differs from its checkpoint receipt")
    if (diagnosis["checkpoint_sha256"], diagnosis["state_sha256"]) != (checkpoint_hash, state_hash):
        raise ValueError("switch diagnosis used a different frozen FLARE checkpoint or state")

    code_hashes = {**LOOP.E._evaluation_sources(),
                   "legacy_loop.py": LOOP.E._sha256_file(LEGACY_RUN),
                   "switch_fix/run.py": LOOP.E._sha256_file(Path(__file__)),
                   "switch_fix/context.py": LOOP.E._sha256_file(CONTEXT_PATH)}
    identity_fields = {
        "parent_run_manifest_sha256": parent_manifest_hash,
        "source_group_hashes": source_group_hashes,
        "diagnosis_manifest_sha256": diagnostic_hash,
        "diagnostic_case_ids": sorted(diagnostic),
        "group_plan_sha256": LOOP.RUNNER.json_sha256(plan),
        "checkpoint_fingerprint_sha256": checkpoint_hash,
        "trainable_state_sha256": state_hash,
        "source_checkpoint_identity_sha256": model_identity["fingerprint_sha256"],
        "evaluation_source_sha256": code_hashes,
        "num_shards": args.num_shards,
    }
    run_identity_hash = LOOP.RUNNER.json_sha256(identity_fields)
    identity = {**identity_fields, "run_identity_sha256": run_identity_hash,
                "model_identity": model_identity}
    manifest = {
        "schema": 1, "experiment": "structure_feedback_loop_switch_fix_20261020",
        "model": model_identity["model"], "revision": model_identity["revision"],
        **identity_fields, "run_identity_sha256": run_identity_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "source_trainable_state_sha256": source_hash, "software": software,
        "group_plan": plan, "source_group_hashes": source_group_hashes,
        "diagnostic_case_ids": sorted(diagnostic), "heldout_group_count": len(plan) - len(diagnostic),
        "context_policy": "original request plus latest failed crystal as plain input data and checker feedback",
        "property_evaluation_status": "not_evaluated",
    }
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")
    LOOP.RUNNER._ensure_manifest(out / "run_manifest.json", manifest, args.resume, shared=True)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    shard_manifest = {"shard_index": args.shard_index, "num_shards": args.num_shards,
                      "gpu_visible": args.gpu,
                      "group_plan_indices": [p["plan_index"] for p in plan
                                             if p["plan_index"] % args.num_shards == args.shard_index],
                      "run_identity_sha256": run_identity_hash}
    LOOP.RUNNER._ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume, shared=True)
    shard_plan = [p for p in plan if p["plan_index"] % args.num_shards == args.shard_index]
    LOOP._sampler_parity(model, tok, shard_plan[0], model_identity, state_hash,
                         checkpoint_hash, shard_dir)
    score = LOOP.RUNNER._load_score_module()
    for item in shard_plan:
        key = (item["request_id"], item["attempt_id"])
        _run_group(item, sources[key], diagnostic, args, out, shard_dir, context.build_context,
                   model, tok, train, score, identity, code_hashes)
    if train.trainable_sha256(model) != state_hash:
        raise RuntimeError("frozen FLARE checkpoint weights changed during inference")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, choices=(1, 2), default=1)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
