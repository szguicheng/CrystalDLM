"""Run the paired repair-model generation and structural-feedback loop."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LOOP_PATH = ROOT / "experiments/structure_feedback_loop_20261020/run.py"
CONTEXT_PATH = ROOT / "experiments/structure_feedback_loop_20261020/switch_fix/context.py"
REPAIR_TRAINER_PATH = ROOT / "experiments/crystal_repair_training_20261021/run.py"
REPAIR_MODEL_PATH = ROOT / "experiments/crystal_repair_training_20261021/model.py"
REPAIR_CHECKPOINT = ROOT / "experiments/crystal_repair_training_20261021/run_001/checkpoint.pt"
EXPECTED_CHECKPOINT_SHA256 = "ca31b64b3920a4b9badced8c678659d880e1a3e7ecb3ffffe989057d4d21105b"
EXPECTED_STATE_SHA256 = "4775e389f53ef972fccbd4c8c86ad80a3c0cc7e93306ad33eb734d74a419160a"
EXPECTED_UPDATES = 1212
SG_NUMBERS = (1, 15, 38, 119, 143, 194, 216)
MAX_LEAD_TOKENS, CRYSTAL_CAPACITY, MAX_CONTEXT, MAX_ATTEMPTS = 256, 544, 4096, 3
TEMPERATURE, TOP_P, DIFFUSION_BLOCK_SIZE = 0.8, 0.95, 4
ARMS = ("independent", "feedback")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOOP = _load_module("repair_feedback_previous_loop", LOOP_PATH)
FEEDBACK_CONTEXT = _load_module("repair_feedback_fresh_context", CONTEXT_PATH)
E = LOOP.E
RUNNER = LOOP.RUNNER


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_checkpoint(trainer, train, model, source_state_hash):
    checkpoint_path = REPAIR_CHECKPOINT.resolve(strict=True)
    checkpoint_hash = _sha256_file(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("repair checkpoint byte hash differs from the frozen run")
    saved = trainer.load_checkpoint(
        model, checkpoint_path, train, expected_arm="repair",
        parent_checkpoint_sha256=trainer._sha256_file(trainer.PARENT_CHECKPOINT),
        parent_trainable_sha256=source_state_hash)
    state_hash = train.trainable_sha256(model)
    if saved.get("trainable_state_sha256") != state_hash:
        raise ValueError("restored repair state differs from its checkpoint receipt")
    if state_hash != EXPECTED_STATE_SHA256:
        raise ValueError("restored repair state differs from the frozen state hash")
    metadata = saved["metadata"]
    if (saved.get("arm") != "repair" or saved.get("completed_updates") != EXPECTED_UPDATES
            or saved.get("total_updates") != EXPECTED_UPDATES
            or metadata.get("config", {}).get("total_updates") != EXPECTED_UPDATES):
        raise ValueError("repair checkpoint is not the completed 1212-update run")
    return checkpoint_path, checkpoint_hash, state_hash, saved


def _code_hashes():
    return {
        **E._evaluation_sources(),
        "feedback_runner.py": _sha256_file(LOOP_PATH),
        "fresh_feedback_context.py": _sha256_file(CONTEXT_PATH),
        "repair_trainer_run.py": _sha256_file(REPAIR_TRAINER_PATH),
        "repair_trainer_model.py": _sha256_file(REPAIR_MODEL_PATH),
        "repair_feedback_full_run.py": _sha256_file(Path(__file__)),
    }


def _seed_manifest(item):
    return LOOP._seed_manifest(item)


def _group_path(shard_dir, item):
    return LOOP._group_path(shard_dir, item)


def _validate_group(group, item, identity_hash, checkpoint_hash, state_hash, code_hashes):
    return LOOP._validate_group(group, item, identity_hash, checkpoint_hash, state_hash, code_hashes)


def _write_group(path, group):
    return LOOP._write_group(path, group)


def _build_feedback_context(item, previous, tok):
    return FEEDBACK_CONTEXT.build_context(item, previous, tok, LOOP)


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
        initial = LOOP._candidate_attempt(
            item, "shared_initial", 0, item["seed"], prompt_ids, None, False,
            model, tok, train, score, identity, checkpoint_hash, state_hash, code_hashes, cif_root)
        arms = {arm: {"attempts": [initial],
                      "success": True if initial["success"] else None,
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
            feedback, synthetic_end = None, False
            if arm == "feedback":
                context, feedback = _build_feedback_context(item, previous, tok)
            else:
                context = prompt_ids
            return LOOP._candidate_attempt(
                item, arm, round_index, seed, context, feedback, synthetic_end,
                model, tok, train, score, identity, checkpoint_hash, state_hash,
                code_hashes, cif_root)

        def persist(_attempt):
            record["attempts_used"] = len(attempts)
            record["success"] = (True if attempts[-1]["success"] else
                                 False if len(attempts) == MAX_ATTEMPTS else None)
            _write_group(path, group)

        LOOP._continue_arm(attempts, item["seed"], generate,
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


def run(args):
    if args.groups_per_sg not in (1, 100):
        raise ValueError("only the seven-group smoke or frozen 700-group evaluation is supported")
    if args.num_shards != 3 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("the run uses three valid generation shards")
    if args.gpu not in (0, 1, 2, 3):
        raise ValueError("GPU must be a physical device index from 0 to 3")
    if ((LOOP.MAX_LEAD_TOKENS, LOOP.CRYSTAL_CAPACITY, LOOP.MAX_CONTEXT, LOOP.MAX_ATTEMPTS)
            != (MAX_LEAD_TOKENS, CRYSTAL_CAPACITY, MAX_CONTEXT, MAX_ATTEMPTS)
            or (E.TEMPERATURE, E.TOP_P, E.OBJECTIVE.BLOCK_SIZE)
            != (TEMPERATURE, TOP_P, DIFFUSION_BLOCK_SIZE)):
        raise ValueError("the reused sampler no longer matches the frozen attempt budget")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    plan = RUNNER.attempt_plan(limit=args.groups_per_sg)
    if (tuple(item["space_group_number"] for item in RUNNER.REQUESTS) != SG_NUMBERS
            or len(plan) != len(SG_NUMBERS) * args.groups_per_sg
            or any("0.08 eV/atom" not in item["prompt"] for item in plan)):
        raise ValueError("the frozen seven-space-group and 80-meV request plan differs")
    shard_plan = [item for item in plan if item["plan_index"] % args.num_shards == args.shard_index]
    out = args.out.expanduser().resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError(f"output must stay under project root: {out}")

    trainer = _load_module("repair_feedback_model_trainer", REPAIR_TRAINER_PATH)
    train, model, tok, identity, source_state_hash, software = trainer.load_model()
    checkpoint_path, checkpoint_hash, state_hash, saved = _restore_checkpoint(
        trainer, train, model, source_state_hash)
    model.eval()
    code_hashes = _code_hashes()
    score = RUNNER._load_score_module()
    if MAX_CONTEXT > int(model.base.config.max_position_embeddings):
        raise ValueError("4096-token experiment context exceeds model positional capacity")

    group_plan_hash = RUNNER.json_sha256(plan)
    identity_fields = {
        "checkpoint_fingerprint_sha256": checkpoint_hash,
        "trainable_state_sha256": state_hash,
        "source_checkpoint_identity_sha256": identity["fingerprint_sha256"],
        "checkpoint_metadata_sha256": saved["metadata_sha256"],
        "evaluation_source_sha256": code_hashes,
        "group_plan_sha256": group_plan_hash,
    }
    identity_hash = RUNNER.json_sha256(identity_fields)
    manifest = {
        "schema": 1, "experiment": HERE.name, "arm_checkpoint": "repair",
        "model": identity["model"], "revision": identity["revision"],
        **identity_fields, "run_identity_sha256": identity_hash,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_metadata": saved["metadata"],
        "checkpoint_completed_updates": saved["completed_updates"],
        "source_trainable_state_sha256": source_state_hash,
        "software": software,
        "group_plan": plan, "groups_per_sg": args.groups_per_sg,
        "group_plan_sha256": group_plan_hash,
        "seed_manifest": [_seed_manifest(item) for item in plan],
        "num_shards": args.num_shards, "context_budget": MAX_CONTEXT,
        "sampling": {"lead_temperature": E.TEMPERATURE, "lead_top_p": E.TOP_P,
                     "crystal_temperature": E.TEMPERATURE, "crystal_top_p": E.TOP_P,
                     "max_lead_tokens": MAX_LEAD_TOKENS,
                     "crystal_capacity_including_END": CRYSTAL_CAPACITY,
                     "diffusion_block_size": DIFFUSION_BLOCK_SIZE,
                     "max_attempts_per_policy": MAX_ATTEMPTS,
                     "max_physical_attempts": len(plan) * (1 + 2 * (MAX_ATTEMPTS - 1))},
        "feedback_context": "fresh User request; latest failed crystal or response decoded as plain input data; checker feedback",
        "ehull_target": "<=0.08 eV/atom (input condition only; not evaluated in the loop)",
        "property_evaluation_status": "not_evaluated",
        "evaluation_scope": "paired independent versus checker-feedback retries; strict structure and requested SG only; no GRACE feedback",
    }
    RUNNER._ensure_manifest(out / "run_manifest.json", manifest, args.resume, shared=True)
    shard_dir = out / f"shard-{args.shard_index:02d}-of-{args.num_shards:02d}"
    shard_manifest = {"shard_index": args.shard_index, "num_shards": args.num_shards,
                      "gpu_visible": args.gpu,
                      "group_plan_indices": [item["plan_index"] for item in shard_plan],
                      "run_identity_sha256": identity_hash}
    RUNNER._ensure_manifest(shard_dir / "manifest.json", shard_manifest, args.resume, shared=True)
    parity = LOOP._sampler_parity(model, tok, shard_plan[0], identity, state_hash,
                                  checkpoint_hash, shard_dir)
    print(json.dumps({"sampler_parity": parity, "shard": args.shard_index}), flush=True)
    for item in shard_plan:
        _run_group(item, args, out, shard_dir, model, tok, train, score, identity,
                   checkpoint_hash, state_hash, code_hashes, identity_hash)
    if train.trainable_sha256(model) != state_hash:
        raise RuntimeError("repair checkpoint weights changed during inference")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups-per-sg", type=int, choices=(1, 100), default=100)
    parser.add_argument("--shard-index", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--num-shards", type=int, choices=(3,), default=3)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
