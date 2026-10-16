"""Evaluate confidence-one commits on the frozen joint AR-anchored B32 checkpoint."""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROJECT = HERE.parents[3]
JOINT_ROOT = PROJECT / "experiments/interleaved_relational_joint_20261006"
JOINT_ARM = JOINT_ROOT / "pilot_001/joint_ar_diffusion_b32"
JOINT_STATUS = JOINT_ARM / "status.json"
JOINT_RECEIPT = JOINT_ARM / "checkpoint_receipt.json"
JOINT_CHECKPOINT = JOINT_ARM / "checkpoint.pt"
EQUALITY_AUDIT = ROOT / "diagnostics/root_cause_20261006/joint_inference_source_equality.json"
DEFAULT_OUT = HERE / "joint_confidence_001"

JOINT_CHECKPOINT_SHA256 = "6a3052775a85e5a75c17e269b897b14ea4975ca77b5baf02736815b6da3a1ce5"
JOINT_FINGERPRINT = "ea511f138dde1598328f5d019a7b5b1701470302196c39aaba666a821f2ae009"
JOINT_MODEL_SHA256 = "51e547b34d9a30a07f601f7cb2027d6d64f067ce4ce4cfc7027baf26c719d448"
LEGACY_MODEL_SHA256 = "0b27ee6f952ebbdaea599b67da1e7b1061e3dd4b43bbe730bf2291eed0811349"
PREPARE_SHA256 = "4889b777e79b15c1761698375f9b1f48316e648129cce4f0439a30714fb1a30b"
SCORE_SHA256 = "3c1760481872726d798dd18ec4fb3557d6940e669e4c4d818a07da276d9c477a"
RECORDS_SHA256 = "ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f"
MANIFEST_SHA256 = "b8c290dd64f210dd469b83cbf41ad321ae4d7c5c40303c70e8cda0e4335cd0c3"
SAMPLER_SHA256 = "081ad3a97090651bba4b146e77fcd47807e0942d3ab4f7da450274a1e594b8d3"
FULL_COMMIT_EVAL_SHA256 = "a9065f34257873d0e7c6cc417054514ed3adc3d97b1f7c69e71232e50a11f596"
FULL_COMMIT_FAST_EVAL_SHA256 = "444e6d1fca8bf7f1de00dca951d64961cec5b3b64e3461034174140742b4bb36"
FAST_METADATA_SHA256 = "0163435947c8d8efa4c8f1455fba8ff22080b4a2969e2b98891f9dc9cfba356d"
FAST_METADATA_TEST_SHA256 = "c41ea06b2b941ca289f4033bf431958783f494cd0f6d8d88a6aa51991b76b5d2"
EQUALITY_AUDIT_SHA256 = "d29ba6ba3d928c737bb04aa850dc8e89bec0bac48dc8beb499675d599cd3a6b9"


def _load_runtime():
    global torch, fast_eval, header_probe, commit_probe, fast_metadata, prepare, runner
    import torch

    sys.path.insert(0, str(HERE))
    import full_commit_fast_eval as fast_eval

    header_probe = fast_eval.header_probe
    commit_probe = fast_eval.commit_probe
    fast_metadata = fast_eval.fast_inference_metadata
    prepare = fast_eval.prepare
    runner = fast_eval.runner


def validate_joint_receipt(status, receipt):
    """Fail closed unless these are the frozen 512-update joint B32 weights."""
    assert status["arm"] == receipt["arm"] == "joint_ar_diffusion_b32"
    assert status["mode"] == receipt["mode"] == "hybrid"
    assert status["updates"] == receipt["checkpoint_update"] == 512
    assert status["prediction_offset"] == receipt["prediction_offset"] == 0
    assert status["block_size"] == receipt["block_size"] == 32
    assert status["clean_crystal_ar_weight"] == receipt["clean_crystal_ar_weight"] == 1.0
    assert status["final_fingerprint"] == receipt["final_fingerprint"] == JOINT_FINGERPRINT
    assert receipt["checkpoint_sha256"] == JOINT_CHECKPOINT_SHA256
    assert receipt["reload_fingerprint"] == JOINT_FINGERPRINT
    assert receipt["reload_verified"] is True and receipt["strict_state_load"] is True
    assert status["dataset_records_sha256"] == RECORDS_SHA256
    assert status["dataset_manifest_sha256"] == MANIFEST_SHA256
    assert status["code_hashes"]["model.py"] == JOINT_MODEL_SHA256
    assert status["code_hashes"]["prepare.py"] == PREPARE_SHA256
    assert status["code_hashes"]["score.py"] == SCORE_SHA256


def _load_joint_model(tok, checkpoint):
    name = "_crystaldlm_joint_confidence_model_20261006"
    spec = importlib.util.spec_from_file_location(name, JOINT_ROOT / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model = module.Hybrid.pretrained(
        tok, prediction_offset=0, block_size=32, clean_crystal_ar_weight=1.0
    ).cuda().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = model.restore_checkpoint(
        saved, expected_mode="hybrid", expected_offset=0, expected_block_size=32,
        expected_clean_crystal_ar_weight=1.0,
    )
    del saved
    gc.collect()
    fingerprint = runner.fingerprint(model)
    assert config["final_fingerprint"] == fingerprint == JOINT_FINGERPRINT
    assert model.block_size == 32 and model.prediction_offset == 0
    assert model.clean_crystal_ar_weight == 1.0
    return model, fingerprint, module


def _verify_equality_audit():
    assert runner.sha_file(EQUALITY_AUDIT) == EQUALITY_AUDIT_SHA256
    equality = json.loads(EQUALITY_AUDIT.read_text())
    assert equality["status"] == "passed"
    assert equality["files"] == {
        "experiments/interleaved_relational_blocks_20261005/model.py": LEGACY_MODEL_SHA256,
        "experiments/interleaved_relational_joint_20261006/model.py": JOINT_MODEL_SHA256,
    }
    assert all(equality["identical_functions"].get(name) is True for name in (
        "attention", "crystal_prefix_times", "reverse_step", "unpack_block",
        "Hybrid.hidden", "Hybrid.logits", "Hybrid.text", "Hybrid.crystal",
    ))


def _snapshot(inputs, status, checkpoint):
    joint_sources = {
        name: runner.sha_file(JOINT_ROOT / name) for name in status["code_hashes"]
    }
    assert joint_sources == status["code_hashes"]
    legacy_sources = {
        name: runner.sha_file(ROOT / name) for name in inputs["status"]["code_hashes"]
    }
    assert legacy_sources == inputs["parent_sources"]
    assert legacy_sources["model.py"] == LEGACY_MODEL_SHA256
    assert legacy_sources["prepare.py"] == PREPARE_SHA256
    assert legacy_sources["score.py"] == SCORE_SHA256
    assert runner.sha_file(commit_probe.__file__) == SAMPLER_SHA256
    assert runner.sha_file(JOINT_ROOT / "model.py") == JOINT_MODEL_SHA256
    assert runner.sha_file(JOINT_ROOT / "prepare.py") == PREPARE_SHA256
    assert runner.sha_file(JOINT_ROOT / "score.py") == SCORE_SHA256
    data_hashes = {
        "manifest_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "manifest.json"),
        "records_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "records.jsonl"),
    }
    assert data_hashes == inputs["data_hashes"]
    assert data_hashes == {
        "manifest_sha256": MANIFEST_SHA256, "records_sha256": RECORDS_SHA256,
    }
    reused_sources = {
        "commit_probe.py": runner.sha_file(commit_probe.__file__),
        "full_commit_eval.py": runner.sha_file(HERE / "full_commit_eval.py"),
        "full_commit_fast_eval.py": runner.sha_file(fast_eval.__file__),
        "fast_inference_metadata.py": runner.sha_file(fast_metadata.__file__),
        "test_fast_inference_metadata.py": runner.sha_file(fast_eval.metadata_check.__file__),
    }
    assert reused_sources == {
        "commit_probe.py": SAMPLER_SHA256,
        "full_commit_eval.py": FULL_COMMIT_EVAL_SHA256,
        "full_commit_fast_eval.py": FULL_COMMIT_FAST_EVAL_SHA256,
        "fast_inference_metadata.py": FAST_METADATA_SHA256,
        "test_fast_inference_metadata.py": FAST_METADATA_TEST_SHA256,
    }
    return {
        "checkpoint_sha256": runner.sha_file(checkpoint),
        "joint_status_sha256": runner.sha_file(JOINT_STATUS),
        "joint_receipt_sha256": runner.sha_file(JOINT_RECEIPT),
        "joint_code_hashes": joint_sources,
        "legacy_parent_code_hashes": legacy_sources,
        "reused_code_hashes": reused_sources,
        "own_source_hashes": {
            path.name: runner.sha_file(path)
            for path in (Path(__file__), HERE / "test_joint_confidence_eval.py",
                         HERE / "joint_confidence_eval.slurm")
        },
        "equality_audit_sha256": runner.sha_file(EQUALITY_AUDIT),
        "data_hashes": data_hashes,
        "tokenizer_hashes": header_probe._tokenizer_hashes(),
        "codec_pin": runner.verify_codec_pin(inputs["manifest"]),
    }


def _prefix_sample(model, tok, row, method):
    task = prepare.tasks(row, "train", "minus", 0)[0]
    prefix = task["prefix"] + task["lead"]
    release_seed = header_probe.case_seed(row["id"], 32, "release")
    release_rng = torch.Generator(device="cuda").manual_seed(release_seed)
    torch.cuda.synchronize()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sample = commit_probe.sample_prefix(
            model, tok, prefix, method, release_rng, prefix_tokens=128
        )
    torch.cuda.synchronize()
    return sample, header_probe.header_metrics(sample["text"]), release_seed


def _parity_gate(model, tok, rows, joint_module, original_functions, original_sampler_alias):
    optimized_functions = (joint_module.attention, joint_module.crystal_prefix_times)
    optimized_sampler_alias = commit_probe.crystal_prefix_times
    assert optimized_functions == (
        fast_metadata.fast_attention, fast_metadata.fast_crystal_prefix_times
    )
    cases = []
    for row in rows:
        try:
            joint_module.attention, joint_module.crystal_prefix_times = original_functions
            commit_probe.crystal_prefix_times = original_functions[1]
            baseline, baseline_header, release_seed = _prefix_sample(
                model, tok, row, "legacy_random"
            )
        finally:
            joint_module.attention, joint_module.crystal_prefix_times = optimized_functions
            commit_probe.crystal_prefix_times = optimized_sampler_alias

        optimized, optimized_header, optimized_seed = _prefix_sample(
            model, tok, row, "legacy_random"
        )
        assert release_seed == optimized_seed
        assert baseline["token_ids"] == optimized["token_ids"]
        assert baseline["text"] == optimized["text"]
        assert baseline["time_trace"] == optimized["time_trace"]
        assert baseline["time_trace_sha256"] == optimized["time_trace_sha256"]
        assert baseline["release_trace_sha256"] == optimized["release_trace_sha256"]
        assert baseline_header == optimized_header
        cases.append({
            "id": row["id"], "split": "train_seen8", "release_seed": release_seed,
            "generated_tokens": len(baseline["token_ids"]),
            "token_ids_sha256": runner.sha(json.dumps(
                baseline["token_ids"], separators=(",", ":")
            ).encode()),
            "time_trace_sha256": baseline["time_trace_sha256"],
            "release_trace_sha256": baseline["release_trace_sha256"],
            "header": baseline_header,
            "exact_tokens_text_time_release_header": True,
            "unoptimized_seconds": baseline["elapsed_seconds"],
            "optimized_seconds": optimized["elapsed_seconds"],
        })
    assert len(cases) == 8
    assert (joint_module.attention, joint_module.crystal_prefix_times) == optimized_functions
    assert commit_probe.crystal_prefix_times is optimized_sampler_alias
    assert original_sampler_alias is not None
    return {
        "status": "passed", "method": "legacy_random_greedy_tokens",
        "prefix_tokens": 128, "seed_source": "header_probe.case_seed(case_id, 32, stream)",
        "exact_cases": len(cases), "expected_cases": 8,
        "compared": ["token_ids", "text", "time_trace", "release_trace", "header"],
        "cases": cases,
    }


def run(out=DEFAULT_OUT):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite joint confidence evaluation: {out}")
    _load_runtime()
    device = header_probe.require_gpu_allocation()

    # Checkpoint metadata and weights are first read after allocation is confirmed.
    status = json.loads(JOINT_STATUS.read_text())
    receipt = json.loads(JOINT_RECEIPT.read_text())
    validate_joint_receipt(status, receipt)
    assert status["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert status["model_revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert status["reader_schedule"] == "alternating_q1_q2" and status["seed"] == 17
    _verify_equality_audit()
    assert runner.sha_file(JOINT_CHECKPOINT) == JOINT_CHECKPOINT_SHA256

    inputs = header_probe._verified_inputs()
    assert len(inputs["rows"]) == 8
    assert runner.sha_file(commit_probe.__file__) == SAMPLER_SHA256
    assert inputs["codec_pin"] == runner.verify_codec_pin(inputs["manifest"])
    tok = prepare.tokenizer()
    seen8 = inputs["rows"]
    assert len(seen8) == 8 and all(row["split"] == "train" for row in seen8)

    hashes_before = _snapshot(inputs, status, JOINT_CHECKPOINT)
    assert hashes_before["checkpoint_sha256"] == JOINT_CHECKPOINT_SHA256
    model, fingerprint_before, joint_module = _load_joint_model(tok, JOINT_CHECKPOINT)
    original_functions = (joint_module.attention, joint_module.crystal_prefix_times)
    original_sampler_alias = commit_probe.crystal_prefix_times
    try:
        metadata_equivalence = fast_eval.install_and_check_fast_metadata(model, "cuda")
        parity = _parity_gate(
            model, tok, seen8, joint_module, original_functions, original_sampler_alias
        )
        assert runner.fingerprint(model) == fingerprint_before == JOINT_FINGERPRINT
        assert model.block_size == 32 and model.prediction_offset == 0
        assert model.clean_crystal_ar_weight == 1.0

        out.mkdir(parents=True, exist_ok=False)
        runner.write_json(out / "optimization_gate.json", {
            "status": "passed", "checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
            "fingerprint": fingerprint_before,
            "optimization_verification": metadata_equivalence,
            "joint_unoptimized_optimized_parity": parity,
            "source_hashes_before": hashes_before,
        })

        fast_eval.bind_policy(model, "confidence_one")
        seen = runner.evaluate(
            model, tok, "diffusion_b32", seen8, out / "seen8"
        )
        fingerprint_after = runner.fingerprint(model)
        assert fingerprint_after == fingerprint_before == JOINT_FINGERPRINT
        assert model.block_size == 32 and model.prediction_offset == 0
        assert model.clean_crystal_ar_weight == 1.0
        hashes_after = _snapshot(inputs, status, JOINT_CHECKPOINT)
        assert hashes_after == hashes_before
        summary = {
            "status": "complete", "diagnostic": "frozen joint-trained confidence-one full-K evaluation",
            "training_arm": "joint_ar_diffusion_b32", "training_updates": 512,
            "clean_crystal_ar_weight": 1.0, "checkpoint_sha256": JOINT_CHECKPOINT_SHA256,
            "fingerprint_before": fingerprint_before, "fingerprint_after": fingerprint_after,
            "model_block_size": model.block_size, "prediction_offset": model.prediction_offset,
            "core_mode": "hybrid", "evaluator_mode": "diffusion_b32",
            "commit_policy": "confidence_one", "token_selection": "greedy_argmax",
            "generated_tokens_per_crystal": prepare.K,
            "seen8": seen, "heldout_evaluated": False,
            "optimizer_updates": 0, "ar_fallback": False, "repair_applied": False,
            "optimization_verification": metadata_equivalence,
            "joint_unoptimized_optimized_parity": parity,
            "runtime": {
                "slurm_job": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
                "packages": {name: importlib.metadata.version(name) for name in (
                    "torch", "transformers", "numpy", "pyxtal", "pymatgen", "spglib"
                )},
            },
            "device": device, "hashes_before": hashes_before, "hashes_after": hashes_after,
        }
        runner.write_json(out / "summary.json", summary)
    finally:
        joint_module.attention, joint_module.crystal_prefix_times = original_functions
        commit_probe.crystal_prefix_times = original_sampler_alias
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
