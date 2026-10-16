"""Run the frozen seen8 evaluator with full-K greedy crystal commit policies."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from types import MethodType
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from diagnostics.header_probe_20261006 import probe as header_probe
from diagnostics.root_cause_20261006 import commit_probe
import fast_inference_metadata
import test_fast_inference_metadata as metadata_check
from model import CrystalBlockError, unpack_block
import prepare
import run as runner


CHECKPOINT_SHA256 = "79359e4c1936782f925b35a9fbd9b9545add49e29dff8398057aa9868c14d4e0"
FINGERPRINT = "474810720624206b4623a586f7ffc9d099ae84ff0faabf579b259d7d09249350"
SAMPLER_SHA256 = "081ad3a97090651bba4b146e77fcd47807e0942d3ab4f7da450274a1e594b8d3"
POLICIES = ("confidence_one", "uniform_one", "left_to_right_one")
DEFAULT_OUT = HERE / "full_commit_fast_001"


def bind_policy(model, method):
    assert method in POLICIES

    def crystal(self, prefix, tok, mode, generator, temperature=.8):
        assert mode == "hybrid"
        sample = commit_probe.sample_prefix(
            self, tok, prefix, method, generator, prefix_tokens=prepare.K
        )
        raw = sample["token_ids"]
        assert len(raw) == prepare.K
        try:
            content = unpack_block(raw, self.end_id, self.pad_id)
        except ValueError as error:
            message = str(error) + ": " + tok.decode(raw, skip_special_tokens=False)
            raise CrystalBlockError(message, raw) from error
        return tok.decode(content, skip_special_tokens=False), raw

    model.crystal = MethodType(crystal, model)


def install_and_check_fast_metadata(model, device="cuda"):
    originals = fast_inference_metadata.install(model)
    commit_probe.crystal_prefix_times = fast_inference_metadata.fast_crystal_prefix_times
    same = metadata_check.same
    old_baselines = (metadata_check.baseline_attention,
                     metadata_check.baseline_crystal_prefix_times)
    counts = {"masks": 0, "integer_vectors": 0, "total": 0}

    def record_equal(actual, expected):
        same(actual, expected)
        counts["masks" if actual.ndim == 4 else "integer_vectors"] += 1
        counts["total"] += 1

    metadata_check.baseline_attention, metadata_check.baseline_crystal_prefix_times = originals
    metadata_check.same = record_equal
    try:
        metadata_check.check(device)
    finally:
        metadata_check.baseline_attention, metadata_check.baseline_crystal_prefix_times = old_baselines
        metadata_check.same = same
    assert counts == {"masks": 360, "integer_vectors": 390, "total": 750}
    return {
        "status": "passed",
        "device": device,
        "bitwise_equal": {
            "attention_masks": True, "regions": True, "crystal_prefix_times": True,
        },
        "reference_binding": "actual originals returned by fast_inference_metadata.install(model)",
        "check": "test_fast_inference_metadata.check",
        "checked_mask_comparisons": counts["masks"],
        "checked_integer_vector_comparisons": counts["integer_vectors"],
        "checked_tensor_comparisons": counts["total"],
    }


def _read_legacy_reference():
    path = header_probe.DEFAULT_OUT / "prefixes.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    selected = [row for row in rows
                if row["block_size"] == 32 and row["selection"] == "greedy"]
    assert len(selected) == 8 and len({row["id"] for row in selected}) == 8
    return path, {row["id"]: row for row in selected}


def verify_legacy_randomgreedy(model, tok, rows, reference):
    cases = []
    for row in rows:
        task = prepare.tasks(row, "train", "minus", 0)[0]
        prefix = task["prefix"] + task["lead"]
        token_seed = header_probe.case_seed(row["id"], 32, "token")
        release_seed = header_probe.case_seed(row["id"], 32, "release")
        release_rng = torch.Generator(device="cuda").manual_seed(release_seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sample = commit_probe.sample_prefix(
                model, tok, prefix, "legacy_random", release_rng,
                prefix_tokens=commit_probe.PREFIX_TOKENS,
            )
        old = reference[row["id"]]
        header = header_probe.header_metrics(sample["text"])
        assert old["token_seed"] == token_seed and old["release_seed"] == release_seed
        assert sample["token_ids"] == old["token_ids"]
        assert sample["text"] == old["text"]
        assert sample["time_trace"] == old["time_trace"]
        assert sample["time_trace_sha256"] == old["time_trace_sha256"]
        assert sample["release_trace_sha256"] == old["release_trace_sha256"]
        assert header == old["header"]
        cases.append({
            "id": row["id"], "token_seed": token_seed, "release_seed": release_seed,
            "token_ids_sha256": runner.sha(json.dumps(sample["token_ids"], separators=(",", ":")).encode()),
            "time_trace_sha256": sample["time_trace_sha256"],
            "release_trace_sha256": sample["release_trace_sha256"],
            "header": header,
            "exact_tokens_text_time_release_header": True,
        })
    assert len(cases) == 8
    return {
        "status": "passed", "method": "legacy_randomgreedy",
        "prefix_tokens": commit_probe.PREFIX_TOKENS,
        "seed_source": "header_probe.case_seed(case_id, 32, stream)",
        "exact_cases": len(cases), "expected_cases": 8,
        "cases": cases,
    }


def _source_hashes():
    return {
        "commit_probe_py_sha256": runner.sha_file(commit_probe.__file__),
        "full_commit_eval_py_sha256": runner.sha_file(HERE / "full_commit_eval.py"),
        "fast_inference_metadata_py_sha256": runner.sha_file(fast_inference_metadata.__file__),
        "test_fast_inference_metadata_py_sha256": runner.sha_file(metadata_check.__file__),
        "full_commit_fast_eval_py_sha256": runner.sha_file(__file__),
    }


def _invariant_snapshot(inputs):
    return {
        "checkpoint_sha256": runner.sha_file(inputs["checkpoint"]),
        "pilot_status_sha256": runner.sha_file(ROOT / "pilot_001/diffusion_b32/status.json"),
        "checkpoint_receipt_sha256": runner.sha_file(
            ROOT / "pilot_001/diffusion_b32/checkpoint_receipt.json"
        ),
        "parent_code_hashes": {
            name: runner.sha_file(ROOT / name) for name in inputs["status"]["code_hashes"]
        },
        "header_probe_py_sha256": runner.sha_file(header_probe.__file__),
        **_source_hashes(),
        "legacy_prefix_reference_sha256": runner.sha_file(
            header_probe.DEFAULT_OUT / "prefixes.jsonl"
        ),
        "data_hashes": {
            "manifest_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "manifest.json"),
            "records_sha256": runner.sha_file(Path(prepare.DATA_DIR) / "records.jsonl"),
        },
        "tokenizer_hashes": header_probe._tokenizer_hashes(),
        "codec_pin": runner.verify_codec_pin(inputs["manifest"]),
    }


def run(out=DEFAULT_OUT):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite full commit evaluation: {out}")
    device = header_probe.require_gpu_allocation()
    inputs = header_probe._verified_inputs()
    tok = prepare.tokenizer()
    rows = inputs["rows"]
    assert len(rows) == 8 and all(row["split"] == "train" for row in rows)
    assert CHECKPOINT_SHA256 == header_probe.CHECKPOINT_SHA256
    assert FINGERPRINT == header_probe.FINGERPRINT
    hashes_before = _invariant_snapshot(inputs)
    assert hashes_before["commit_probe_py_sha256"] == SAMPLER_SHA256
    assert hashes_before["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert hashes_before["parent_code_hashes"] == inputs["parent_sources"]
    assert hashes_before["data_hashes"] == inputs["data_hashes"]
    assert hashes_before["codec_pin"] == inputs["codec_pin"]

    model, fingerprint_before = header_probe._load_frozen_model(tok, inputs["checkpoint"])
    assert fingerprint_before == FINGERPRINT
    assert model.block_size == 32 and model.prediction_offset == 0
    metadata_equivalence = install_and_check_fast_metadata(model, "cuda")
    legacy_path, legacy_reference = _read_legacy_reference()
    legacy_parity = verify_legacy_randomgreedy(model, tok, rows, legacy_reference)
    legacy_parity["reference_path"] = str(legacy_path)
    legacy_parity["reference_sha256"] = runner.sha_file(legacy_path)
    assert runner.fingerprint(model) == fingerprint_before == FINGERPRINT
    out.mkdir(parents=True, exist_ok=False)
    runner.write_json(out / "optimization_gate.json", {
        "status": "passed",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fingerprint": fingerprint_before,
        "optimization_verification": metadata_equivalence,
        "legacy_randomgreedy_parity": legacy_parity,
        "source_hashes_before": _source_hashes(),
        "data_hashes_before": hashes_before["data_hashes"],
    })
    print(json.dumps({
        "status": "optimization_gates_passed", "output": str(out),
        "legacy_exact_cases": legacy_parity["exact_cases"],
        "cuda_metadata_comparisons": metadata_equivalence["checked_tensor_comparisons"],
    }), flush=True)

    policies = {}
    for method in POLICIES:
        policy_dir = out / method
        policy_dir.mkdir()
        bind_policy(model, method)
        evaluation = runner.evaluate(
            model, tok, "diffusion_b32", rows, policy_dir / "seen8"
        )
        assert runner.fingerprint(model) == fingerprint_before
        policies[method] = {
            "policy": method,
            "token_selection": "greedy_argmax",
            "raw_full_K_token_ids_per_crystal": prepare.K,
            "block_size": 32,
            "split": "train_seen8_only",
            "evaluation": evaluation,
        }
        runner.write_json(policy_dir / "policy.json", policies[method])

    fingerprint_after = runner.fingerprint(model)
    assert fingerprint_after == fingerprint_before == FINGERPRINT
    assert model.block_size == 32 and model.prediction_offset == 0
    hashes_after = _invariant_snapshot(inputs)
    assert hashes_after == hashes_before
    summary = {
        "status": "complete",
        "diagnostic": "bitwise-equivalent metadata optimization plus raw full-K seen8 strict evaluation",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "fingerprint_before": fingerprint_before,
        "fingerprint_after": fingerprint_after,
        "model_block_size": model.block_size,
        "prediction_offset": model.prediction_offset,
        "generated_tokens_per_crystal": prepare.K,
        "policies": policies,
        "cases": [row["id"] for row in rows],
        "split": "train_seen8_only",
        "heldout_evaluated": False,
        "optimizer_updates": 0,
        "repair_applied": False,
        "optimization_verification": metadata_equivalence,
        "legacy_randomgreedy_parity": legacy_parity,
        "runtime": {
            "slurm_job": os.environ["SLURM_JOB_ID"], "host": socket.gethostname(),
            "packages": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "numpy", "pyxtal", "pymatgen", "spglib")},
        },
        "evaluation_rng_seeds": {
            row["id"]: int(runner.sha((row["id"] + "20261004").encode())[:8], 16)
            for row in rows
        },
        "trace_recording": "full per-step traces are not persisted; the exact audited sampler is pinned and raw full-K IDs are saved",
        "scope_goal_achieved": False,
        "scope_note": "Optimization equivalence verified; raw full-K strict scoring only, with no heldout cases or repairs.",
        "device": device,
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
    }
    runner.write_json(out / "summary.json", summary)
    del model
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    print(json.dumps({"status": "complete", "output": str(run(args.out))}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
