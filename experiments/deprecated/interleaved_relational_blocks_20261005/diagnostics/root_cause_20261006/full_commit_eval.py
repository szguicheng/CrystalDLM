"""Run the frozen seen8 evaluator with full-K greedy crystal commit policies."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from types import MethodType


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.header_probe_20261006 import probe as header_probe
from diagnostics.root_cause_20261006 import commit_probe
from model import CrystalBlockError, unpack_block
import prepare
import run as runner


CHECKPOINT_SHA256 = "79359e4c1936782f925b35a9fbd9b9545add49e29dff8398057aa9868c14d4e0"
FINGERPRINT = "474810720624206b4623a586f7ffc9d099ae84ff0faabf579b259d7d09249350"
SAMPLER_SHA256 = "081ad3a97090651bba4b146e77fcd47807e0942d3ab4f7da450274a1e594b8d3"
POLICIES = ("confidence_one", "uniform_one", "left_to_right_one")
DEFAULT_OUT = HERE / "full_commit_001"


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
        "commit_probe_py_sha256": runner.sha_file(commit_probe.__file__),
        "full_commit_eval_py_sha256": runner.sha_file(__file__),
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
    out.mkdir(parents=True, exist_ok=False)

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
        "diagnostic": "full-K seen8 crystal commit-policy evaluation",
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
        "scope_note": "Seen8 diagnostic only; the broader evaluation goal remains open.",
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
