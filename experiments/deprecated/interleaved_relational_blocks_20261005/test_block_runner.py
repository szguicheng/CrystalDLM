"""CPU-only contracts for the four-arm pilot runner."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from model import Hybrid
import prepare
import run as runner


ROOT = Path(__file__).resolve().parent
Q2_EVENT_SHA256 = "d9c1f0576a30c62d81a9e00bdcf84cd13412057012b85da34d8c154f8424f560"


class _PackOnlyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)


def test_arm_and_source_contract():
    assert runner.ARMS == ("ar", "diffusion_full", "diffusion_b32", "diffusion_b64")
    assert runner.CORE_MODES == {
        "ar": "ar", "diffusion_full": "hybrid", "diffusion_b32": "hybrid", "diffusion_b64": "hybrid",
    }
    assert runner.BLOCK_SIZES == {
        "ar": prepare.K, "diffusion_full": prepare.K, "diffusion_b32": 32, "diffusion_b64": 64,
    }
    assert runner.MODE_OFFSETS == {arm: 0 for arm in runner.ARMS}
    assert runner.DEFAULT_READER_SCHEDULE == "alternating_q1_q2"
    assert prepare.MODEL == "Qwen/Qwen2.5-1.5B-Instruct"
    assert prepare.REVISION == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert prepare.DATA_DIR == ROOT.parent / "interleaved_relational_20261004" / "data_pilot_001"
    assert set(runner.runner_source_hashes()) == {
        path.name for pattern in ("*.py", "*.slurm") for path in ROOT.glob(pattern)
    }

    manifest = prepare.load_manifest()
    provenance = manifest["provenance"]
    assert provenance["model"] == runner.DATA_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"
    assert provenance["revision"] == runner.DATA_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775"
    pin = runner.verify_codec_pin(manifest)
    assert pin["runtime_hashes_stable"]


def test_teacher_forced_metrics_pool_by_token_count():
    aggregate = runner.aggregate_teacher_forced([
        {"metrics": {
            "body": {"count": 2, "correct": 1, "nll_sum": 3.0},
            "end": {"count": 1, "correct": 0, "nll_sum": 2.0},
            "pad": {"count": 0, "correct": 0, "nll_sum": 0.0},
        }},
        {"metrics": {
            "body": {"count": 1, "correct": 1, "nll_sum": 6.0},
            "end": {"count": 0, "correct": 0, "nll_sum": 0.0},
            "pad": {"count": 0, "correct": 0, "nll_sum": 0.0},
        }},
    ])
    assert aggregate["body"] == {
        "count": 3, "correct": 2, "nll_sum": 9.0,
        "accuracy": 2 / 3, "mean_nll": 3.0,
    }
    assert aggregate["end"]["accuracy"] == 0.0 and aggregate["end"]["mean_nll"] == 2.0
    assert aggregate["pad"]["accuracy"] is None and aggregate["pad"]["mean_nll"] is None


def test_schedule_and_actual_pack_lengths():
    _, rows, train, _ = runner.read_data(prepare.DATA_DIR)
    exposure = runner.exposure_receipt(train)
    event_hash = runner.sha(json.dumps(exposure["events"], sort_keys=True, separators=(",", ":")).encode())
    assert event_hash == Q2_EVENT_SHA256 == exposure["events_sha256"]
    assert exposure["total"] == {"E": 512, "G": 1024, "Q1": 256, "Q2": 256}
    actual = runner.summarize_exposure(exposure["events"])
    assert actual["per_material"] == exposure["per_material"]
    for counts in exposure["per_material"].values():
        assert counts == {
            "tasks": {"E": 64, "G": 128, "Q1": 32, "Q2": 32},
            "E_factors": {"minus": 32, "plus": 32},
            "Q2_factors": {"minus": 16, "plus": 16},
        }
    for update in range(1, runner.UPDATES + 1):
        epoch, j, _, _, examples = runner.update_examples(train, update)
        if epoch % 2:
            q2 = examples[-1]
            expected_direction = "minus" if (epoch // 2 + j) % 2 == 0 else "plus"
            assert q2["task"] == "Q2" and q2["reader_direction"] == expected_direction

    tok = prepare.tokenizer()
    lengths = runner.pack_audit(rows, tok)
    preflight_batch, worst = runner._worst_joint_batch(train, tok)
    assert len(preflight_batch) == 4 and len(worst) == 3
    forced_batch = runner.preflight_examples(preflight_batch)
    assert all(example.get("noise_step") == prepare.STEPS for example in forced_batch if "crystal" in example)
    assert all("noise_step" not in example for example in forced_batch if "crystal" not in example)
    assert all(item["worst_actual_packed_tokens"] <= prepare.MAXLEN for item in worst)
    assert max(item["actual_sequence_tokens_by_arm"]["diffusion_full"]
               for item in worst if item["sequence_kind"] == "dual_stream") <= prepare.MAXLEN
    assert all(
        item["actual_sequence_tokens_by_arm"][arm] <= prepare.MAXLEN
        for item in worst for arm in runner.ARMS
    )
    encode = lambda value: len(tok.encode(value, add_special_tokens=False))
    for row in rows:
        for direction in ("minus", "plus"):
            for variant in ((0, 1) if row["split"] == "train" else (0,)):
                for example in prepare.tasks(row, row["split"], direction, variant):
                    for arm in runner.ARMS:
                        body, packed = runner.task_length(example, tok, arm)
                        assert packed <= prepare.MAXLEN
                        if "crystal" in example:
                            assert body == encode(example["crystal"]) + 1 <= prepare.K
                            base = encode(example["prefix"]) + encode(example["lead"])
                            streams = 1 if arm == "ar" else 2
                            assert packed == base + streams * prepare.K
    example = preflight_batch[2]
    ids = [tok.convert_tokens_to_ids(token) for token in (prepare.START, prepare.END, prepare.MASK, prepare.PAD)]
    for arm in runner.ARMS:
        model = Hybrid(
            _PackOnlyBase(), *ids, prediction_offset=runner.MODE_OFFSETS[arm],
            block_size=runner.BLOCK_SIZES[arm],
        )
        packed = model.pack(
            example, tok, runner.CORE_MODES[arm], torch.Generator().manual_seed(17)
        )
        assert packed["ids"].numel() == runner.task_length(example, tok, arm)[1]
        del packed, model
    assert all(lengths[arm][split][task]["max"] <= prepare.MAXLEN
               for arm in runner.ARMS for split in ("train", "heldout")
               for task in ("G", "Q1", "E", "Q2"))
    runner.check_runner_boundaries(rows)


if __name__ == "__main__":
    test_arm_and_source_contract()
    test_teacher_forced_metrics_pool_by_token_count()
    test_schedule_and_actual_pack_lengths()
    print("Block runner checks passed")
