"""CPU-only B4 receipt, sampler, and full-heldout diagnostic contract."""
import ast
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import evaluate_full_history_b4 as evaluation
from test_evaluate_full_history import _new_receipt_fixture


def sampler_contract():
    import torch
    sampler_path = evaluation.BASE.DIAGNOSTIC_ROOT / "commit_probe.py"
    source = sampler_path.read_bytes()
    text = source.decode()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "sample_prefix")
    assert [ast.unparse(node) for node in function.decorator_list] == ["torch.inference_mode()"]
    original = "@torch.inference_mode()\n" + ast.get_source_segment(text, function)
    adapted = evaluation.adapted_sampler_source(source)

    class Tokenizer:
        all_special_ids = [0, 1, 2, 3]

        def encode(self, text, add_special_tokens=False):
            return [1]

        def decode(self, ids, skip_special_tokens=False):
            return json.dumps(ids)

    class Model:
        start_id, end_id, mask_id, pad_id = 1, 2, 3, 0

        def __init__(self):
            self.time = SimpleNamespace(weight=torch.zeros(1, requires_grad=True))
            self.forward_count = 0
            self.spans = []

        def logits(self, ids, mode, span, k, queries, times):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled(), \
                "adapted sampler lost the original inference context"
            self.forward_count += 1
            self.spans.append((span[1] - span[0], k, len(queries)))
            result = torch.full((len(queries), 7), -4.0)
            for index, query in enumerate(queries):
                result[index, 4 + query % 2] = 2.0 + index / 100
            return result + self.time.weight.sum()

    def sample(body, block_size):
        namespace = {
            "torch": torch, "time": time, "json": json, "BLOCK_SIZE": block_size,
            "PREFIX_TOKENS": 128, "METHODS": ("confidence_one",),
            "prepare": SimpleNamespace(STEPS=32, MAXLEN=8192),
            "crystal_prefix_times": lambda ids, *_: torch.zeros_like(ids),
            "top1_confidences": lambda logits: logits.softmax(-1).max(-1).values,
            "header_probe": SimpleNamespace(sha256=lambda data: hashlib.sha256(data).hexdigest()),
        }
        exec(compile(body, str(sampler_path), "exec"), namespace)
        model = Model()
        result = namespace["sample_prefix"](model, Tokenizer(), "prefix", "confidence_one",
                                             torch.Generator().manual_seed(17), prefix_tokens=64)
        return result, model

    old32, _ = sample(original, 32)
    new32, _ = sample(adapted, 32)
    assert {k: v for k, v in old32.items() if k != "elapsed_seconds"} == {
        k: v for k, v in new32.items() if k != "elapsed_seconds"
    }
    small, model4 = sample(adapted, 4)
    assert small["model_calls"] == small["released_positions"] == len(small["token_ids"]) == 64
    assert small["time_trace"] == [[32, 24, 16, 8]] * 16
    assert all(size == 4 and k * 4 == remaining * 32 for size, k, remaining in model4.spans)


def demo():
    assert "torch" not in sys.modules
    assert evaluation.BASE.ARM_DIR == evaluation.TRAIN_DIR / "joint_ar_diffusion_b4"
    assert evaluation.TRAIN_DIR == evaluation.HERE / "training_full_history_b4_001"
    assert evaluation.BASE.DEFAULT_PIN == evaluation.HERE / "eval_full_history_b4_input_pin.json"
    assert evaluation.BASE.DEFAULT_OUT == evaluation.HERE / "eval_full_history_b4_001"
    pin, status, receipt, summary = _new_receipt_fixture()
    try:
        evaluation.validate_training_receipts(pin, status, receipt, summary)
    except ValueError as error:
        assert "training configuration" in str(error)
    else:
        raise AssertionError("B32 receipts passed the B4 gate")
    for value in (status, receipt):
        value.update(arm="joint_ar_diffusion_b4", block_size=4)
    for sources in (status["code_hashes"]["parent"], receipt["source_hashes"]["parent"]):
        sources.update({"../interleaved_relational_scaled32_20261006/" + name: digest
                        for name, digest in evaluation.TRAIN_SOURCE_PINS.items()})
    for key, path in (
        ("status_path", evaluation.ARM_DIR / "status.json"),
        ("checkpoint_receipt_path", evaluation.ARM_DIR / "checkpoint_receipt.json"),
        ("summary_path", evaluation.TRAIN_DIR / "summary.json"),
        ("checkpoint_path", evaluation.ARM_DIR / "checkpoint.pt"),
    ):
        pin[key] = path.relative_to(evaluation.PROJECT).as_posix()
        assert evaluation.BASE._pin_path(pin, key, path) == path.resolve()
    assert evaluation.validate_training_receipts(pin, status, receipt, summary)[:2] == (
        pin["final_fingerprint"], pin["checkpoint_sha256"],
    )
    prepared = evaluation.build_input_pin(status, receipt, summary)
    assert prepared["training_job"] == int(status["slurm_job"])
    assert prepared["checkpoint_sha256"] == receipt["checkpoint_sha256"]
    assert evaluation.validate_training_receipts(prepared, status, receipt, summary)[:2] == (
        pin["final_fingerprint"], pin["checkpoint_sha256"],
    )
    bad_status = copy.deepcopy(status)
    bad_status["reader_recipe"] = "neutral only"
    try:
        evaluation.validate_training_receipts(pin, bad_status, receipt, summary)
    except ValueError as error:
        assert "reader recipe" in str(error)
    else:
        raise AssertionError("wrong reader history passed")

    runner = SimpleNamespace(STEPS=32, K=2560, MAXLEN=8192)
    evaluation.configure_runner(runner)
    assert runner.ARMS == ("joint_ar_diffusion_b4",)
    assert runner.BLOCK_SIZES == {"joint_ar_diffusion_b4": 4}
    assert runner.seen8_generation_gate_passed({"counts": {"G": 0}}) is True
    assert (runner.STEPS, runner.K, runner.MAXLEN) == (32, 2560, 8192)
    assert 'len(train_ids) == 32' in evaluation.adapted_source
    assert 'expected = {"G": 128, "E": 64, "Q1": 32, "Q2": 32}' in evaluation.adapted_source
    assert 'model, tok, "joint_ar_diffusion_b4", cohort["heldout16"]' in evaluation.adapted_source

    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / "eval"
        out.mkdir()
        (out / "summary.json").write_text(json.dumps({"sampling": {}}))
        original_run = evaluation.BASE.run
        evaluation.BASE.run = lambda *_: out
        try:
            evaluation.run("unused-pin", out)
        finally:
            evaluation.BASE.run = original_run
        result = json.loads((out / "summary.json").read_text())
    assert result["diagnostic_all_heldout_even_if_seen_gate_failed"] is True
    assert result["training_variant"] == "full_history_reader_repair_b4"
    assert result["sampling"] == {
        "block_size": 4, "noise_time_rule": evaluation.SAMPLER_TIME_RULE,
    }
    assert "torch" not in sys.modules
    sampler_contract()
    print("B4 evaluation CPU checks passed: receipts, strict full-history protocol, full-heldout gate, inference-time sampler")


if __name__ == "__main__":
    demo()
