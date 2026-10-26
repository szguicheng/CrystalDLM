"""CPU-only contracts for the coverage256 strict evaluator."""
import ast
import hashlib
import json
import sys
import time
from copy import deepcopy
from types import SimpleNamespace

import evaluate as evaluation


def cohort_contract():
    cohort = evaluation.read_cohort()
    assert len(cohort["seen8"]) == 8
    assert len(cohort["heldout16"]) == 16
    assert len(cohort["remaining248"]) == 248
    assert len(cohort["train_ids"]) == 256
    assert len({row["id"] for row in cohort["seen8"] + cohort["heldout16"]}) == 24
    assert all(row["split"] == "train" for row in cohort["seen8"] + cohort["remaining248"])
    assert all(row["split"] == "heldout" for row in cohort["heldout16"])

    status = {"actual_exposure_counts": {
        "per_material": {case_id: {"tasks": {"G": 16, "E": 8, "Q1": 4, "Q2": 4}}
                         for case_id in cohort["train_ids"]},
        "total": {"G": 4096, "E": 2048, "Q1": 1024, "Q2": 1024},
    }}
    evaluation.validate_training_exposure(status, cohort["train_ids"])
    status["actual_exposure_counts"]["per_material"][cohort["train_ids"][0]]["tasks"]["Q2"] = 3
    try:
        evaluation.validate_training_exposure(status, cohort["train_ids"])
    except ValueError:
        pass
    else:
        raise AssertionError("an incorrect per-case exposure passed the strict gate")


def holdout_contract():
    runner = SimpleNamespace(seen8_generation_gate_passed=lambda _: False)
    assert evaluation.continue_after_seen(runner, {"counts": {"G": 0}}) is True
    assert runner.coverage256_seen8_generation_gate_passed is False


def receipt_contract():
    fingerprint = "final"
    sources = {"parent": {"model.py": "parent"}, "coverage256": evaluation.TRAIN_SOURCE_PINS}
    common = {"mode": "hybrid", "arm": "joint_ar_diffusion_b8", "prediction_offset": 0,
              "block_size": 8, "clean_crystal_ar_weight": 1.0,
              "reader_mode": "grounded", "reader_recipe": evaluation.READER_RECIPE,
              "final_fingerprint": fingerprint}
    status = dict(common, status="trained", updates=2048, completed_updates=2048, seed=17,
                  K=2720, MAXLEN=8448, initial_fingerprint="e97aed646b7289c06f10be376b37d67cd7aa6a42ad6214b61da008cd85b8404b",
                  reader_schedule="alternating_q1_q2", code_hashes=sources,
                  data_hashes=evaluation.DATA_HASHES, dataset_manifest_sha256=evaluation.MANIFEST_SHA,
                  dataset_records_sha256=evaluation.RECORDS_SHA, checkpoint_reload_verified=True,
                  checkpoint_reload_fingerprint=fingerprint, checkpoint_sha256="checkpoint")
    ids = evaluation.read_cohort()["train_ids"]
    status["actual_exposure_counts"] = {
        "per_material": {i: {"tasks": {"G": 16, "E": 8, "Q1": 4, "Q2": 4}} for i in ids},
        "total": {"G": 4096, "E": 2048, "Q1": 1024, "Q2": 1024},
    }
    receipt = dict(common, status="verified", checkpoint_update=2048, reload_verified=True,
                   reload_fingerprint=fingerprint, strict_state_load=True,
                   source_hashes=sources, checkpoint_sha256="checkpoint")
    summary = dict(common, status="training_only_checkpoint_verified",
                   stage="coverage256_fixed2048_b8_training_only", updates=2048, seed=17,
                   K=2720, MAXLEN=8448, epochs_per_train_case=8,
                   initial_fingerprint=status["initial_fingerprint"], checkpoint_reload_verified=True,
                   model="Qwen/Qwen2.5-1.5B-Instruct", revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
                   evaluation={"train256": "not_run", "heldout16": "not_run", "AR_crystal_inference": "not_run"})
    pin = {"status": "verified", "updates": 2048, "final_fingerprint": fingerprint,
           "checkpoint_sha256": "checkpoint"}
    evaluation.validate_training_receipts(pin, status, receipt, summary)
    for key, value in (("completed_updates", 2047), ("K", 2560), ("checkpoint_reload_verified", False)):
        invalid = deepcopy(status)
        invalid[key] = value
        try:
            evaluation.validate_training_receipts(pin, invalid, receipt, summary)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid terminal receipt accepted: {key}")


def capacity_contract():
    runtime = {name: SimpleNamespace(K=2560, MAXLEN=8192)
               for name in ("prepare", "model_module", "runner")}
    evaluation.configure_capacity(runtime)
    assert all(module.K == 2720 and module.MAXLEN == 8448 for module in runtime.values())


def sampler_contract():
    import torch

    sampler_path = evaluation.FROZEN_B8.BASE.DIAGNOSTIC_ROOT / "commit_probe.py"
    source = sampler_path.read_bytes()
    text = source.decode()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "sample_prefix")
    assert [ast.unparse(node) for node in function.decorator_list] == ["torch.inference_mode()"]
    adapted = evaluation.FROZEN_B8.adapted_sampler_source(source)

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

        def logits(self, ids, mode, span, k, queries, times):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            self.forward_count += 1
            return torch.zeros((len(queries), 7)) + self.time.weight.sum()

    namespace = {
        "torch": torch, "time": time, "json": json, "BLOCK_SIZE": 8,
        "PREFIX_TOKENS": 2720, "METHODS": ("confidence_one",),
        "prepare": SimpleNamespace(STEPS=32, MAXLEN=8448),
        "crystal_prefix_times": lambda ids, *_: torch.zeros_like(ids),
        "top1_confidences": lambda logits: logits.softmax(-1).max(-1).values,
        "header_probe": SimpleNamespace(sha256=lambda data: hashlib.sha256(data).hexdigest()),
    }
    exec(compile(adapted, str(sampler_path), "exec"), namespace)
    model = Model()
    result = namespace["sample_prefix"](
        model, Tokenizer(), "prefix", "confidence_one",
        torch.Generator().manual_seed(17), prefix_tokens=8,
    )
    assert len(result["token_ids"]) == result["released_positions"] == 8
    assert result["time_trace"] == [list(range(32, 0, -4))]
    assert model.forward_count == 8


def demo():
    assert "torch" not in sys.modules
    cohort_contract()
    holdout_contract()
    receipt_contract()
    capacity_contract()
    sampler_contract()
    print("coverage256 evaluation CPU contracts passed")


if __name__ == "__main__":
    demo()
