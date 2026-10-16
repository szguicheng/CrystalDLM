"""CPU orchestration check for the fast full-evaluation gate."""
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import full_commit_fast_eval as evaluation


class GateRejected(RuntimeError):
    pass


def check_cpu_equivalence():
    name = "_full_commit_fast_eval_cpu_check"
    module = types.ModuleType(name)
    module.attention = evaluation.metadata_check.baseline_attention
    module.crystal_prefix_times = evaluation.metadata_check.baseline_crystal_prefix_times
    sys.modules[name] = module
    Model = type("Model", (), {"__module__": name})
    result = evaluation.install_and_check_fast_metadata(Model(), "cpu")
    assert result["bitwise_equal"] == {
        "attention_masks": True, "regions": True, "crystal_prefix_times": True,
    }
    assert result["checked_mask_comparisons"] == 360
    assert result["checked_integer_vector_comparisons"] == 390
    assert result["checked_tensor_comparisons"] == 750
    del sys.modules[name]


def check_gate_precedes_evaluation():
    name = "_full_commit_fast_eval_gate_check"
    module = types.ModuleType(name)
    module.attention = lambda *args: None
    module.crystal_prefix_times = lambda *args: None
    sys.modules[name] = module
    Model = type("Model", (), {"__module__": name})
    model = Model()
    model.block_size = 32
    model.prediction_offset = 0
    originals = module.attention, module.crystal_prefix_times
    checked, evaluated = [], []
    inputs = {
        "checkpoint": Path("checkpoint.pt"), "rows": [{"id": str(i), "split": "train"} for i in range(8)],
        "status": {"code_hashes": {}}, "parent_sources": {}, "data_hashes": {},
        "codec_pin": None, "manifest": {},
    }
    snapshot = {
        "commit_probe_py_sha256": evaluation.SAMPLER_SHA256,
        "checkpoint_sha256": evaluation.CHECKPOINT_SHA256,
        "parent_code_hashes": {}, "data_hashes": {}, "codec_pin": None,
    }

    def reject_gpu_check(device):
        assert device == "cuda"
        assert evaluation.metadata_check.baseline_attention is originals[0]
        assert evaluation.metadata_check.baseline_crystal_prefix_times is originals[1]
        checked.append(True)
        raise GateRejected("synthetic CUDA-gate failure")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "full_commit_fast_001"
        with patch.object(evaluation.header_probe, "require_gpu_allocation", return_value={"name": "A100"}), \
             patch.object(evaluation.header_probe, "_verified_inputs", return_value=inputs), \
             patch.object(evaluation.prepare, "tokenizer", return_value=object()), \
             patch.object(evaluation, "_invariant_snapshot", return_value=snapshot), \
             patch.object(evaluation.header_probe, "_load_frozen_model", return_value=(model, evaluation.FINGERPRINT)), \
             patch.object(evaluation.runner, "fingerprint", return_value=evaluation.FINGERPRINT), \
             patch.object(evaluation.runner, "evaluate", side_effect=lambda *args: evaluated.append(args)), \
             patch.object(evaluation.metadata_check, "check", side_effect=reject_gpu_check):
            try:
                evaluation.run(out)
            except GateRejected:
                pass
            else:
                raise AssertionError("full evaluation continued after the CUDA gate failed")
        assert checked == [True]
        assert evaluated == []
        assert not out.exists()
    assert module.attention is evaluation.fast_inference_metadata.fast_attention
    assert module.crystal_prefix_times is evaluation.fast_inference_metadata.fast_crystal_prefix_times
    assert evaluation.commit_probe.crystal_prefix_times is evaluation.fast_inference_metadata.fast_crystal_prefix_times
    assert set(evaluation._source_hashes()) == {
        "commit_probe_py_sha256", "full_commit_eval_py_sha256",
        "fast_inference_metadata_py_sha256", "test_fast_inference_metadata_py_sha256",
        "full_commit_fast_eval_py_sha256",
    }
    del sys.modules[name]


if __name__ == "__main__":
    check_cpu_equivalence()
    check_gate_precedes_evaluation()
    print("Fast full-evaluation CPU gate check passed")
