"""Small CPU contract for the frozen-weight greedy AR crystal control."""
from types import SimpleNamespace
import torch
import evaluate_full_history_b8_ar_control as control


class Tokenizer:
    all_special_ids = [2, 3, 4, 5, 6]

    def encode(self, text, add_special_tokens=False):
        return [1, 2]

    def decode(self, ids, skip_special_tokens=False):
        return str(list(ids))


class Model:
    start_id, end_id, mask_id, pad_id = 2, 3, 4, 5
    block_size, prediction_offset, clean_crystal_ar_weight = 8, 0, 1.0

    def __init__(self, values):
        self.values = values
        self.calls = 0
        self.time = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(1)))
        self.text = object()

    def logits(self, ids, mode, positions):
        assert mode == "ar" and positions == [len(ids) - 1]
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        token = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        logits = self.time.weight * torch.zeros((1, 8))
        logits[0, token] = 10
        logits[0, self.mask_id] = 20  # Must be banned despite highest logit.
        return logits


def main():
    runtime = control.load_runtime()
    module = runtime["model_module"]
    assert module.K == 2560
    assert runtime["runner"].seen8_generation_gate_passed({"counts": {"G": 0}})
    assert runtime["runner"].CORE_MODES == {"joint_ar_diffusion_b8": "hybrid"}
    assert runtime["runner"].BLOCK_SIZES == {"joint_ar_diffusion_b8": 8}
    tok = Tokenizer()
    model = Model([7, 3])
    original_text = model.text
    control.bind_native_ar(model, module)
    assert model.text is original_text
    value, raw = model.crystal("prefix", tok, "hybrid", None)
    assert value == "[7]" and raw == [7, 3] + [5] * 2558
    assert model.calls == 2 and model.block_size == 8
    for values, expected_prefix in (([7], [7, 7]), ([5, 3], [5, 3])):
        model = Model(values)
        control.bind_native_ar(model, module)
        try:
            model.crystal("prefix", tok, "hybrid", None)
        except module.CrystalBlockError as error:
            assert len(error.raw_ids) == 2560
            assert error.raw_ids[:2] == expected_prefix
        else:
            raise AssertionError("Invalid raw boundary was accepted")
    summary = {"sampling": {"policy": "confidence_one", "early_eos": False,
                            "block_size": 8, "noise_time_rule": "CF"}, "seen8": {}, "heldout16": {}}
    control.annotate_summary(summary)
    assert summary["sampling"]["policy"] == control.POLICY
    assert summary["sampling"]["autoregressive_end_then_pad"] is True
    assert summary["sampling"]["early_eos"] is True
    assert "noise_time_rule" not in summary["sampling"]
    assert "block_size" not in summary["sampling"]
    assert summary["diagnostic_all_heldout"] is True
    assert control.frozen.BASE.TRAIN_DIR == control.frozen.TRAIN_DIR
    print("passed: native AR routing, disabled gradients, bans, END/PAD, raw failures, frozen B8 config")


if __name__ == "__main__":
    main()
