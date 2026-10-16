"""CPU checks for commit policy invariants; no checkpoint is loaded."""
from types import SimpleNamespace

import torch

from commit_probe import METHODS, sample_prefix, top1_confidences


class Tokenizer:
    all_special_ids = [0, 1, 2, 3, 4, 5, 6]

    def encode(self, text, add_special_tokens=False):
        return [1, 3]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


class Model:
    def __init__(self, tied=False):
        self.time = SimpleNamespace(weight=SimpleNamespace(device=torch.device("cpu")))
        self.start_id, self.end_id, self.mask_id, self.pad_id = 3, 4, 5, 6
        self.block_size = 32
        self.prediction_offset = 0
        self.forward_count = 0
        self.tied = tied

    def logits(self, ids, mode, span, k, positions, times=None):
        self.forward_count += 1
        start, end = span
        assert mode == "hybrid" and end - start == 32
        assert times[:2].tolist() == [-1, -1]
        assert times[2:start].tolist() == [0] * (start - 2)
        assert times[start:end].tolist() == [k] * 32
        block = ids[start:end]
        masked = (block == self.mask_id).nonzero().flatten().tolist()
        assert positions == [start + i for i in masked]
        result = torch.full((len(positions), 8), -20.0)
        for row, position in enumerate(positions):
            result[row, self.end_id] = 1.0 if self.tied else float(position - start + 1) / 16
            result[row, self.pad_id] = 0.0
        return result


def check_one_commit_methods_fill_128_tokens_and_preserve_mask_times():
    for method in ("uniform_one", "confidence_one", "left_to_right_one"):
        model = Model()
        result = sample_prefix(model, Tokenizer(), "prefix", method,
                               torch.Generator().manual_seed(7), prefix_tokens=128)
        assert method in METHODS
        assert len(result["token_ids"]) == 128
        assert result["token_ids"].count(model.end_id) == 128
        assert len(result["blocks"]) == 4
        for block in result["blocks"]:
            assert len(block["steps"]) == 32
            assert [step["k"] for step in block["steps"]] == list(range(32, 0, -1))
            assert [len(step["committed_positions"]) for step in block["steps"]] == [1] * 32
            assert all(
                step["committed_positions"][0] in step["active_positions"]
                and step["active_after"] == step["active_before"] - 1
                for step in block["steps"]
            )
        assert model.block_size == 32 and model.prediction_offset == 0


def check_confidence_ties_choose_lowest_position_and_end_does_not_stop_generation():
    model = Model(tied=True)
    result = sample_prefix(model, Tokenizer(), "prefix", "confidence_one",
                           torch.Generator().manual_seed(7), prefix_tokens=128)
    assert result["blocks"][0]["steps"][0]["committed_positions"] == [0]
    assert result["blocks"][0]["steps"][1]["committed_positions"] == [1]
    assert result["token_ids"] == [model.end_id] * 128

    model = Model()
    result = sample_prefix(model, Tokenizer(), "prefix", "confidence_one",
                           torch.Generator().manual_seed(7), prefix_tokens=32)
    assert result["blocks"][0]["steps"][0]["committed_positions"] == [31]


def check_top1_confidence_is_softmax_probability():
    logits = torch.tensor([[2.0, 1.0, 0.0], [102.0, 101.0, 100.0]])
    confidence = top1_confidences(logits)
    assert torch.all((confidence > 0) & (confidence < 1))
    torch.testing.assert_close(confidence[0], confidence[1])
    torch.testing.assert_close(confidence[0], torch.tensor(0.66524094))


if __name__ == "__main__":
    torch.set_num_threads(1)
    check_one_commit_methods_fill_128_tokens_and_preserve_mask_times()
    check_confidence_ties_choose_lowest_position_and_end_does_not_stop_generation()
    check_top1_confidence_is_softmax_probability()
    print("Commit probe checks passed: fixed releases, clean-prefix times, stable ties, softmax confidence")
