"""Small checks for paired prefix-only sampling, without pretrained weights."""
from pathlib import Path
import json
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
assert (ROOT / "probe.py").exists(), "prefix sampling probe implementation is missing"
from probe import sample_prefix, header_metrics


class Tokenizer:
    all_special_ids = [0, 1, 2, 3, 4, 5, 6]
    def encode(self, text, add_special_tokens=False):
        return [1, 3]
    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


class Model:
    def __init__(self):
        self.time = nn.Embedding(33, 1)
        self.start_id, self.end_id, self.mask_id, self.pad_id = 3, 4, 5, 6
        self.block_size = 32
        self.forward_count = 0
        self.blocks = []
    def logits(self, ids, mode, span, k, positions, times=None, **kwargs):
        self.forward_count += 1
        assert len(times) == len(ids)
        assert times[2:span[0]].tolist() == [0] * (span[0] - 2)
        assert times[span[0]:span[1]].tolist() == [k] * (span[1] - span[0])
        if k == 32:
            assert ids[span[0]:span[1]].tolist() == [5] * (span[1] - span[0])
            self.blocks.append(span)
        logits = torch.full((len(positions), 8), -torch.inf)
        for i, position in enumerate(positions):
            logits[i, 4 if position == 2 else 6] = 0
        return logits


def check_sampler():
    for b in (32, 16, 8):
        paired = []
        for selection in ("multinomial", "greedy"):
            model = Model()
            result = sample_prefix(
                model, Tokenizer(), "prefix", b, selection,
                torch.Generator().manual_seed(7), torch.Generator().manual_seed(13),
                prefix_tokens=128,
            )
            assert result["token_ids"] == [4] + [6] * 127
            assert len(model.blocks) == 128 // b
            assert model.block_size == 32
            paired.append(result["release_trace_sha256"])
        assert paired[0] == paired[1], "token choice must not alter the paired release schedule"


def check_header():
    text = ("The input cell edges a, b, and c measure 4.8, 4.8, and 6.6 angstrom; "
            "the angles alpha, beta, and gamma measure 90.0, 90.0, and 120.0 degrees.\n"
            "The public setting is space group 181.")
    assert header_metrics(text)["header_syntax_valid"]
    assert header_metrics("\n  " + text)["header_syntax_valid"]
    unfinished = header_metrics(text.split("\n", 1)[0])
    assert unfinished["header_syntax_valid"]
    assert not unfinished["first_nonempty_line_terminated_by_newline"]
    assert not unfinished["complete_header_syntax_valid"]
    assert not header_metrics(text.replace("4.8, 4.8", "4.8,, 4.8"))["header_syntax_valid"]
    assert not header_metrics(text[:50])["header_syntax_valid"]
    overflow = header_metrics(text.replace("4.8, 4.8", "1e999, 4.8"))
    assert not overflow["header_numbers_finite"]
    json.dumps(overflow, allow_nan=False)


if __name__ == "__main__":
    torch.set_num_threads(1)
    check_sampler()
    check_header()
    print("Prefix probe checks passed: paired release, fixed prefix, unchanged model B, strict header grammar")
