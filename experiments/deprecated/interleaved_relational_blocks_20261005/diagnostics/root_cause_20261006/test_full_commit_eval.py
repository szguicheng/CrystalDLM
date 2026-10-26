"""CPU integration check for full-K commit-policy binding; no model weights."""
from types import SimpleNamespace
from unittest.mock import patch

import torch

import full_commit_eval
import prepare
from model import CrystalBlockError


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return ",".join(map(str, ids))


def check_full_k_raw_block_boundary():
    model = SimpleNamespace(end_id=4, pad_id=6)
    tok = Tokenizer()
    valid = [7] * (prepare.K - 2) + [model.end_id, model.pad_id]
    calls = []

    def sample_prefix(model, tok, prefix, method, generator, prefix_tokens):
        calls.append((method, prefix_tokens))
        return {"token_ids": valid}

    with patch.object(full_commit_eval.commit_probe, "sample_prefix", sample_prefix):
        full_commit_eval.bind_policy(model, "confidence_one")
        crystal, raw = model.crystal("prefix", tok, "hybrid", torch.Generator())
    assert calls == [("confidence_one", prepare.K)]
    assert len(raw) == prepare.K and raw == valid
    assert crystal == tok.decode(valid[:-2], skip_special_tokens=False)

    for invalid in (
        [7] * prepare.K,
        [7, model.pad_id, model.end_id] + [model.pad_id] * (prepare.K - 3),
        [7, model.end_id, 8] + [model.pad_id] * (prepare.K - 3),
    ):
        def bad_sample_prefix(model, tok, prefix, method, generator, prefix_tokens):
            assert prefix_tokens == prepare.K
            return {"token_ids": invalid}

        with patch.object(full_commit_eval.commit_probe, "sample_prefix", bad_sample_prefix):
            try:
                model.crystal("prefix", tok, "hybrid", torch.Generator())
            except CrystalBlockError as error:
                assert error.raw_ids == invalid
            else:
                raise AssertionError("invalid full-K END/PAD boundary was accepted")


if __name__ == "__main__":
    torch.set_num_threads(1)
    check_full_k_raw_block_boundary()
    print("Full commit evaluation CPU check passed: exact K and strict raw END/PAD boundary")
