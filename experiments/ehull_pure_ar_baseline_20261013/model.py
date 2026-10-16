"""Freeze the shared model's existing causal route as a pure-AR baseline."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

from prepare import MAXLEN

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/ehull_interleaved_smoke_20261013/model.py"
_spec = spec_from_file_location("ehull_interleaved_shared_model", SOURCE)
_shared = module_from_spec(_spec)
_spec.loader.exec_module(_shared)


class ARBaseline(_shared.Hybrid):
    @classmethod
    def pretrained(cls, tok):
        return super().pretrained(tok, prediction_offset=0)

    def restore_checkpoint(self, saved, expected_mode="ar", expected_offset=0):
        assert expected_mode == "ar" and expected_offset == 0
        return super().restore_checkpoint(saved, expected_mode="ar", expected_offset=0)

    def loss(self, examples, tok, mode, generator):
        assert mode == "ar"
        return super().loss(examples, tok, "ar", generator)

    @torch.inference_mode()
    def text(self, prefix, tok, mode, trigger=False, limit=160):
        assert mode == "ar"
        device = self.time.weight.device
        ids = tok.encode(prefix, add_special_tokens=False)
        generated = []
        banned = [self.mask_id, self.pad_id, self.end_id]
        if not trigger:
            banned.append(self.start_id)
        for _ in range(limit):
            x = torch.tensor(ids + generated, device=device)
            if len(x) >= MAXLEN:
                raise ValueError("text context capacity exceeded")
            logits = self.logits(x, "ar", positions=[len(x) - 1])[0].float()
            logits[banned] = -torch.inf
            generated.append(int(logits.argmax()))
            if generated[-1] == tok.eos_token_id or (trigger and generated[-1] == self.start_id):
                break
        response = tok.decode(generated, skip_special_tokens=False)
        if trigger and (not generated or generated[-1] != self.start_id):
            error = ValueError("AR failed to trigger crystal generation: " + response)
        elif not trigger and (not generated or generated[-1] != tok.eos_token_id):
            error = ValueError("AR response failed to terminate: " + response)
        else:
            return response
        error.raw_tokens, error.raw_text = generated, response
        raise error

    def crystal(self, prefix, tok, mode, generator, temperature=.8):
        assert mode == "ar"
        return super().crystal(prefix, tok, "ar", generator, temperature)
