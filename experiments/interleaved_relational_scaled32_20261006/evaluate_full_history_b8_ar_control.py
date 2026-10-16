"""Same frozen hybrid B8 weights; greedy causal AR crystal bodies as a diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType
import torch

HERE = Path(__file__).resolve().parent
assert hashlib.sha256((HERE / "evaluate_full_history_b8.py").read_bytes()).hexdigest() == (
    "edbccaedd80cc0fb2d064fc7315d2623a5ec7c471c72786f427ff5aa9af9cbf7"
)
import evaluate_full_history_b8 as frozen

POLICY = "greedy_native_ar_crystal_control"
DEFAULT_OUT = HERE / "eval_full_history_b8_ar_control_001"
_LOAD_RUNTIME = frozen.BASE._load_runtime
_SNAPSHOT = frozen.BASE._snapshot


def bind_native_ar(model, module):
    @torch.inference_mode()
    def crystal(self, prefix, tok, mode, generator, temperature=.8):
        assert mode == "hybrid" and self.block_size == 8
        assert self.prediction_offset == 0 and self.clean_crystal_ar_weight == 1.0
        context = tok.encode(prefix, add_special_tokens=False)
        assert context[-1] == self.start_id and len(context) + module.K <= module.MAXLEN
        banned = list(set(tok.all_special_ids) - {self.end_id, self.pad_id})
        raw = []
        for _ in range(module.K):
            ids = torch.tensor(context + raw, device=self.time.weight.device)
            logits = self.logits(ids, "ar", positions=[len(ids) - 1])[0].float()
            logits[banned] = -torch.inf
            value = int(logits.argmax())
            raw.append(value)
            if value == self.end_id:
                raw += [self.pad_id] * (module.K - len(raw))
                break
        try:
            content = module.unpack_block(raw, self.end_id, self.pad_id)
        except ValueError as error:
            raise module.CrystalBlockError(
                str(error) + ": " + tok.decode(raw, skip_special_tokens=False), raw
            ) from error
        return tok.decode(content, skip_special_tokens=False), raw

    model.crystal = MethodType(crystal, model)


def load_runtime():
    runtime = _LOAD_RUNTIME()

    def bind(model, method):
        assert method == "confidence_one"  # Frozen runner's single bind call.
        bind_native_ar(model, runtime["model_module"])

    runtime["fast_eval"].bind_policy = bind
    # This diagnostic evaluates all frozen heldout IDs even when its seen G is zero.
    runtime["runner"].seen8_generation_gate_passed = lambda seen: True
    return runtime


def snapshot(inputs, runtime):
    result = _SNAPSHOT(inputs, runtime)
    result["ar_control_sources"] = {
        name: frozen.BASE.sha_file(HERE / name) for name in (
            "evaluate_full_history_b8_ar_control.py", "test_evaluate_full_history_b8_ar_control.py",
            "evaluate_full_history_b8_ar_control.slurm",
        )
    }
    result["ar_control_policy"] = POLICY
    return result


def annotate_summary(summary):
    sampling = summary.setdefault("sampling", {})
    sampling.pop("noise_time_rule", None)
    sampling.pop("block_size", None)
    sampling.update(policy=POLICY, body_mode="ar", token_selection="greedy_argmax",
                    raw_tokens_per_crystal=2560, autoregressive_end_then_pad=True,
                    early_eos=True, ar_fallback=False, grammar_repair=False,
                    physical_repair=False, oracle_numbers_at_inference=False)
    summary["diagnostic_all_heldout"] = True
    summary["diagnostic_scope"] = "Same hybrid-trained checkpoint; only crystal body inference switches to causal AR."
    summary["boundary_limitation"] = (
        "Native AR stops on END and fills PAD mechanically; PAD is not model-predicted. "
        "Termination advantage alone cannot establish diffusion infeasibility."
    )


def run(pin=frozen.DEFAULT_PIN, out=DEFAULT_OUT):
    frozen.BASE._load_runtime = load_runtime
    frozen.BASE._snapshot = snapshot
    result = frozen.run(pin, out)
    for path in (result / "summary.json", result / "seen8/evaluation_summary.json",
                 result / "heldout16/evaluation_summary.json"):
        summary = json.loads(path.read_text())
        annotate_summary(summary)
        if path.name == "summary.json":
            summary["stage"] = "frozen_b8_ar_crystal_control"
            summary["optimization_gate_semantics"] = "Original CF metadata/parity gate; not an AR quality gate."
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, default=frozen.DEFAULT_PIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run(args.pin, args.out)
