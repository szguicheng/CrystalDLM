# Shared AR / crystal diffusion technical preflight

Scope: use the completed seed-42 one-fifth subset to test the agreed shared core: English autoregression, PLaID token diffusion, then autoregression conditioned on the crystal. This is an implementation and runtime test. The complete scientific training protocol and property evaluation remain on their existing decision tickets.

## Frozen inputs and initialization

- Train 7,272 / validation 900, using the existing fixed material IDs. No test records.
- Join the immutable fact rows and successful annotation receipts by material ID; retain exact archived MP ehull in eV/atom, original provenance, PLaID text and clean English structure description.
- Fresh base: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, already cached on Burgundy. Do not use the earlier eight-material adaptation checkpoint.
- Reuse the existing four crystal control tokens and language tokenizer. No new crystal tokenizer or property encoder.
- Technical preflight constants: crystal capacity `544` tokens including END, 32 diffusion steps, maximum complete sequence `2048`, diffusion prediction offset `0`. Verify these against every sample with the actual 1.5B tokenizer; never truncate.

## Sequence and supervision

1. An English request supplies the original continuous ehull value and unit as a target condition.
2. The shared model autoregressively generates the cleaned short structure description followed by `CRYSTAL_START`.
3. The same model performs absorbing-mask diffusion on PLaID tokens, including END/PAD boundaries. Only earlier clean text and the current noisy crystal are visible to these queries.
4. The model can continue with an AR factual summary of the crystal. During training this is a separate next-segment example conditioned on the completed reference crystal; during generation it must receive the actual generated crystal.

The generation example and continuation example share weights and optimizer. Future continuation targets never enter crystal denoising context. No editing task, property scorer or inferred performance explanation is added.

## Runtime and checks

- Single A100, four CPUs, 48 GB host RAM, at most two hours allocation. Twenty AdamW updates at `2e-5`, a generation/continuation pair per update, seed 42, BF16 autocast with the existing FP32 parameter implementation.
- Before training: source bindings, tokenizer lengths, tiny-model future isolation, finite text/crystal losses and gradients.
- During training: record both losses, finite gradient norm, step time and peak GPU memory.
- Save and reload the checkpoint and verify exact model-state equality.
- Record three raw validation generation attempts, END/PAD validity, PLaID parser validity and continuation text. Retain failures in the denominator; do not repair outputs or substitute reference structures.

This preflight can establish that the new data and training chain run on A100. Twenty updates do not establish convergence, property control or generalization. A raw-generation failure is useful evidence for the subsequent short learning experiment, not a reason to erase the run.
