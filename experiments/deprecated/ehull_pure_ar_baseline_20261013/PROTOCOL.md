# Matched pure-AR baseline: authorized one-epoch run

The current ehull/text/PLaID task has no completed pure-AR baseline. The earlier 20-update hybrid engineering probe is separate evidence and does not satisfy this stage. Pure AR must precede the formal shared AR+diffusion learning experiment.

## Matching contract

- Exact frozen data: train7,272, validation900, seed42. Reuse the verified triples from `research/crystal_training_preflight_20261013/data_001/` without resampling or changing captions/labels.
- Fresh Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Preserve tokenizer, four crystal control tokens, K544 and MAXLEN2048. Record the initial state hash so a later hybrid run must begin from the same parameters.
- Reuse the same property-request, English description, PLaID target and crystal-conditioned continuation examples. The sole decoding/training-mode difference is that all target tokens use causal next-token prediction; no MASK corruption or bidirectional crystal attention.
- Seed42 material order and fixed32 validation IDs. Preserve these identities for the later hybrid run. Report all generation attempts, including trigger, boundary and parser failures.

## Approved first learning run

One epoch over all7,272 training materials, two G/Q examples per material. Microbatch2, accumulation8, effective batch16: **909 optimizer updates**. AdamW learning rate2e-5, weight decay0.01, 5% warmup then cosine decay. Single A100, four CPUs,48GB host RAM, allocation limit4hours within the existing48GPU-hour total budget.

Record finite losses/gradients, exact material exposure/order, runtime, initial/final state fingerprints and raw validation outputs. Structural validity, completed post-crystal reading and property satisfaction are separate metrics. A completed text response does not imply that its facts or property claims are correct.

## Property evaluation boundary

This first run can test sample learning and raw crystal generation. New-crystal ehull satisfaction requires the selected independent evaluator and a compatible reference-hull protocol. That scoring has not been implemented or calibrated. A supplied target ehull, source/reference ehull or generated paragraph is not a measured score of the generated crystal.

Execution authorized by the user on 2026-10-13. Current job identity and measured state are recorded separately in submission.json and run receipts. Run only this pure-AR baseline; analyze its results before the formal joint experiment.
