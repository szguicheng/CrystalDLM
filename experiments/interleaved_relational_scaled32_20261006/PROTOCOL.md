# Scaled32 data preparation

This folder prepares the next data cohort only. It does not train, evaluate a model, submit a job, or write data until `prepare_data.py --prepare` is explicitly run after review.

The input is the frozen 100-case selection and its verified `relational_cnl_v1` source artifacts. The existing 24 `records.jsonl` rows remain the byte-for-byte prefix: original train8 followed by the unchanged heldout16. Append only selection IDs `[24:48]` as train, giving 32 train and 16 heldout rows. Selection IDs `[48:100]` remain untouched. The preparation verifies all 100 selection, input-CIF, plan, facts, text, tokenizer, codec, and runtime pins against the saved receipts before building rows.

For each appended structure, reuse the saved plan, facts, CIF, and source CNL text. Recheck exact plan decoding and canonical text, source geometry, and the physical audit. Build plus/minus targets by multiplying only the first three lattice lengths by exact decimal factors `1.05` and `0.95`; require exact target decode, canonical text, whole-geometry match, and a passing physical audit for both. Any failed source, edit, or capacity check aborts the batch with its case and direction. Cases are never filtered, replaced, or truncated.

Use `K=2560`, `MAXLEN=8192`, the four existing blockrunner arms, both edit directions, train variants 0 and 1, and heldout variant 0. Every crystal body plus END must fit K. For G/E, gate the dual-stream length using `len(tokenizer(prefix + lead)) + 2K`, and also gate the runner’s actual separately tokenized prefix and lead plus `2K`. Q1/Q2 use their separately tokenized prefix and target lengths. Check every existing and appended row without constructing model masks.

After all checks pass, `data_001/records.jsonl` contains 48 rows and retains the original 24 bytes exactly. Its manifest records the unchanged heldout IDs and source-CIF hashes, the 24 appended IDs and source hashes, the original prefix SHA-256, full record SHA-256, reference audits, and blockrunner pack-length audit. `prepare_data.py --check` validates the saved file hashes, prefix, split, train/heldout ID and CIF separation, and the 52 untouched IDs without loading the codec or model.

The result is a data-preparation artifact, not learned-model evidence. A later 64 epochs per train case would mean 2,048 updates for 32 train cases; training is not scheduled by this protocol.

## Authorized training and final evaluation stage (2026-10-07)

The data-preparation stage above is complete. The subsequent training stage uses a fresh pinned Qwen2.5-1.5B-Instruct initialization, seed 17, B=32, the shared AR/diffusion backbone, and clean-crystal AR loss weight 1. Job 607943 runs `train.slurm` for exactly 2,048 updates, with 32 training structures and the original 16 heldout structures excluded from training. Each training structure receives G=128, E=64, Q1=32 and Q2=32 examples. Grounded reader examples alternate neutral and conflicting train-only history; Q2 uses the frozen parent's reader direction and omits the earlier Q1 answer in neutral contexts.

This stage changes the cohort, total update budget and reader contexts together. It tests the resulting recipe and cannot establish a data-only causal effect. Training loss and a saved checkpoint are not completion evidence for the research goal.

`evaluate.py` requires terminal update-2048 status, matching source/data pins, a verified strict checkpoint reload, and a root-created `eval_input_pin.json`. Checkpoint hashing and model loading run only in an allocated A100 compute job. The evaluator first runs the original eight training cases, preserving comparison with earlier controls; if any G case passes, it also runs the unchanged heldout16. The other 24 trained cases are explicitly excluded from this evaluation. A blocked heldout gate is reported as not evaluated, never as 0/16.

Evaluation uses full-K=2560 diffusion generation with greedy candidates and confidence-based one-token commits. It retains raw outputs and strict END/PAD, decoder, geometry, requested-condition and exact-edit checks. It does not inject an edit target, repair outputs or use AR crystal inference as a fallback. Actual generated C1 and C2 feed the intervening reader tasks. Task rates, all seven read fields, replacement-crystal controls and failure denominators are reported separately. CUDA metadata equality and paired eight-case original/optimized prefix parity must pass before full evaluation; source, tokenizer, codec, data and checkpoint snapshots must match before and after it.
