# Ehull interleaved technical preflight implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Test that the frozen annotated subset can train and sample with the existing shared text-AR/crystal-diffusion core on Burgundy.

**Architecture:** Reuse the existing single-view absorbing-mask Hybrid implementation in a new experiment directory. Train generation and crystal-conditioned continuation examples with one shared model and optimizer. Preserve the full material/property/text binding.

**Tech Stack:** Standard-library dataset assembly, existing tokenizers/Transformers/PyTorch/PLaID environments, Slurm.

**Spec:** `experiments/ehull_interleaved_smoke_20261013/PROTOCOL.md`.

## Global constraints

- Seed 42, train 7,272 and validation 900, no test or full-dataset annotation.
- Cached Qwen2.5-1.5B revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; no old task-trained initialization.
- Text causal; current noisy crystal bidirectional internally and unable to see future text. Prediction offset 0; 32 diffusion steps; K544; MAXLEN2048; no truncation.
- One A100, four CPUs, 48GB host memory, two-hour allocation maximum, 20 AdamW updates at 2e-5, seed42.
- Original source data, annotation receipts and previous experiments remain unchanged. No GRACE scores or inferred property success.

## Task 1: Assemble and verify immutable triples

Files: `research/crystal_training_preflight_20261013/assemble.py`, `test_assemble.py`, `data_001/`.

Consumes fixed subset gzip facts and `run_001/successes.jsonl`. Produces split gzip triples with `material_id`, `split`, `plaid_text`, `structure_description`, `structure_description_raw`, `property_labels`, `core_facts`, `facts_sha256`, `request_sha256`.

- [x] Reject a receipt whose label or source fingerprint differs: `assert receipt['facts_sha256'] == sha(row)` and `assert receipt['property_labels'] == row['property_labels']`.
- [x] Join only matching fixed IDs and splits; write atomic gzip outputs and manifest hashes.
- [x] Check count7272/900, no duplicates/missing records; record existing tokenizer lengths without truncation.

## Task 2: Adapt the existing shared model and test it

Files: `experiments/ehull_interleaved_smoke_20261013/prepare.py`, `model.py`, `smoke.py`, `test_smoke.py`.

Consumes assembled triples. `tasks(row)` yields generation (`prefix`, `lead`, `crystal`) and continuation (`prefix`, `target`) dictionaries accepted by the existing Hybrid pack/loss API.

- [x] Reuse model.py with the new constant module and 1.5B initialization.
- [x] Test that changing future crystal tokens cannot change preceding text logits; assert finite text/crystal objectives and gradients on a tiny real Transformer.
- [x] Validate all complete packed sequences using the actual cached 1.5B tokenizer before optimization.
- [x] Run20 updates, save/reload exact parameters, record raw generation and PLaID validity for3 validation materials. Continue AR from actual generated structure; no reference substitution.

## Task 3: Execute and record evidence on Burgundy

Files: `smoke.slurm`, run receipts, execution ticket12.

- [x] Sync only the new experiment and verified triples, create remote logs directory, and run `sbatch --test-only smoke.slurm`.
- [x] Submit `sbatch --parsable smoke.slurm` and record the job ID.
- [x] Check Slurm state, test logs, finite losses, checkpoint-reload evidence and raw generation outputs. Retrieve small receipts, keep large checkpoints on Burgundy.
- [x] Update the execution ticket with observed results. Formal scientific tickets remain open until their actual decisions and criteria are settled.
