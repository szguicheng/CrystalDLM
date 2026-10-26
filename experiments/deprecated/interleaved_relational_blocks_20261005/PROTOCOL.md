# Relational CNL: clean-prefix block diffusion pilot

Approved on 2026-10-05 in the current chat. Stage 1 tests block diffusion with complete fixed-capacity output; it does not reduce PAD supervision or enable early termination.

## Frozen representation and cohort

Reuse `relational_cnl_v1`, its public geometry table and scorer, and byte-identical `interleaved_relational_20261004/data_pilot_001/{manifest.json,records.jsonl}`. Records SHA-256 is `ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f`. The first eight frozen IDs are training/seen8, the next sixteen heldout16. The other 76 are excluded. No structure, sentence, numerical precision or correction is deleted. The prior 100/100 result is an offline codec control only.

The native tokenizer remains the frozen Qwen vocabulary at `interleaved_ar_diffusion_20261002/data_002/tokenizer`, with START/END/MASK/PAD controls. The data manifest's 0.5B model provenance describes the original adapter preparation, not the new training model. Its hashes are checked against the original adapter sources.

## Matched arms

All arms start fresh from `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, seed 17, FP32 trainable parameters and BF16 autocast, zero-initialized time embedding. Gradient checkpointing is enabled uniformly to accommodate duplicated clean/noisy inputs. Never initialize from pilot_001 or pilot_002 fine-tuned checkpoints.

| Arm | Text | Crystal | B | Diffusion prediction offset |
|---|---|---|---:|---:|
| ar | AR | AR | not used | not used |
| diffusion_full | AR | masked diffusion | 2560 | 0 |
| diffusion_b32 | AR | masked diffusion | 32 | 0 |
| diffusion_b64 | AR | masked diffusion | 64 | 0 |

K=2560 is full crystal capacity, STEPS=32, MAXLEN=8192. Tokenization and all packing counts must pass without truncation. The diffusion input contains prefix/lead, one clean K-token target stream and one noisy K-token stream; its actual packed length is the original packed length plus K. The AR input remains the original path.

## Training information flow and objective

Clean positions use token-causal attention and cannot read noisy positions. Noisy block j can read its own noisy block, the prompt/history/lead and clean target positions strictly before jB. It cannot read its own clean target or any later clean/noisy block, including through another position's hidden state. Clean and noisy copies share the same logical position IDs; their physical concatenation must not add K to the noisy RoPE positions. All noisy queries predict their own target position. Ordinary text uses next-token prediction. Every preceding generated crystal position, including PAD after a generated END, uses clean timestep zero during sequential sampling.

Each crystal example uses one shared timestep k sampled uniformly from 1..32 and independently masks each of K positions with probability k/32. The target includes the complete body, one END and the original PAD suffix. Supervise every masked target exactly once: body/END weight 1, PAD weight 0.1, inverse-mask factor 32/k. Sum all block numerators and divide by the original whole-target weight sum; do not average block losses independently. B changes visibility and sequential generation, not target exposure or the loss weighting.

Use the frozen 512-update/effective-batch-4 schedule, two generation tasks, one edit task and one reader task per update. Reader slots alternate Q1/Q2 by epoch, exposing every material to G=128, E=64, Q1=32, Q2=32. Both edit directions remain represented. Keep the existing optimizer, learning-rate schedule and gradient clipping. The 512-update cap tests mechanism behavior, not sufficient pretraining or population-level generalization.

## Sampling and scoring

For each diffusion sample, process K/B blocks in left-to-right order. Start the active B-token block fully MASKed, condition on the actual completed prefix, and use the existing temperature 0.8 and 32-step random reverse release. Previously completed blocks are frozen; no future blocks or gold length are supplied. Generate all K positions, concatenate token IDs, and perform the single original global END/PAD validation. Do not stop early, manufacture PAD, trim garbage or repair malformed output. Whole-document decoder, composition/symmetry/physical checks and exact edit grading remain strict.

The complete flow is G(actual C1) -> Q1(actual C1) -> E(actual C1 to C2) -> Q2(actual C2). Keep reference reading/editing, replacement-crystal and no-crystal checks distinct. Report failures in the denominator. Separate seen8 and heldout16, noting the heldout length shift.

Save raw token IDs including failures, attempt seconds, forward-call counts, checkpoint/block configuration, data/code hashes, exposures and initialization/reload receipts. Diagnose masked body/END/PAD predictions at k=32,16,1 on seen targets if implemented; teacher-forced diagnostics are separate from free-generation success. No oracle values enter sampling. Sequential small blocks may be slower: the maximum denoising calls are 32*ceil(K/B), excluding AR triggers/readers.

## Execution gates

Run compact CPU leakage/alignment/loss/boundary tests and all 24-case pack/source-pin checks first. An independent review must resolve correctness findings before submission. Synchronize the new experiment only; preserve predecessor hashes and artifacts. Downloading pinned public weights is I/O on the login node; model inference/optimization runs only in a Slurm GPU allocation.

On one A100 80GB allocation, perform a disposable effective-batch optimizer step for each of the four arms, including Q2 and the longest actual diffusion pack. Set copied G/E preflight examples to k=32 (all MASK) to measure the maximum supervised-logit load; this does not change the real training noise schedule. Require finite losses/gradients, identical initial fingerprints and peak reserved memory <=70GiB. Disposable steps are excluded from training; reinitialize before each real arm. If this check fails, retain its receipt and fix the actual issue before starting the pilot. Do not silently reduce K, drop data, shorten the history or change only one arm.

Output: a new `pilot_001` directory. A submission receipt or preflight pass is not a learned-model result. This pilot does not change the previously paused monitoring automation.
