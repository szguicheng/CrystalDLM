# Explicit crystal repair training

Authorized 2026-10-21: train a new model that learns to modify erroneous crystals from checker feedback, using three idle GPUs. Continue the completed unified FLARE Qwen3-8B checkpoint; preserve the parent and all previous results. No GRACE feedback or property scoring in this round.

## Data and task

Use only the frozen 7272 training and 900 validation material records. Retain their split membership and property labels. First audit each original PLaID target with the frozen strict auditor; only valid targets receive repair labels. Record all excluded IDs and reasons. No benchmark generations or failed validation outputs become training targets.

For each eligible material, create one deterministic, verified single-field error: formula count, declared Sites count, Wyckoff multiplicity/label, or symmetry-constrained coordinate. Use a recorded count-error fallback when a coordinate fault is inapplicable. Verify the corrupted structure fails the intended check and the original target passes.

R input uses the validated fresh-request format: original space-group/property requirement, corrupted crystal as plain input data, checker feedback computed from that corrupted crystal only. R output is a short, factual English correction explanation followed by START and the original valid crystal. Correct target values appear in output labels, not in feedback derived from the reference. Actual source ehull bounds are retained, including 0.08/0.1/0.2/0.3 eV/atom where appropriate.

Preserve the original G generation and Q factual-reading examples for retention. Require original G/Q lengths ≤2048, target crystal length including END ≤544, and R clean trajectory ≤4096 before training. Do not truncate invalid or long examples silently.

## Shared-model objective and optimization

Parent: `experiments/flare_crystal_full_20261019/flare/checkpoint.pt`.

Per material, optimize `0.5 × R_joint + 0.25 × G_joint + 0.25 × Q_AR`. Both R and G use the existing FLARE clean AR plus complementary crystal-token denoising objective; Q remains causal AR. One model shares its core, LoRA and control-token parameters across all tasks. R prefixes contain the previous crystal as plain data, so the original single-current-crystal diffusion masking contract applies.

One epoch over 7272 training rows (1212 updates); fixed shuffle seed 42; three DDP ranks on explicitly selected physical GPUs 1/2/3. Global batch six materials, two microsteps per rank. The final incomplete batch uses zero-weight dummy slots solely to keep collective calls aligned; every real material appears exactly once, and gradients are normalized by the actual global count.

Formal data are `data_002`, algorithm `single-field-diverse-v2`: count faults vary over −4/−2/−1/+1/+2/+4 where counts remain positive, and coordinate faults vary in sign and magnitude. All 8172 targets passed strict validation; all corruptions failed their intended check. `data_001` and its exact `prepare_v1.py` source are retained as engineering history. Formal training does not use the v1 constant-increment faults or smoke weights.

AdamW, LR 5e-5 constant, weight decay0.01, gradient clipping1.0, BF16 autocast, existing gradient checkpointing and trainable parameter set. Save atomic latest checkpoints every50 updates and at controlled stops, with optimizer, per-rank RNG, parent/data/code hashes, ordering and next material index. Save the final checkpoint only after the epoch is complete. Keep only latest and final in each run.

## Gates and evaluation

1. Run focused CPU contracts for verified corruption, input/target separation, objective weights, rank partitioning and final-batch normalization.
2. Finish and inspect the data manifest, counts, corruption types and token limits.
3. On all three GPUs, train two real smoke updates and resume to the third in a separate small data/run directory. Verify finite losses, changing trainable weights, restored optimizer/RNG, and exact material coverage. The formal run starts again from the original parent, not smoke weights.
4. Run the formal epoch, then evaluate losses and teacher-forced START prediction on the independent eligible validation split. These are engineering/learning metrics, not free-generation success.
5. Compare parent and newly trained model on the same fixed32 validation repair inputs, balanced across available corruption types, one generation each with fixed seeds. Let each model generate its correction explanation and crystal; provide no reference explanation or target crystal as input. Report natural switching, strict validity, original requested-space-group success, and unchanged failed-crystal copies separately. Save raw outputs and failure receipts. No training-set examples enter this generation check.

This is a first supervised-repair learning trial. Synthetic-error validation and real model-generated failures are different distributions. Parent-versus-new results are a before/after comparison; a matched continuation without R would be needed to isolate the repair-task contribution for a method claim.

## Execution plan and persistence

- `prepare.py` / `test_prepare.py`: deterministic audited repair data with resumable per-material CPU caches.
- `model.py` / `run.py` / `test_training.py`: joint objectives, three-rank training, strict checkpoint restore and validation.
- `evaluate.py`: fixed validation generation inputs and parent/new comparison, after training releases all GPUs.
- `launch.sh`: project mamba/cache environment, GPU checks, named tmux supervision, stage and exit files, automatic validation/evaluation.

All artifacts stay under `/home/liugang/src/crystalDLM/experiments/crystal_repair_training_20261021`. Project GPU usage never exceeds three. Initial disk check found21GiB free; parent checkpoints are approximately1.1GiB each. Recheck before smoke/formal checkpoint saves. Do not overwrite old data, model weights or failures.
