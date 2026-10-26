# Repair model: full conditional generation and structural feedback

Authorized 2026-10-22: evaluate the completed repair-trained shared AR/diffusion model on seven space groups, 100 seeded groups each, using the previous feedback loop and the full downstream ehull evaluation. This extends the existing approved protocol; no training is scheduled.

## Frozen model and requests

- Checkpoint: `experiments/crystal_repair_training_20261021/run_001/checkpoint.pt`, completed 1212 updates.
- Trainable state SHA256: `4775e389f53ef972fccbd4c8c86ad80a3c0cc7e93306ad33eb734d74a419160a`.
- Space groups: 1, 15, 38, 119, 143, 194, 216; 100 fixed seeded requests each, 700 groups per policy.
- Request: original space-group constraint and ehull ≤0.08 eV/atom. No reference structure, correct target, or reference description enters inference.
- Both policies use the same new model. Their first generated candidate is shared; retries use the original deterministic paired seeds. The independent policy receives the original request each time. The feedback policy receives the original request, latest failed response/crystal as plain input data, and checker facts in the corrected fresh-user serialization.

## Loop and output contract

At most three distinct seeded candidate rounds per policy, including the first candidate; stop at the first strict-valid crystal whose decoded space group matches the original request. No relaxed constraints, forced START, hidden candidate repair, or extra quality retries. Each policy has 700 output slots, including failed slots. In an uninterrupted run, at most 3500 physical generations are planned, because the initial 700 are shared. Resume skips persisted attempts; a process interruption before its round receipt is saved can replay that same seeded round. Such operational replay is not an additional candidate-selection opportunity, but its extra compute is not included in receipt-derived generation counts.

Keep frozen sampling: AR temperature0.8/top-p0.95, lead limit256, crystal capacity544 including END, B4 diffusion, context limit4096. Only the latest failed attempt enters retry context; every attempt remains in the stored trajectory. Structural feedback does not include GRACE or ehull scores.

Select exactly the first successful output for each group/policy. If all attempts fail, retain the terminal failed receipt. Scores never select among intermediate candidates. Export one selected receipt per group into the existing GRACE scorer contract, with source-group hash and selected round, keeping 700 as the full formal denominator.

## Full downstream evaluation

Use the existing frozen GRACE-3L-OAM-L asset, relaxation settings, MP2020 corrections and fixed MP2023 hull reference through `structure_request_eval_20261017/grace_score.py`. Report this as the established GRACE ehull proxy, not a DFT result. Retain all score failures and per-material atomic receipts for resume.

Report structural success within1/2/3 attempts, natural switches, error recurrence, cost and paired policy outcomes. Separately report property scoring, post-relax validity and requested-space-group preservation, ehull ≤80meV/atom, uniqueness and novelty against the fixed own-training reference. Use the same existing additional thresholds only as supplementary measurements. Deduplication occurs within each policy; a shared candidate appearing in both policies is not cross-policy duplication to remove.

The main comparison is feedback versus independent retries for the same repair-trained model and generation budget. Earlier parent-model pilot results are historical context, not a matched700-group parent comparison.

## Execution

Use the existing project mamba environments and `/home/liugang/src/crystalDLM` symlink into `/data1/users/liugang/crystalDLM`. Check mount, disk and GPU occupancy at launch. Use three free H20 GPUs, one generation shard per GPU, explicit CUDA_VISIBLE_DEVICES. After generation workers exit, run both independent and feedback GRACE jobs on separate GPUs; project-wide usage stays at most three.

Run CPU contracts and a separate seven-group smoke to check checkpoint restoration, fresh feedback context, sampler parity, stopping, receipt integrity, resume, scoring export and GRACE integration. The smoke gate is engineering correctness; low structural success alone does not block the authorized formal evaluation. Formal outputs go to a new `run_001` directory with a named tmux supervisor, stage/exit files and logs. No previous artifacts are overwritten.
