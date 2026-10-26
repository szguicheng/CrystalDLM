# Q2 coverage repair

This is a separate fresh-initialization run to give the reader slot Q2 exposure. The completed pilot 002 (Slurm job 605522) and its receipts are immutable. The prior `q1_only` schedule remains the runner default; its event hash is pinned here as `79e62e1a7e4f3c7e92ef5d4354fbd907d4c3ba90076924300861c43bb04f07f4`.

## Frozen inputs

- Representation: `relational_cnl_v1`; keep the model, codec, scorer, and raw records unchanged.
- Cohort: frozen first 8 `train/seen8` cases and next 16 `heldout16`; records SHA-256 `ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f`.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; fresh seed-17 initialization, expected initial fingerprint `6bffa202d8485a6ee8bce3553044c9657948e25be702a45a9efb860ced9d1429`.
- Packing: `K=2560`, `STEPS=32`, `MAXLEN=8192`; no truncation.

## Schedule and budget

Run 512 updates with 8 updates per epoch and effective batch 4 (`microbatch=1`, accumulation 4). Keep the existing generation/edit slots and optimizer unchanged: `G_j`, `G_(j+4)%8`, `E_j`, then one reader task; AdamW at learning rate `2e-5`, weight decay `0.01`, 40-update warmup followed by cosine decay, gradient clipping at 1.0, FP32 parameters with BF16 autocast.

With `--reader-schedule alternating_q1_q2`, even-numbered epochs use Q1 and odd-numbered epochs use Q2. For Q2, choose its edit factor independently as `minus` when `(epoch//2 + j) % 2 == 0`, otherwise `plus`. This gives each material `G=128`, `E=64`, `Q1=32`, `Q2=32`; E has 32 examples per factor and Q2 has 16 per factor. Totals are `G=1024`, `E=512`, `Q1=256`, `Q2=256`. The ordered Q2 schedule event hash is `d9c1f0576a30c62d81a9e00bdcf84cd13412057012b85da34d8c154f8424f560`.

Q2 must use the complete C1 history and source answer, the edit instruction, and the selected C2 text in its prefix; its target is the frozen C2 answer. `pack_audit` checks all Q2 examples, both factors, train variants, and heldout cases against `MAXLEN`, and checks generated crystal blocks against `K`.

## Receipts and gates

Use a new output directory, `q2_coverage_001` (relative to the experiment root); do not resume either earlier pilot. `training_exposure.json` records the full scheduled event list and per-material factors. Each arm's `status.json` records the schedule, event hash, actual exposure counts including Q2 factors, optimizer/update budget, and fingerprints. `checkpoint_receipt.json` records the schedule and event hash with the final checkpoint hash and verified reload.

`gpu_preflight.json` is the disposable probe: one effective-batch optimizer step per arm, excluded from the 512 updates. For the alternating schedule its reader candidate search includes Q1 and Q2, and the receipt records the selected reader task. Training proceeds only on one A100 80 GB when both probes stay at or below 70 GiB peak reserved memory. The regular seen8/heldout16 evaluation remains the learned-model result; the preflight is not an experiment result.

Offline validation is `test_q2_schedule.py` plus `python run.py --check-pack --reader-schedule alternating_q1_q2` in the existing `.venv`. No GPU job has been submitted for this repair. Training requires a separate root acceptance and independent review.

`resume_pilot_001.py` is not an entrypoint for this Q2 run. Its current copy is retained as predecessor-maintenance code and is outside this schedule design.
