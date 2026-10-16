# Clean crystal AR anchor causal repair

The active user goal authorizes root-cause investigation and repair. This run tests one confirmed implementation difference from FLARE: the completed pilot's clean crystal stream was not supervised by next-token prediction. Causality remains unproven until this matched intervention runs.

## Invariants

Freeze `relational_cnl_v1`, its decoder/scorer, all original numerical precision and corrections, the same eight training and sixteen heldout IDs, the original native tokenizer, and byte-identical original `data_pilot_001` records SHA-256 `ef2ec2dcdee1d73c62e48987a9cec86a4db3076f9c48d9ab17b8745b9eca785f`. Do not use the heldout structures for training or checkpoint selection.

Fresh initialization: Qwen/Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, seed 17, expected fingerprint `e97aed646b7289c06f10be376b37d67cd7aa6a42ad6214b61da008cd85b8404b`. FP32 parameters, BF16 autocast and gradient checkpointing. B=32, K=2560, STEPS=32, MAXLEN=8192, prediction_offset=0.

The frozen baseline is completed `interleaved_relational_blocks_20261005/pilot_001/diffusion_b32`. Its model/source/data artifacts remain immutable. The baseline and this repair both train 512 updates with effective batch 4, microbatch 1, accumulation 4, AdamW learning rate 2e-5, weight decay .01, 40-update warmup then cosine decay, and gradient norm clipping 1.0. Keep G,G,E,Q1/Q2 ordering, task exposure hash `d9c1f0576a30c62d81a9e00bdcf84cd13412057012b85da34d8c154f8424f560`, uniform k in 1..32, Bernoulli mask probability k/32 and original inverse-mask weighting. Record realized noise steps; do not increase full-mask exposure in this intervention.

## Single intervention

On hybrid crystal examples, add causal next-token supervision of the same clean crystal body plus END with weight 1. The clean queries are the positions immediately preceding their target tokens; causal clean attention remains isolated from the noisy stream. Do not supervise PAD in this auxiliary AR term: AR terminates at END, while the existing diffusion objective still learns the complete END/PAD boundary with PAD weight .1. Normalize the auxiliary term by body-plus-END token count and retain the original noisy whole-target normalization. Both terms use the same shared backbone/LM head in one physical forward; no extra model or AR fallback.

Generation of crystal bodies and edits remains diffusion using the original temperature .8 and random absorbing release. Text, triggers and readers remain AR. Thus a successful result cannot be attributed to replacing crystal diffusion with AR. Log the clean-crystal AR contribution separately from noisy body/END/PAD metrics and check checkpoint metadata stores the anchor weight.

## Gates and evidence

Run compact checks for clean query alignment, causal/noisy information isolation, exact added loss, unchanged noisy labels/weights/RNG, one-forward packing and anchor-zero compatibility. Reuse the earlier source/data pin and actual pack-length checks. Independent review must close correctness findings before submission.

On an allocated A100 80GB, measure a disposable worst-case full-mask effective-batch optimizer step including the longest G/E and Q2; require finite gradients, expected fresh fingerprint and peak reserved memory <=70GiB. Exclude that step and recreate the pristine model for the 512 updates.

After training, verify checkpoint reload and actual scheduled exposures, run fixed-mask diagnostics and the full original seen8 G/read/edit/read/swap evaluation. If seen8 G is zero, retain the complete seen result and mark heldout not evaluated due to the mechanism gate. Never count this as 0/16 or as evidence of heldout failure. If seen8 G is positive, evaluate all sixteen heldout once with the original strict scorer and no retries/repair. All failed cases remain in the denominator.

Report actual generation, physical validation, exact edit and actual-structure grounding results separately. Prefix success, low loss, native preflight and a memorized loop do not satisfy the overall user goal.

Source basis: FLARE https://arxiv.org/html/2606.01774v2 sections 3.2 and 4.2. This is an isolated adaptation of the clean AR anchor, not a complete reproduction of FLARE's complementary masking, shifted logits or hybrid recurrent backbone.
