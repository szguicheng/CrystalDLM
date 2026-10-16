# `pilot_002` denoising diagnostic

## Status

Prepared for review only. It has not been submitted with `sbatch`. The job reads the `pilot_002/hybrid_shift` final checkpoint and the eight seen training rows, performs no optimizer steps, and refuses to overwrite `diagnostics/diagnostic001`.

## Pinned run

- Checkpoint: `pilot_002/hybrid_shift/checkpoint.pt`, SHA-256 `3e1b0e30…0cdf25a2`, final fingerprint `01f8ce73…161aabd8`.
- Reload: strict state load, `mode=hybrid`, `prediction_offset=1`.
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Device guard: one Slurm CUDA allocation with an A100 of at least 79 GiB; Slurm also sets the existing `interleaved_ar_diffusion_20261002/env` and offline `hf_cache`.
- Data guard: compare raw manifest and records SHA-256 to the saved Hybrid status. The probe parses only the first eight `train` JSONL rows; it does not parse or evaluate heldout, test, or excluded cases.
- Core guard: current `model.py`, `prepare.py`, and `score.py` hashes must match the Hybrid receipt. Sampling uses the frozen `q2_coverage_preparation/source_snapshot/run.py` helper whose hash matches the receipt; the later top-level `run.py` is recorded but not used.

## Measurements

For each seen ID, run the existing Hybrid trigger and crystal sampler once with the original direction rule and CUDA seed `sha256((id + "20261004").encode()).hexdigest()[:8]`, `temperature=0.8`, `STEPS=32`, and the unchanged reverse-release path. An observer wraps `model.unpack_block`: it copies the complete input IDs and END/PAD positions, calls the original function with the original arguments, then returns its result or re-raises its exception unchanged. This captures the new diagnostic sample if validation fails; it does not recover a missing pilot output or turn an invalid block into a valid one.

For the same eight G training targets, use each fixed example's gold prefix and lead with the existing `Hybrid.pack` once at each fixed noise step `k=32,16,1`, with independent recorded CPU mask seeds. Score masked body, true END, and PAD labels separately for top-1 accuracy and mean target NLL; retain target/top-1 IDs and positions, class mean probabilities, END argmax positions/count among masked positions, and strict boundary validity of the teacher-forced top-1 block. At `k=16` or `k=1`, the random mask may contain no END position; those metrics must be `count=0` and JSON `null`. Do not force-mask END. The oracle body length is used only to group teacher-forced positions and never enters free sampling or padding decisions.

Teacher-forced metrics use raw full-vocabulary logits from the existing BF16 autocast context; they do not apply the free sampler's special-token ban. Free generation uses the original sampler filters. The two metric sets have different prediction conditions and should not be read as directly comparable accuracy rates.

The training code gave G/E examples no explicit `noise_step`; `model.pack` therefore drew uniformly from 1..32. Full-mask had theoretical probability 1/32 per example, but the saved curve has no per-example mask log, so the realized training count is unknown. Probe mask conditions are diagnostic settings, not a change to the training or planned Q2 comparison schedule.

The output is `diagnostics/diagnostic001/{summary.json,results.jsonl}`. The JSONL stores one raw token-ID block per G attempt, free and teacher-forced prefix/lead/block token lengths, and teacher-forced top-1 IDs; it never stores the K-by-vocabulary logits. The summary records GPU, Slurm job, checkpoint/data/core hashes, mask seeds, and before/after model fingerprint. A mismatch or incomplete checkpoint/data pin aborts before sampling.

## Offline check and submission

On the local machine, run only the dependency-free hook check and syntax check; neither loads the real checkpoint:

```bash
python3 experiments/interleaved_relational_20261004/diagnostics/probe_denoising.py --self-check
python3 -m py_compile experiments/interleaved_relational_20261004/diagnostics/probe_denoising.py
```

After independent code review, submit from the CrystalDLM project root with:

```bash
sbatch experiments/interleaved_relational_20261004/diagnostics/probe_denoising.slurm
```

This protocol records the command; this preparation turn does not submit it.
