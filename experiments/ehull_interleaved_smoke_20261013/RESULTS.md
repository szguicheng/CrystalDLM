# Shared-model technical preflight results

Slurm job **626135**, `gpu_a100`, node `gpu-a100-07`, A100-SXM4-80GB: **COMPLETED**, exit **0:0**, elapsed **00:02:38**. One GPU was allocated; the checkpoint remains on Burgundy. Small receipts and logs are synchronized locally.

## Observed runtime results

- Actual 1.5B tokenizer checked all 8,172 records. Crystal body including END maximum531, generation sequence maximum804, reading sequence maximum814; no truncation, K544/MAXLEN2048.
- Four tiny-model tests passed: exact property-condition/task construction, future-question isolation, actual prefix-logit isolation and finite objectives/gradients.
- Twenty updates completed using twenty training materials, one generation/reading pair per update. This is a technical preflight rather than an epoch over the 7,272 training materials.
- Text and crystal losses and all gradients remained finite. Final text loss0.6671, crystal loss3.2811, total2.3077. Different updates use different materials/noise masks; these endpoints are not a held-out convergence measurement.
- Optimizer loop8.213 seconds; peak allocated GPU memory29,458 MiB (28.77 GiB). End-to-end Slurm time includes startup, checkpoint save/reload and evaluation.
- The checkpoint's weights and model configuration matched exactly after reload. Saved checkpoint size6,173,560,702 bytes.

## Raw generation outcomes

Three validation attempts were recorded. All three failed to produce `CRYSTAL_START` in the AR lead. There were **zero crystal-sampling attempts and zero continuation attempts**. Thus this run establishes that the shared training objectives, data interfaces and checkpoint cycle execute; it does not establish successful interleaved generation or property control.

The next learning test should train on the fixed subset for substantially more updates, track the AR trigger separately from diffusion-boundary/parser validity, and continue AR only from the actual generated crystal. Preserve all failures. The complete pure-AR comparison and external property scoring remain on the existing scientific decision tickets.

Evidence: `run_001/summary.json`, `run_001/updates.jsonl`, `run_001/evaluation.jsonl`, `logs/smoke-626135.{out,err}`, `input-manifest.json`, `submission.json`.

Large checkpoint location: `/scratch/chenggui2/CrystalDLM/experiments/ehull_interleaved_smoke_20261013/run_001/checkpoint.pt`.

## Sequencing correction

The agreed scientific order is matched pure AR first, then shared AR+diffusion. This hybrid engineering run was executed before the current-data pure-AR baseline; that baseline has not been performed. Keep this receipt as an out-of-sequence engineering diagnostic, not as completion of either the pure-AR baseline or the formal joint experiment. The next learning run must be pure AR on the frozen subset.
