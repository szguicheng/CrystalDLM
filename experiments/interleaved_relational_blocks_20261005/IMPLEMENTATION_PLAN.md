# Relational crystal block diffusion implementation plan

User approved this design in the current chat on 2026-10-05. Execution is authorized, including a fresh four-arm Burgundy pilot after offline checks, independent review, and GPU preflight. This checkout has no Git repository; use a new experiment directory and retain predecessor files and outputs.

Goal: test complete relational-CNL generation with causal clean prefixes and diffusion blocks of 2560, 32, or 64 tokens against pure AR.

Architecture: one pinned Qwen2.5-1.5B-Instruct backbone per fresh arm; token-causal clean stream and block-bidirectional noisy stream during diffusion training. Sample blocks sequentially and validate the complete K-token result once. Reuse the frozen codec, scorer, 8 training/16 heldout records, and tokenizer.

## Contracts

- K=2560 capacity, B in {2560,32,64}, STEPS=32, MAXLEN=8192.
- Model revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; seed 17.
- Every diffusion arm uses prediction_offset=0; AR predicts the next token.
- Clean/noisy copies reuse logical position IDs, and sequential generation marks all preceding crystal slots (including PAD after END) at clean timestep zero. Verify numerical training/inference view agreement.
- All K targets retain body/END weight 1 and PAD weight 0.1, one shared noise step per example, existing inverse-mask weighting, and whole-target normalization.
- Four arms: ar, diffusion_full, diffusion_b32, diffusion_b64; fresh initialization, 512 updates each, effective batch 4, Q1/Q2 alternating by epoch.
- No truncation, target-length input, early END stop, repair, retry, or relaxed scoring. All diffusion samples fill K and pass one global END/PAD validator.
- Enable gradient checkpointing uniformly; measure one disposable optimizer step per arm on an allocated A100 80GB; peak reserved memory must be <=70GiB.
- Old data provenance identifies the original 0.5B adapter/tokenizer; new model and training provenance identify 1.5B. Do not rewrite the old manifest.

## Tasks and ownership

- [ ] Model worker owns model.py and test_block_model.py. Interface: Hybrid.pretrained(tok, prediction_offset=0, block_size=K). A diffusion pack has prefix+lead+clean K+noisy K; its mask permits each noisy block only its own noisy tokens, the prompt, and clean target indices strictly before the block. Tests must reject direct/indirect gold leakage, changed loss normalization, and per-block END validation.
- [ ] Runner worker owns run.py and test_block_runner.py. Reuse frozen tasks and grading; enumerate all four arms in preflight, training, reload and evaluation; record actual dual-stream lengths, Q2 exposures, target alignment, fingerprints, timings and calls. Verify original adapter hashes using their original files.
- [ ] Primary owns prepare.py, score.py (unchanged copy), protocol, Slurm launch, integration and final acceptance. Freeze old-source hashes; verify 1.5B revision and tokenizer compatibility; do not run heavy work on login nodes.
- [ ] Run compact CPU checks in the existing local venv; use a small randomly initialized transformer for numerical leakage checks. Run pack/source-pin checks and bash syntax checks.
- [ ] Fresh independent reviewer audits mask information flow, label placement, normalization, sampling termination and experiment comparability. Primary verifies critical findings and repairs them.
- [ ] Sync only the new experiment to /scratch/chenggui2/CrystalDLM; validate remote hashes and dependencies; acquire one Slurm allocation. Preflight must pass before the bounded pilot starts.
- [ ] Read structured run receipts and raw results. Distinguish implementation checks, GPU preflight, job submission, training completion and actual generation outcomes.

## Verification commands

```bash
rtk proxy experiments/interleaved_ar_diffusion_20261002/.venv/bin/python experiments/interleaved_relational_blocks_20261005/test_block_model.py
rtk proxy experiments/interleaved_ar_diffusion_20261002/.venv/bin/python experiments/interleaved_relational_blocks_20261005/test_block_runner.py
rtk proxy experiments/interleaved_ar_diffusion_20261002/.venv/bin/python experiments/interleaved_relational_blocks_20261005/run.py --check-pack
rtk proxy bash -n experiments/interleaved_relational_blocks_20261005/pilot_001.slurm
```

The new source snapshot and output directory must be frozen before submission. PAD reduction, semantic blocks and confidence-based release are later experiments and are not implemented here.
