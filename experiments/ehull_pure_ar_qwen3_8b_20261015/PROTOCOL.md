# Qwen3-8B pure-AR LoRA experiment

This is the approved larger-base pure-AR run for the frozen ehull–description–PLaID task. It reuses the 7,272/900 triples, seed-42 order, 32 fixed validation IDs, task builder, strict PLaID parser, prompt strings, and greedy-text/temperature-0.8 crystal sampling from `ehull_pure_ar_baseline_20261013`. The custom `User:` / `Assistant:` stream is retained; no HF chat template, reasoning filter, output repair, or truncation is applied. Generation stops on the existing tokenizer EOS token `<|im_end|>` only, even though the model generation config may list a secondary EOS ID. Raw `<think>` output, if emitted, remains in the evaluation receipt.

The fixed base is Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. The runtime reads a project-local model directory with `--model-path` and does not contact the Hub. Frozen base weights use BF16; LoRA and selected-token weights use FP32 with AdamW. LoRA uses rank 32, alpha 64, dropout 0, and targets Qwen3 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head`. PEFT 0.18.0 `trainable_token_indices` trains the four added input embedding rows without saving either full vocabulary matrix. The shared model’s zero-initialized `time.weight` remains trainable as a 33-by-hidden-size FP32 auxiliary parameter; it is included in the trainable-state hash and checkpoint. `get_peft_model(...).get_base_model()` preserves the existing `base.model` and `base.lm_head` access paths.

One epoch covers all 7,272 materials once. Each global update consumes eight consecutive materials (two G/Q tasks each), for 16 effective task examples and 909 updates. On two ranks, each rank accumulates four materials and DDP averages the gradients, preserving the global batch and update schedule. AdamW uses LR 2e-4, weight decay 0.01, 5% linear warmup followed by cosine decay, and gradient clipping at 1.0. Gradient checkpointing uses non-reentrant checkpointing. Checkpoints atomically replace one bounded `checkpoint.pt` containing trainable adapter/token parameters, optimizer state, per-rank RNG state, and next material index; the frozen 8B weights are never copied into a checkpoint or CPU-hashed.

The run identity combines the pinned revision, streaming SHA-256 fingerprints of the local model files, the model config, frozen data hashes, and source hashes. Trainable state receives its own tensor hash. This replaces the prior full-parameter CPU fingerprint while keeping the base identity auditable.

## Commands

Run from the project root with the isolated project environment and an explicit pair of free GPUs in `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 experiments/ehull_pure_ar_qwen3_8b_20261015/train.py \
  --mode preflight \
  --model-path /home/liugang/src/crystalDLM/models/Qwen3-8B-b968826d9c46 \
  --data research/crystal_training_preflight_20261013/data_001 \
  --out experiments/ehull_pure_ar_qwen3_8b_20261015/preflight_001
```

Preflight checks the longest real G/Q example, finite forward/backward and nonzero gradients for the input-token and output-head paths, performs at most three optimizer updates, then saves and reloads the small trainable state and verifies identical control logits. Formal training requires that preflight `summary.json` as its receipt:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 experiments/ehull_pure_ar_qwen3_8b_20261015/train.py \
  --mode train \
  --model-path /home/liugang/src/crystalDLM/models/Qwen3-8B-b968826d9c46 \
  --data research/crystal_training_preflight_20261013/data_001 \
  --out experiments/ehull_pure_ar_qwen3_8b_20261015/run_001 \
  --preflight-receipt experiments/ehull_pure_ar_qwen3_8b_20261015/preflight_001/summary.json
```

After training exits and releases both GPUs, run the fixed 32-sample evaluation on one selected free GPU:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 experiments/ehull_pure_ar_qwen3_8b_20261015/train.py \
  --mode evaluate \
  --model-path /home/liugang/src/crystalDLM/models/Qwen3-8B-b968826d9c46 \
  --data research/crystal_training_preflight_20261013/data_001 \
  --checkpoint experiments/ehull_pure_ar_qwen3_8b_20261015/run_001/checkpoint.pt \
  --out experiments/ehull_pure_ar_qwen3_8b_20261015/evaluation_001
```

To resume, pass `--resume experiments/ehull_pure_ar_qwen3_8b_20261015/run_001/checkpoint.pt` with the same output directory and arguments. Training writes 50-update checkpoints and update receipts; a stop-after-update invocation can be resumed from the latest checkpoint.

The preflight receipt covers engineering behavior only. Formal results separate objective/gradient health, trigger and crystal-boundary outcomes, strict structural validity, factual text/structure agreement, and generated-crystal ehull. No generated ehull scorer is inferred from the input target or model text. The 1.5B comparator also differs in full fine-tuning versus LoRA, base dtype, optimizer LR, and model family; this run tests the authorized stronger-base setup and is not a model-size-only ablation.

## Dependency and source references

The pinned PEFT API documents selective token training with `trainable_token_indices`, its use alongside LoRA, and in-place adapter injection. See the [PEFT 0.18.0 LoRA API](https://huggingface.co/docs/peft/v0.18.0/package_reference/lora), [trainable-token API](https://huggingface.co/docs/peft/v0.18.0/package_reference/trainable_tokens), and [PEFT model API](https://huggingface.co/docs/peft/v0.18.0/package_reference/peft_model).
