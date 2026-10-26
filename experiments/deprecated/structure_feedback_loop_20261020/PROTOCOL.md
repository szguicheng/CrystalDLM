# Structure-feedback loop pilot

Authorized on 2026-10-20: compare independent resampling with structure-feedback retries using the existing unified AR/diffusion model. GRACE is excluded from feedback. No training or checkpoint changes.

## Frozen comparison

- Checkpoint: `experiments/flare_crystal_full_20261019/flare/checkpoint.pt` (completed 909-update FLARE arm). Both policies use the identical restored state.
- Requests: the existing seven space groups 1, 15, 38, 119, 143, 194, 216; ehull request remains ≤0.08 eV/atom. No reference structure or reference description is supplied.
- Pilot: 10 seeded groups per space group, 70 paired groups. At most three candidate attempts per policy per group, including the initial candidate. Stop each policy on its first structural success.
- Initial seed and prompt: existing `attempt_plan(limit=10)`. Generate the first candidate once and share it between policies. Retry seeds are derived deterministically from the group's seed and round and paired between policies. Their RNG draws need not stay aligned after different text lengths.
- Independent policy: each retry starts from the original request only.
- Feedback policy: each retry uses the original request, the latest candidate's generated lead and crystal tokens, and the checker feedback. Older attempts remain in the saved trajectory but are not accumulated in the model context. The original target cannot be relaxed by model prose.
- Sampling: AR lead with temperature 0.8/top-p 0.95, at most 256 tokens; crystal generation with the existing B=4 diffusion sampler, at most 544 tokens including END. No forced crystal START, candidate repair, or extra uncounted quality retries.
- Inference context budget: 4096 tokens, checked against the underlying Qwen context capacity. The larger budget accommodates the preceding candidate; training weights and training sequence lengths are unchanged. Context overflow is recorded as a failed attempt, not silently truncated.

## Feedback and structural success

Success requires both the existing frozen strict structural audit and agreement between the decoded space group and the original requested space group. Formula/expanded orbit counts, declared sites, lattice symmetry, Wyckoff constraints, overlap, and decoded symmetry retain the existing definitions.

Feedback is short English generated deterministically from saved checker facts. It reports boundary/parse errors, strict audit reason codes, and requested versus detected space group where available. It does not invent a physical cause or include a reference answer. ehull is explicitly unevaluated.

Incomplete crystal spans are closed only when serializing them as previous context, so the next user feedback is treated as text. Such synthetic context delimiters are recorded separately; raw candidate tokens and the failed boundary decision are unchanged. Both AR and diffusion must assign crystal time embeddings to prior crystal spans and ordinary text embeddings to the feedback/new lead.

The diffusion forward path reuses the frozen shifted-block mask and position scheme. Its local extension handles earlier crystal spans. On a single-crystal prefix it must match the existing denoising logits; it must not expose hidden current-block targets.

## Measurements

- Main: groups successful within 1, 2, and 3 attempts, always divided by all 70 scheduled groups per policy.
- Paired comparison: both succeed, only independent succeeds, only feedback succeeds, neither succeeds.
- Cost: mean attempts per completed group and time to success or budget exhaustion. Shared first-generation cost is charged logically to both policies; actual executed candidate count is reported separately.
- Failures: counts by checker failure category and consecutive-attempt error recurrence; distinguish no START, no END, strict-invalid, and valid-but-wrong requested space group.
- Report per-space-group results and retain every failed attempt. Structural success is not evidence of ehull success. No GRACE, DFT, novelty matching, or training is scheduled in this pilot.

## Execution and persistence

Use the project mamba environment on LiuGangGPU. One explicitly selected GPU per process; two shards on idle GPUs 0 and 1, all project tasks remain below three GPUs. All files remain under `/home/liugang/src/crystalDLM`.

Atomic per-attempt receipts and terminal group receipts support restart without regenerating completed work. A resumed run verifies frozen checkpoint/code/request identities. Operational errors stop that worker with a visible nonzero exit; model failures remain evaluation outcomes. No prior experiment artifacts are overwritten.

Run CPU contract tests and a seven-group smoke pass in a separate directory before the 70-group pilot. The smoke gate checks engineering integrity and checkpoint/sampler parity, not a minimum success rate. An all-invalid but correctly executed feedback policy is an experimental finding.

## Implementation plan

Spec: this protocol and the user's approved five-step plan in the conversation.

1. Implement `run.py` and `test_run.py` in this isolated experiment directory, reusing the existing model loader, AR sampler, diffusion sampling rule, and strict auditor. Tests cover seed pairing, stopping, context boundaries, target preservation and denoising parity.
2. Implement `summarize.py`: recount terminal group and attempt receipts, enforce complete paired identities, calculate the metrics above, and write `summary.json` plus `RESULTS.md`. Test a small synthetic paired example where one policy fails all attempts.
3. Sync only this experiment directory; run tests in the existing mamba environment. Launch the smoke in named tmux sessions; inspect actual failure feedback and saved next-round context.
4. Launch two formal shards with explicit GPU assignments and separate logs, automatically summarize after both workers exit successfully. Verify that first candidates and retry seeds match, then report live state or final measured results.
