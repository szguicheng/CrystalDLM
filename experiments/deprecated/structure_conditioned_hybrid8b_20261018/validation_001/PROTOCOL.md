# Bounded validation of hybrid generation failure

Authorized 2026-10-18 after the root-cause report. Historical code, data, checkpoint and benchmark outputs remain unchanged. No new tokenizer, 700-attempt benchmark, or GRACE scoring in this run.

## Frozen inputs and endpoints

`plan.json` contains the first four attempt IDs in each of the seven existing space-group requests: 28 candidates. Selection precedes intervention outputs. Keep requested SG and ehull <= 0.08 eV/atom, and reuse exact original-model lead token IDs across every condition. Thus this isolates crystal generation; it does not evaluate newly trained text generation end to end.

All conditions restart the crystal CUDA RNG from the recorded attempt seed. The original sampler is rerun because historical full-trajectory sampling consumed RNG while generating the lead. Historical outputs cannot substitute for that new RNG contract.

Report raw strict-valid / 28, boundary-pass / 28, valid-and-requested-SG / 28, unrepaired failure types and generation seconds. No post-generation END/PAD repair. Failures remain in the denominator. No property success claim without independent property evaluation.

## Sampling experiment: same frozen checkpoint

1. **original:** existing 32-step random irreversible release, temperature 0.8, top-p 0.95.
2. **confidence:** same model, temperature/top-p and 32 forwards. Rank sampled tokens by their probability under the temperature-scaled, special-token-filtered distribution before top-p renormalization. Commit the most confident positions to a linear known-token count `floor(544 * step / 32)`. Committed positions stay fixed.
3. **remask:** same confidence policy and known-token count, but before each forward after the first, remask the lowest stored-confidence `ceil(0.1 * known_count)` committed positions. Those positions can be predicted again. Both confidence policies set timestep to `round(32 * actual_mask_count / 544)`, clipped to 1..32, so the remasked input is conditioned on its actual corruption level.

Original vs confidence measures the confidence/schedule policy jointly. Confidence vs remask measures the added revisiting policy, including the corresponding corruption conditioning. This is a bounded heuristic probe, not an exact posterior sampler or a direct reproduction of a named paper method. It cannot isolate every internal component with only these conditions.

## Training experiment: matched continuation budget

Both arms start from the same completed 909-update source checkpoint. Each uses the same first 1,024 materials of the frozen training epoch order, 128 updates, 8 materials/update, unchanged G+Q targets and loss weights, fresh AdamW state, constant LR 5e-5, weight decay 0.01, gradient clipping 1, BF16 base and existing trainable parameters. Each arm uses one H20. The user explicitly authorized up to three GPUs simultaneously on 2026-10-18, allowing source-checkpoint evaluation to overlap both training arms; this changes scheduling only.

- **control:** original uniform noise timestep k in 1..32.
- **highmask:** probability 0.5 selects k=32; otherwise uses the same uniform draw as control. Hence expected full-mask probability is 51.5625% vs 3.125%. Mask RNG is separate and matched. Existing 32/k token weighting remains unchanged; the objective intentionally changes noise-level weighting, without inverse-proposal correction.

Both are evaluated using the **original sampler** and the same 28 frozen prefixes. Comparing highmask against control isolates the noise distribution from extra training time. Comparing either continuation with source also reflects additional training. No source optimizer/history is overwritten; independent resumable checkpoints and update logs are saved inside the project.

## Interpretation and stop rules

Five conditions total: source-original, source-confidence, source-remask, control-original, highmask-original. Training stops after 128 updates per arm. Generation stops after 28 per condition, even if quality is poor; this small fixed comparison is the authorized diagnostic. Finite-loss/gradient failure or identity mismatch stops the affected run and is investigated.

A gain of at least three valid candidates over the direct comparator is an exploratory signal for a larger independent confirmation, not proof of improvement and not automatic authorization to run 700. Zero validity in both controls means this intervention/budget was insufficient; it does not prove the hypothesized mechanism irrelevant. One training seed and 28 requests cannot establish a final method ranking.

## Execution provenance

Source-checkpoint evaluation uses the implementation preserved as `source_evaluation_snapshot.py`; its SHA matches the `evaluate.py` hash in those receipts. Before continuation evaluation, the wrapper added an assertion that checkpoint metadata matches the requested arm and hashes for dynamically loaded evaluation dependencies. Sampling, inputs, parser, and metrics are unchanged. All four remote CPU tests passed. Each training arm first completed two updates, verified changed trainable weights and finite nonzero gradients, then restored optimizer and RNG state to continue the same 128-update run.
