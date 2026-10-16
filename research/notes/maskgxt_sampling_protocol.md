# MaskGXT-derived multi-step protocol for conditional Wyckoff count blocks

## Scope

This note reads the local full text of `maskgxt-2606.22866.txt` (main text and appendices A–C/D where sampling and evaluation are specified) and the public `kiyoung98/maskgxt` `sample.py` implementation. It extracts a minimal reusable masked-parallel decoding protocol for the current model that is conditioned on a space group and composition and predicts Wyckoff count blocks.

MaskGXT is not evidence that the proposed count-block tokenizer works. Its paper uses a different tokenization, full crystal output, and composition-conditioned CSP task. The value here is a sampling/evaluation design that can expose whether a block tokenizer only helps one-step MAP recovery or also helps iterative conditional generation.

## Exact MaskGXT source locations

### Representation and conditioning

- §4, §4.1, PDF pp. 5–6: MaskGXT represents a crystal as `1 + 6 + 5N` discrete tokens: one space-group token, six lattice tokens, and five per-site tokens (three coordinate bins, a Wyckoff token and an atom-type token). Atom types/composition are side information; SG, lattice, coordinates and Wyckoff tokens are predicted.
- Fig. 3(c), PDF p. 6: starts from a fully masked sequence with atom types known; SG posterior seeds branches, then remaining tokens are greedily unmasked over `T` steps.
- §4.2–§4.3, PDF pp. 6–7: origin-shift and orbit-permutation augmentations; independent random masking with `t ~ Uniform(0,1)`; masked cross-entropy on the masked positions.
- Appendix A, Algorithm 1, PDF p. 16: training pseudocode confirms random masking and prediction of all masked positions in parallel.

For the current model, the direct adaptation is simpler: leave `G` and composition/count condition fields visible, mask only the target count-block sequence, and predict masked blocks. Do not add a generated SG branch unless SG is intentionally made a target.

### Iterative sampling and remasking

- §4.5, PDF pp. 9–10: confidence-ranked greedy decoding starts with all non-condition tokens masked, predicts every masked stream in parallel, fills the highest-confidence positions, and re-masks the rest according to a schedule.
- Appendix A, Algorithm 2, PDF p. 16: at each step, sample or argmax a token at every masked position, score the chosen token by posterior confidence, keep the highest-confidence positions fixed, and remask the remainder. A final pass resolves any residual masks.
- Appendix B, Table 6, PDF p. 17: the reported full MaskGXT configuration uses `T=150`; this is a full-crystal setting and should not be copied as a mandatory block-model step count.
- Official `sample.py`: `sample_batch` implements the same logic. It predicts all streams in parallel, obtains multinomial samples (or argmax for greedy rows), scores chosen-token log probabilities, adds optional decaying Gumbel confidence noise for stochastic chains, and uses a cosine retention schedule. The code computes `gamma_next = cos(0.5*pi*(1 - t_next))`, where `t_next = 1 - (step+1)/T`; the number of positions left masked is `floor(gamma_next * total_positions)`. On the final step it forces zero masked positions, then runs a final argmax pass if any residual masks remain.

The proposal should use the same conceptual loop but report the exact schedule, because “multi-step masked decoding” is under-specified without a remask policy.

### Conditional diversity and quality evaluation

- §4.5 and Appendix A, Algorithm 3, PDF pp. 9–10 and 16: for polymorph coverage, MaskGXT first reads `p(g|A)` from one fully masked forward pass, selects top-S space groups, fixes one SG per branch, and decodes the remaining tokens independently. A low posterior-mass fallback leaves SG free and samples i.i.d.
- §5.2, PDF p. 9: standard one-to-one CSP uses StructureMatcher with `ltol=0.3`, `stol=0.5`, `angle_tol=10°`; reports match rate and RMSE, both unfiltered and validity-filtered.
- §5.5, Tables 2–3, PDF pp. 11–12: polymorph-aware evaluation uses METRe and corrected RMSE (cRMSE), both a reference-coverage metric and a geometric-precision metric. Table 3 fixes a uniform budget `S=2` per composition when the number of polymorphs is unknown; Appendix C sweeps `S=1…5`.
- §5.6, Table 5, PDF pp. 12 and 15: removing SG stratification causes the largest METRe drop on the polymorph split; removing greedy decoding lowers METRe and worsens cRMSE. These are ablations of MaskGXT’s sampler, not validation of any other tokenizer.

## Minimal reusable protocol for the current count-block model

### Protocol A: deterministic one-step baseline

1. For each held-out `(G, composition)` context, provide exactly the same visible condition fields used by training.
2. Mask every target count-block position.
3. Preserve the actual existing baseline: score each complete canonical candidate by the sum of its word log probabilities, then use the same exact fixed-site-exclusion joint MAP reader. This is not equivalent to independent token argmax followed by an unspecified repair.
4. Record exact block recovery, exact expanded `(WP,count)` recovery, composition conservation, legality, and candidate rank/top-k recovery.

This preserves the existing all-mask MAP measurement as a baseline. It is not a generative sampling result.

### Protocol B: confidence-ranked iterative decoding

Use a small step sweep appropriate to the short block sequence, for example `T ∈ {1, 4, 8, 16, 32}`. The paper’s `T=150` is a reference configuration for a much longer full-crystal token sequence, not a required value.

For each `T`:

1. Keep `G` and composition visible; mask all target blocks.
2. At step `t`, predict all masked block positions in parallel.
3. Deterministic branch: select each position’s MAP block. Stochastic branch: sample from the posterior at temperature 1.0 (or a pre-registered temperature grid). In both cases store the chosen-token log probability or posterior confidence.
4. Rank only currently masked positions by confidence. Fix the highest-confidence positions and remask the remainder using the pre-registered retention schedule. With zero-based step k, the decreasing schedule is `n_masked(k+1)=floor(N_target*cos(0.5*pi*(k+1)/T))`, with zero masks forced on the final step. This follows by substituting `t_next=1-(k+1)/T` into the official code expression; the remaining-mask count must decrease, not increase.
5. Repeat until all blocks are fixed. Run a final argmax repair pass if any mask remains.
6. Decode the final blocks to the same legal candidate reader and verify that expansion is lossless.

Run at least three random seeds for the stochastic branch. For the deterministic branch, the only stochasticity should be explicitly disabled; do not silently mix Gumbel noise with MAP decoding.

### Minimal candidate outputs and metrics

For each context and step budget, report:

- exact target block recovery and exact expanded `(WP,count)` recovery;
- legal-candidate rate after expansion and exact composition/count conservation;
- top-1 and top-k candidate recovery/rank when multiple legal candidates exist;
- average target length, number of forward passes, wall-clock decode time and peak memory;
- NLL/per-token cross-entropy on masked positions, separated by mask ratio if available;
- deterministic versus stochastic diversity across repeated decodes;
- recovery as a function of `T`, rather than only the best step count.

Because the current blocks describe protostructures/count assignments rather than full coordinates, MaskGXT’s MR, RMSE, METRe and cRMSE cannot be used directly. They become applicable only after a separately specified structure instantiation/relaxation stage. If that stage is added, use the paper’s StructureMatcher tolerances and report both one-to-one MR/RMSE and composition-pooled METRe/cRMSE, with a fixed candidate budget `S`.

## What transfers directly and what needs adaptation

| MaskGXT component | Direct transfer? | Adaptation required |
|---|---|---|
| All-mask initialization with condition fields visible | Yes | Conditions are `G + composition`; no predicted SG branch when SG is fixed. |
| Parallel posterior prediction at every step | Yes | Predict block symbols, not lattice/coordinate/Wyckoff streams. |
| Confidence-ranked fixing and remasking | Yes | Define a comparable confidence across block positions; use log posterior of the selected block. |
| Cosine remaining-mask schedule | Yes as a baseline | Sweep `T` because block sequences are much shorter than 1+6+5N. |
| Gumbel/multinomial stochastic branch | Optional | Pre-register temperature/noise; keep separate from deterministic MAP. |
| SG posterior stratification | No for fixed-G conditioning | Could be added only in an SG-generating variant; do not claim it is part of the current task. |
| StructureMatcher, MR/RMSE, METRe/cRMSE | No for protostructures | Use block/expanded occupancy metrics; apply crystal metrics only after full-structure realization. |
| MaskGXT 150-step setting | No direct transfer | It is tied to the much longer full-crystal token sequence and should not be treated as a default. |

## Interpretation guardrails

- A gain from `T>1` over one-step MAP would show that iterative conditional denoising matters for the count-block model; it would not by itself prove the tokenizer is superior.
- A gain in block exact match without a gain in expanded legal-candidate recovery may be only a vocabulary-level effect.
- A memory reduction with unchanged or unresolved matching metrics is an efficiency observation, not generation-quality evidence.
- The MaskGXT paper’s reported MR/METRe results validate its own full-crystal tokenization and sampler; they do not validate the current Wyckoff count-block tokenizer.

## Reading record

- Local source: `/Volumes/Extention/Users/guicheng/Code/CodeX-tmp/2026-09-01/bang-w/work/papers/maskgxt-2606.22866.txt`; read the full extracted paper, with sampling/evaluation emphasis on §§4–5 and Appendices A–C/D.
- Official implementation checked for sampling details: <https://github.com/kiyoung98/maskgxt/blob/main/sample.py>, especially `sample_batch`, `probe_sg_posterior`, `assign_group_space_groups`, `SG_POST_FLOOR=0.02`, cosine remask retention and final residual-mask repair.
- Paper metadata/source: <https://arxiv.org/abs/2606.22866>.

No code was changed and no experiment was run.
