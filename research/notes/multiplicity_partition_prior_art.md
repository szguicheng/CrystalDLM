# Wyckoff multiplicity-partition representation: bounded prior-art check

## Candidate under review

The alternative representation would condition on space group `G` and composition, then represent each element by a multiplicity histogram/partition

`sum_m m * k_{e,m} = N_e`,

where `k_{e,m}` is the number of independent orbits of element `e` at multiplicity `m`. WP identity and occupancy assignment would be predicted as a residual after this coarse multiplicity allocation. The intended claim under review is narrower than a new crystallographic concept: does any existing generator already make this histogram/partition an explicit discrete token or hierarchical latent variable, rather than merely using multiplicity in enumeration or a hard constraint?

## 1. WyCryst (Zhu et al., arXiv:2311.17916 v2)

### Primary source and reading scope

- arXiv record: <https://arxiv.org/abs/2311.17916> (42-page v2 PDF).
- Official code: <https://github.com/RaymondZhurm/WyCryst>.
- The full PDF was read through the representation/model and validation sections, including PDF pp. 5–15 and supplementary Wyckoff Gene tables (pp. 30–39).

### Representation

The main Workflow section (PDF p. 5) defines a Wyckoff representation `W=(X,S)` with one-hot SG array `S` and Wyckoff array `X=(F,V,W)`. `F` is element stoichiometry, `V` atomic features, and `W` contains the Wyckoff site occupancy and Wyckoff site multiplicity of each element. The “Wyckoff Gene” is the decoded composition + SG + corresponding Wyckoff site occupancy.

The representation section (PDF pp. 6–8) introduces degrees of freedom (DoF) for each WP and uses the combination of SG and `W` to constrain fractional coordinates. The PVAE reconstructs the whole arrays with MSE/cross-entropy plus a Wyckoff loss comparing the original formula to the reconstructed Wyckoff-weighted formula (p. 8). This is a full per-element WP assignment array, not a separate latent variable whose only content is `k_{e,m}`.

The supplementary generated-gene tables make this concrete. For example, Table S6 (PDF p. 32) records entries such as `{'Ca':['1b'], 'Ti':['1a'], 'O':['3d']}` and equivalent configurations; Table S7 (PDF pp. 36–39) contains elements assigned to repeated or multiple WPs such as `Te:['6c','6c','9d']`. The multiplicity histogram can be derived from these lists, but the model output is the WP identity list itself.

### Generation model and whether a histogram token exists

WyCryst uses a property-directed VAE latent space and then generates Wyckoff Genes, followed by PyXtal instantiation and DFT refinement (PDF pp. 5, 8–9). In ground-state CSP and polymorph CSP, it samples latent points around same-stoichiometry or same-symmetry training materials and decodes complete Wyckoff Genes (PDF pp. 12–15). There is no stated factorization “first predict a multiplicity histogram, then predict WP identities,” no histogram vocabulary, and no discrete token corresponding to `k_{e,m}`.

**Boundary:** WyCryst is the closest generative precedent for composition + SG + WP occupancy arrays and strictly ternary data. It does not establish a separate multiplicity-partition token or hierarchy. Recasting its `W` array into a histogram is a deterministic feature transform of an existing output.

## 2. NextCrystal (Mu et al., arXiv:2602.17176 v4)

### Primary source and reading scope

- Official arXiv HTML/current version: <https://arxiv.org/html/2602.17176> (v4, 6 July 2026).
- Read the abstract and main method/benchmark text, especially the lines corresponding to the introduction and method formulas (HTML approximately 26–82), plus the stated Appendix A.1–A.4 pointers. The available HTML does not expose the appendices as separate readable sections in this audit.

### Representation and assignment mechanism

NextCrystal expands composition into an atomic sequence with repeated element symbols. `LLM_g` predicts an SG; `LLM_w` predicts a Wyckoff letter for each atom conditioned on the expanded composition and SG via FiLM. The explicit optimization (Eqs. 1–3) then imposes:

- each assigned letter must be admissible in the predicted SG;
- for each element, the assigned count at a letter must be divisible by that WP’s multiplicity;
- the sum of assigned counts for each element must equal its input atom count.

The paper says this assignment is solved with a heuristic beam search (main method around HTML lines 68–78), selecting Top-5 SGs and Top-4 valid assignments per SG to obtain 20 templates before diffusion. This is a constrained per-atom letter assignment with multiplicity arithmetic, not a separate histogram token. The count partition is an implicit consequence of the assignment and beam constraints.

**Boundary:** NextCrystal is a strong prior for composition-conditioned multiplicity-aware site assignment and hierarchical candidate construction. It does not report a model variable `k_{e,m}`, a multiplicity histogram vocabulary, or a decoder stage that predicts only the histogram before WP identity. Calling the proposed histogram “first composition-to-multiplicity conditioning” would therefore be unsafe, but claiming a distinct factorization may still be a bounded architectural hypothesis.

## 3. Hornfeck & Červený: explicit multiplicity partitions, but combinatorics rather than generation

### Primary source and reading scope

- Hornfeck and Červený, *On the combinatorics of crystal structures. II. Number of Wyckoff sequences of a given subdivision complexity*, Acta Crystallographica Section A 79, 280–294 (2023), DOI <https://doi.org/10.1107/S2053273323002437>.
- Official IUCr PDF: <https://journals.iucr.org/a/issues/2023/03/00/uv5014/uv5014.pdf>.
- Read the full 15-page paper, focusing on pp. 2–5 (§2.1–§2.5), pp. 6–8 examples/algorithm discussion, and the appendices/pseudocode.

### Exact overlap

Section 2.1 (PDF p. 2) defines integer multipliers `μ_i` for each WP and sums the coupled multiplicity/arity vectors `(M_i,A_i)` to a target `(M,A)`. The paper explicitly identifies this as a restricted integer-partition/coin-change problem. Multiplicities in primitive cells are restricted to `{1,2,3,4,6,8,12,16,24,48}` and fixed sites have multiplier 0 or 1 (p. 2).

Section 2.3 (PDF pp. 3–4) gives the Cartesian-product search space over per-WP multipliers and notes that its solutions are found by exhaustive checking. Section 2.4 (PDF p. 4) uses generating polynomials where each WP contributes `sum_j x^{j M_i} y^{j A_i}`; the coefficient of `x^M y^A` counts valid combinations. Section 2.5 (PDF pp. 5–6) adds a third variable `z` for sequence length `k`.

This is a direct mathematical precedent for a multiplicity partition/histogram abstraction, including fixed-site restrictions and optional sequence length. However, it is not a learned tokenizer, latent variable, masked decoder or crystal generator. It counts/constructs WP sequences; it does not learn from MP-20 frequencies or predict a histogram as a neural generation stage.

## Comparison against the proposed factorization

| Question | WyCryst | NextCrystal | Hornfeck–Červený | Proposed histogram residual |
|---|---|---|---|---|
| Uses WP multiplicity to satisfy composition? | Yes, implicitly in WP occupancy arrays and Wyckoff loss | Yes, hard divisibility and exact per-element totals | Yes, explicit integer multipliers/generating functions | Yes, first-stage `k_{e,m}` condition |
| Explicit histogram/token `k_{e,m}`? | No evidence | No; per-atom letter LLM + beam | Mathematical multipliers, not model token | Intended yes |
| Learned generative model? | PVAE over full Wyckoff arrays | LLMs + constrained beam + diffusion | No | Intended hierarchical/masked model |
| WP identity residual after histogram? | Full WP identities directly decoded | Per-atom WP letters directly assigned | WP sequences counted/constructed | Intended residual WP assignment |
| Coordinates/full structure | PyXtal + DFT refinement | Diffusion conditioned on selected templates | None | Not specified in this audit |

## Bounded conclusion

Within these three closest sources, no paper explicitly implements `k_{e,m}` as a separate learned discrete token/latent stage followed by residual WP-identity prediction. The strongest overlap is conceptual rather than architectural:

- WyCryst already decodes composition + SG + per-element WP occupancy arrays through a PVAE;
- NextCrystal already separates composition-conditioned site assignment from geometric diffusion and enforces multiplicity divisibility exactly;
- Hornfeck–Červený already formalize multiplicity/arity partitions and generating-function enumeration.

Thus the proposed histogram factorization is at most a new **modeling decomposition over known Wyckoff variables**. It is not a new crystallographic concept, and it should not be sold as first use of multiplicity partitions. The meaningful test would be whether the explicit histogram stage improves conditional assignment calibration, exact composition legality, OOD composition/SG generalization or compute, relative to a model that predicts the existing WP occupancy array directly.

## What is not established

- No direct source in this bounded search was found that uses histogram `k_{e,m}` as a neural token vocabulary.
- No claim is made about unpublished code, later papers, or every crystal CSP implementation.
- WyCryst’s strict-ternary PVAE and NextCrystal’s beam search should not be conflated with a learned histogram representation.

## Sources

- WyCryst arXiv: <https://arxiv.org/abs/2311.17916>; official repository: <https://github.com/RaymondZhurm/WyCryst>.
- NextCrystal arXiv HTML/current v4: <https://arxiv.org/html/2602.17176>.
- Hornfeck & Červený DOI: <https://doi.org/10.1107/S2053273323002437>; official IUCr PDF: <https://journals.iucr.org/a/issues/2023/03/00/uv5014/uv5014.pdf>.

No code was changed and no experiment was run.
