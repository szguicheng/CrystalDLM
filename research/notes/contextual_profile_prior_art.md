# Contextual profile prior-art audit

## Scope and precise candidate definition

The candidate under review is deliberately narrower than a generic crystal generator:

> Given a space group `G` and per-element atom counts `N_e`, retain only contexts whose complete legal single-element Wyckoff assignment set has size at most 256, and exhaustively enumerate the full assignment set for every retained context. Encode each candidate with shared embeddings of WP identity, multiplicity, degrees of freedom, and occupancy count. Use each element as a masked prediction unit; do not allocate a separate learned vector to every complete combination. Compare against an additive per-WP energy baseline, while the proposed candidate score is a nonlinear encoding of the complete occupancy vector. For the zero-free-DOF positions, enforce cross-element exclusivity with a fixed position bitmask and exact MAP dynamic programming.

The pilot is a bounded diagnostic range selection, not a candidate proposal or truncation stage: contexts with more than 256 legal assignments are excluded from this pilot, while retained train/validation/test contexts are fully enumerated. The reported retained fractions are train 96.24%, validation 95.92%, and test 96.60%.

The stated “at most eight zero-free-DOF WPs across the 230 groups” fact and the resulting fixed-mask DP bound are treated here as the candidate’s crystallographic premise; they were not independently recomputed in this audit.

The question is whether this is already directly present in primary literature or official code, and which part—if any—can be defended as a limited contribution. No implementation or experiment was run.

## Strongest direct overlap: exhaustive Wyckoff assignment plus learned ranking

The closest prior art is Parackal, Goodall, Faber and Armiento, *Identifying crystal structures beyond known prototypes from x-ray powder diffraction spectra*, Phys. Rev. Materials 8, 103801 (2024), arXiv:2309.16454, together with the authors’ Wren model (Goodall et al., Sci. Adv. 8, eabn4117, 2022).

The 2024 paper’s official abstract states the workflow explicitly: enumerate all symmetry-confined ways a composition can fit a space group, rank the element-assigned prototype representations with a Wyckoff-representation regression ML model, then assign/perturb free coordinates and validate with DFT. The full paper’s implementation section (AflowStringGenerator) describes the enumeration as a modified subset-sum search over WP multiplicities. It solves each element’s count as a nonnegative integer multiplicity combination and chains the per-element solutions while disallowing two elements from occupying a zero-DOF special position simultaneously. The paper’s own example enumerates all ways to place eight atoms in SG 61 before element decoration.

This is a direct overlap with the candidate’s first half:

- given composition and `G`;
- complete legal WP assignments;
- per-element multiplicity accounting;
- cross-element exclusion of fixed special positions;
- candidate ranking after enumeration.

The differences are also material. The 2024 workflow is a CSP/XRD solver, not a tokenizer or masked generative model. Its ranker is Wren, a coordinate-free Wyckoff-representation regression model. Goodall et al. define a Wyckoff representation as a multiset of `(Wyckoff letter, element)` members and use a message-passing network over all members followed by permutation-invariant pooling (Sci. Adv. 2022, pp. 2 and 6). Thus Wren is already a nonlinear whole-representation ranker; “nonlinear candidate score” alone is not a new contribution. On the other hand, the source does not describe the candidate as a shared token embedding followed by per-element masked prediction, nor does it use this pilot’s bounded context-range selection.

The fairest claim is therefore: a learned ranker over enumerated complete Wyckoff assignments is established prior art; a shared candidate-vector architecture tied to masked per-element occupancy prediction remains unverified as a direct precedent, but must be distinguished from Wren’s multiset ranker.

## Older and practical direct precedents

### PyXtal

The PyXtal paper (arXiv:1911.11123) states that, given chemical composition and space group, PyXtal can find a suitable combination of WP multiplicities with a step-wise merging scheme. This is an algorithmic legal-assignment generator, not a learned tokenizer or candidate embedding. It is enough to reject novelty claims about “first composition + SG → legal WP allocation.”

### GenSym

Han et al., *GenSym: A tool for random symmetric structure generation and its application in crystal structure prediction software*, J. Chem. Phys. 164 (2026), DOI 10.1063/5.0322405, is an especially close algorithmic precedent. The accepted manuscript describes:

- generating per-species WP combinations whose multiplicities sum to that species’ atom count;
- pairing/permuting the single-species combinations into multicomponent candidates;
- excluding collisions where two species simultaneously occupy fixed special positions;
- grouping WP choices by site symmetry (not only multiplicity), while preserving distinct site-symmetry classes;
- a second method that systematically generates all possible combinations under constraints.

GenSym is not a neural tokenizer. It has no shared learned candidate vector, masked predictor, or ranker of the proposed kind. Nevertheless, it directly overlaps the “single-element legal assignment then cross-element bitmask pairing” decomposition. A claim that this decomposition itself is novel would be untenable. A fixed bitmask plus exact MAP dynamic program may be a cleaner solver for the restricted zero-DOF subproblem, but the underlying constraint is already explicit in GenSym and the 2024 AflowStringGenerator.

### ShotgunCSP-GW

The 2024 *Shotgun crystal structure prediction using machine-learned formation energies* paper describes a Wyckoff generator that predicts a space group and Wyckoff-label configuration, then assigns elements according to composition/multiplicity. Its workflow is a candidate generator plus ML formation-energy screening, not a tokenization contribution. It is another reason to avoid claiming that a composition-aware Wyckoff candidate ranker is unprecedented.

## Diffusion and masked prediction precedents

### WyckoffDiff

WyckoffDiff represents a fixed SG using constrained positions `L0(s)` and unconstrained positions `L∞(s)`. Its `z∞` variable is a `|L∞| × N_a` matrix of per-element occupancy counts, while `z0` stores the element at each fixed position. D3PM diffuses these discrete variables and WyckoffGNN embeds WP identity, DOF, SG, time, and element/count information. This is a strong precedent for shared WP-aware occupancy representations and non-AR masked/discrete prediction, though it does not enumerate a small candidate list or rank complete occupancy vectors with a separate nonlinear candidate encoder.

### NextCrystal (arXiv:2602.17176, current v4)

NextCrystal takes an atom-count-expanded composition sequence, predicts a space group and per-atom Wyckoff letters with LLMs, then solves a constrained assignment problem. Its hard constraints require each element’s assigned count at a WP to be divisible by that WP’s multiplicity and require the per-element totals to equal the input composition. It uses constrained beam search to retain Top-4 assignments for each of Top-5 space groups, producing 20 templates before diffusion. This directly overlaps composition-conditioned legal candidate construction and masked/per-atom assignment, but differs in using LLM probabilities plus heuristic beam search rather than a shared candidate-vector masked model. It also weakens any claim that composition-conditioned symmetry assignment is new.

## Does the precise proposal have a direct precedent?

No source checked here gives all of the following in one model:

1. an explicit bounded pilot range retaining only contexts with at most 256 complete candidates, while exhaustively enumerating every retained context;
2. candidate vectors built from shared WP identity/multiplicity/DOF/occupancy embeddings;
3. per-element masked prediction over those candidate vectors;
4. a learned nonlinear whole-candidate score compared directly to an additive per-WP energy;
5. exact MAP DP exploiting at most eight zero-DOF positions across all 230 SGs.

That absence is only a bounded search result, not a global absence claim. More importantly, the components are not equally novel:

- (1) is an engineering budget and needs a coverage/recall curve; it is not a scientific novelty by itself.
- (2) is close to standard shared embedding of structured categorical fields and to WyckoffDiff/Wren representations. It becomes interesting only if the candidate vector is canonical, permutation-invariant, and composition-exact.
- (3) is a meaningful task formulation if it predicts a whole legal allocation for one element while conditioning on the remaining occupancy state. NextCrystal and WyckoffDiff are the closest conceptual baselines, but neither proves this exact architecture is already published.
- (4) is not novel as a ranker: Wren already performs nonlinear message passing over a complete Wyckoff multiset. The useful distinction would be an ablation showing that whole-vector interactions improve assignment selection beyond Wren-like set regression and additive WP scoring.
- (5) may be a useful exact solver result for the restricted fixed-position constraint, but it is an algorithmic optimization of a known legal-assignment problem. It should be reported as an exactness/complexity contribution, not a new crystallographic representation.

## Strongest objections a reviewer would raise

1. **“This is a re-packaging of enumeration + Wren.”** The 2024 XRD workflow already enumerates legal composition/SG/WP assignments and ranks element-assigned representations with Wren. The proposal must show why a masked predictor over candidate vectors is better than simply scoring the enumerated Wren inputs.
2. **“This is a re-packaging of GenSym/PyXtal.”** Per-element occupancy generation followed by cross-element exclusion is already a standard crystallographic construction. A bitmask/DP implementation is valuable only if it materially changes exactness or scaling in a measured regime.
3. **“The bounded pilot range may under-cover hard cases.”** The current implementation does not propose or truncate candidates: it excludes contexts whose complete legal assignment set exceeds 256 and fully enumerates the retained contexts. The resulting train/validation/test coverage (96.24%/95.92%/96.60%) is useful evidence for a pilot, but can still bias diagnosis toward easier composition/space-group contexts. The excluded contexts and their assignment-count distribution must be reported before generalizing beyond this bounded range.
4. **“A shared embedding is not automatically a tokenizer.”** If the model embeds fields and scores complete candidates but never changes the atomic prediction unit or supports reversible segmentation/merging, reviewers may call it a structured candidate ranker/decoder rather than a tokenizer innovation.
5. **“Nonlinear beats additive only by capacity.”** A nonlinear candidate encoder can win simply because it has more interaction terms. The additive baseline must match parameter count, candidate features, and training data; otherwise the result is an architecture-size comparison rather than evidence for a crystallographic representation.
6. **“The <=8 zero-DOF fact is not enough.”** Fixed-position exclusivity is only one legality condition. Per-element multiplicity sums, repeated use of non-fixed WPs, equivalent WP settings, site-symmetry compatibility, and possible higher-symmetry collapse must be handled independently.

## Limited contribution that can survive the prior art

A defensible contribution would be a **composition-exact, permutation-invariant candidate decoder** whose unit of prediction is a legal single-element WP allocation, with:

- a reproducible exhaustive candidate generator and explicit ≤256 range-selection policy, with the excluded-context distribution;
- shared WP/multiplicity/DOF/occupancy features rather than one vector per full combination;
- exact cross-element legality via DP/bitmask for the fixed-position part;
- a controlled comparison to Wren-like multiset regression, additive per-WP energy, NextCrystal-style constrained search, and WyckoffDiff occupancy diffusion;
- ablations separating candidate enumeration, feature sharing, masked prediction, nonlinear interactions, and DP from the downstream energy/structure model;
- metrics for exact composition/legality, assignment accuracy, OOD compositions, inference cost, and final relaxed stability, plus coverage of contexts excluded by the bounded range.

This would be a bounded architecture/solver contribution. It should be described as combining and specializing known crystallographic enumeration, shared set embeddings, and masked prediction, not as the first legal Wyckoff assignment, first composition-conditioned symmetry model, first candidate ranker, or first use of multiplicity constraints.

## First-party sources checked

- Goodall et al., Sci. Adv. 8, eabn4117 (2022), official Cambridge-hosted PDF: coordinate-free Wyckoff representation, multiset regression, message passing and permutation-invariant pooling (pp. 1–2, 6).
- Parackal et al., Phys. Rev. Materials 8, 103801 (2024), DOI 10.1103/PhysRevMaterials.8.103801; official APS abstract and author-hosted full PDF: exhaustive symmetry-confined enumeration, element assignment, Wren ranking, and subset-sum implementation.
- PyXtal, arXiv:1911.11123: composition + SG to suitable WP combinations with a step-wise merging scheme.
- Han et al., GenSym, J. Chem. Phys. 164 (2026), DOI 10.1063/5.0322405: per-species multiplicity combinations, cross-species pairing, fixed-site exclusivity, site-symmetry grouping, and systematic/random generation methods.
- WyckoffDiff, arXiv:2502.06485 and official `httk/wyckoffdiff`: `z∞` per-element/WP occupancy counts, `z0` fixed-position elements, D3PM and WyckoffGNN.
- NextCrystal, arXiv:2602.17176v4: composition-expanded sequence, SG/WP prediction, multiplicity-constrained assignment and beam-search candidate templates.
- ShotgunCSP-GW, npj Computational Materials 10 (2024), DOI 10.1038/s41524-024-01471-8: SG/Wyckoff generator and ML formation-energy screening.

No code was changed and no experiment was run.
