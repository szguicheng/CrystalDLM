# Count-block Wyckoff tokenizer: bounded prior-art check

## Candidate under review

The revised candidate starts from each element’s nonzero `(Wyckoff position, orbit-occupancy count)` RLE records. From the training set it learns at most three-record unordered blocks, retaining all singleton WP records as RLE fallback. A block is directly masked/predicted under space-group and composition conditioning, then decoded through the same legal-candidate reader. All fields remain standard crystallographic WP identity, multiplicity and occupancy counts. The current seed17 result is only a small-budget forward NLL signal; matching-rate status is uncertain, and the reported memory reduction is about 15%. Full training and review are incomplete, so no effectiveness claim is made here.

The narrow question is whether WyckoffDiff or WyFormer already uses the same **multi-WP count block as one discrete prediction/denoising unit**, rather than merely representing occupancy counts or predicting WP fields separately.

**Counting convention:** throughout this comparison, a WP occupancy count is a count of independent orbit instances / representative sites, not the number of physical atoms after symmetry expansion. An occupied fixed WP of multiplicity m contributes m atoms in the tabulated cell. Thus a count of one must not be interpreted as one atom in the full conventional cell. The paper sometimes uses atom/site wording at the representative level.

## WyckoffDiff: exact source boundary

Source read: local full extracted text for `arXiv:2502.06485`, including main-text §3.1–§3.4 and Appendix A. The official code is `https://github.com/httk/wyckoffdiff`.

### Fixed versus free WP variables

Section 3.1, PDF pp. 3–4, defines all WPs for a space group as `L(s)` and partitions them into:

- `L0(s)`: constrained positions with no degrees of freedom; a WP is either empty or occupied by one orbit of a single element, with m symmetry-expanded atoms for multiplicity m;
- `L∞(s)`: unconstrained positions with degrees of freedom on a line, plane or volume; each element can occupy a given unconstrained WP from 0 to a dataset-derived maximum `P` times.

The representation is explicit in Eq. (3): `(s, z∞, z0)`. The free-position variable is `z∞ ∈ {0,…,P}^{|L∞(s)|×Na}`. Its cell `z∞(i,j)` counts occupied representative instances of element `j` on unconstrained WP `i` (PDF p. 4; see counting convention above). The fixed-position variable is `z0 ∈ {0,…,Na}^{|L0(s)|}`, where 0 means empty (same page). This is already a per-element × WP occupancy-count representation, with exact fixed-site occupancy semantics.

### Generation and denoising unit

Section 3.2, PDF p. 4, factorizes generation as empirical space-group sampling followed by `pθ(z0,z∞|s)`. Section 3.3, PDF pp. 4–5, says both `z0` and `z∞` are discrete variables and uses D3PM. A datapoint is `x=(x1,…,xD)`; noise is added independently to each discrete variable, and the backward model predicts `pθ(x0^k | xt+1)` for each variable `k` (Eqs. 5–6).

Section 3.4, PDF pp. 5–6, makes the unit explicit: the network outputs `D = |L∞(s)| × Na + |L0(s)|` categorical distributions, one for each element/WP count variable in `z∞` and one for each fixed WP in `z0`. Each WP is a node in a fully connected WyckoffGNN; it is not a merged block of several same-element WP count records.

Appendix A.1, PDF p. 13, confirms the embedding path: `Embedding(z0)` and `Linear(Embedding(z∞))`; the number of atoms of each element in `z∞` is embedded into a scalar and combined with position features. Algorithm 2 (same page) outputs `p∞` over `L∞ × Na × P` and `p0` over `L0 × Na`. The position features include DOF, Wyckoff letter, space group and timestep. None of the main text or Appendix A describes concatenating multiple nonzero `(WP,count)` records for one element into one discrete diffusion variable, nor a learned frequency block vocabulary.

### What overlaps the candidate

WyckoffDiff already supplies the physical occupancy substrate: exact WP identity, per-element counts on free WPs, fixed-site constraints, shared WP embeddings and non-AR discrete denoising. A count-block tokenizer would therefore be a reparameterization/compression of `z∞` (plus any analogous fixed-site records), not a new crystallographic variable family.

Independent corruption and factorized categorical outputs do **not** imply independent neural modeling: the fully connected WyckoffGNN conditions each output on the noisy protostructure and can learn dependencies across WPs. The block proposal must not claim to be the first way to model co-occupancy; it changes the categorical prediction/corruption unit and sequence organization.

### What was not found

Within the cited sections and appendix, no explicit implementation was found for:

1. an unordered block containing up to three distinct WP count records for the same element;
2. a frequency-learned block vocabulary with singleton fallback;
3. one D3PM categorical variable representing the entire block;
4. a block-level mask/prediction head whose output is decoded by the same legal candidate reader.

This is a bounded source-search conclusion, not a global absence claim about all WyckoffDiff forks or unpublished experiments.

## WyFormer: exact source boundary

Source read: local full extracted text for `arXiv:2503.02407v4`, especially §2.1–§2.3, Appendix C/Algorithm 1 and Appendix R; official repository `https://github.com/SymmetryAdvantage/WyckoffTransformer` was also checked in the earlier audit.

Section 2.1, PDF pp. 4–5, defines `R=(G,E,W)`, where each WP `wi=(si,ni)` is paired with an element `ei`. The structure is tokenized as one SG token followed by groups of **three tokens per WP**: element, site symmetry and enumeration. The main-text probability factorization is:

`p(ei | G, previous E,S,N)`, then `p(si | previous tokens, ei)`, then `p(ni | previous tokens, ei,si)`.

Section 2.2, PDF p. 5, says the three fields are embedded with separate lookup tables and concatenated. There is no occupancy-count field in the generated WP token. Multiplicity is recoverable from SG/site-symmetry/enumeration but is not the prediction unit.

Section 2.3, PDF pp. 5–6, trains next-part prediction with MASK and cross-entropy, then generates element → site symmetry → enumeration with STOP. WP order is shuffled for permutation-invariance augmentation, but each prediction remains a field of one WP. Appendix R (PDF p. 29) identifies the token as `(element, site symmetry, enumeration)` and reports the field-combination vocabulary statistics; it does not describe a learned unordered multi-WP count block.

### What overlaps the candidate

WyFormer establishes that WP sets can be modeled without positional encoding and that fields can be composed from shared categorical embeddings. Its field-wise MASK training is conceptually adjacent to block masking.

### What was not found

No explicit same-element multi-WP count record, RLE count token, frequency merge rule, or one-block prediction head appears in the cited WyFormer sections/appendices or official tokenizer configurations. The candidate therefore differs from WyFormer at the sequence unit, but still uses the same underlying WP identity/count semantics rather than a new crystallographic ontology.

## Direct comparison

| Aspect | WyckoffDiff | WyFormer | Proposed count-block pilot |
|---|---|---|---|
| Physical variable | `z∞(WP, element)=occupancy count`, `z0(fixed WP)=element/empty` | SG + per-WP element/site-symmetry/enumeration fields | RLE records `(WP, nonzero count)` grouped into ≤3-record same-element blocks |
| Basic denoising/prediction unit | One categorical `z∞` cell or one `z0` WP variable | One field of one WP triplet | One learned unordered block, with singleton RLE fallback |
| Shared WP features | WP/DOF/letter/SG/timestep embeddings | Separate element/site-symmetry/enumeration embeddings | WP identity/multiplicity/count features reused inside block encoder |
| Frequency merge vocabulary | Not reported | Not reported | 182 rules from training frequency |
| Exact physical constraints | Fixed WP 0/1; free WP 0…P | SG/WP field semantics; no count-unit block | Same legal candidate reader; composition + SG conditioning |
| Output level | Protostructure occupancy variables | Wyckoff representation | Same candidate/occupancy representation, compressed into blocks |

## Upper bound of the method difference

The strongest defensible distinction is **unit-of-prediction reparameterization**: WyckoffDiff predicts independent WP-element count cells, while the candidate predicts a small, unordered, frequency-learned group of count records as one masked unit and retains exact singleton fallback. This may change sequence length, conditional factorization, and memory use.

The distinction is not a new physical representation: each block expands to the same standard WP/count records. It is also not yet evidence of better generation. With seed17 only, forward NLL is a small-budget signal, matching rate is unresolved, and the approximately 15% memory reduction is an engineering observation pending matched full training. The correct claim is therefore “count-block factorization over an existing occupancy representation,” subject to validation against the unblocked RLE and the original WyckoffDiff-style independent-cell factorization.

## Required caution in novelty wording

- Do not say “first occupancy representation”: WyckoffDiff already has explicit per-element × WP counts.
- Do not say “first shared WP/count embedding”: WyckoffDiff Appendix A embeds count and WP position features.
- Do not say “first masked discrete crystal model”: WyckoffDiff is a D3PM with per-variable denoising.
- A bounded claim may be: “We test a frequency-learned, unordered, ≤3-record count-block codec with singleton RLE fallback on top of a Wyckoff occupancy representation.”
- Any effectiveness statement must wait for full training, matched evaluation, and resolved matching-rate results.

## Reading record

- WyckoffDiff local PDF/TXT: full main text plus Appendix A; key exact locations are §3.1–§3.4 (PDF pp. 3–6) and Appendix A.1/Algorithms 2–4 (PDF pp. 13–14).
- WyFormer local PDF/TXT: full main text plus relevant generation/token-analysis appendices; key exact locations are §2.1–§2.3 (PDF pp. 4–6), Appendix C/Algorithm 1 (pp. 14–16), and Appendix R (p. 29).
- Official WyckoffDiff repository: <https://github.com/httk/wyckoffdiff>.
- Official WyFormer repository: <https://github.com/SymmetryAdvantage/WyckoffTransformer>.

No code was changed and no experiment was run.
