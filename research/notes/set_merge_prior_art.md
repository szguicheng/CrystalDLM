# Set-level Wyckoff merge tokenizer: prior-art audit

## Audited proposal

The proposed second tokenizer is a multiset analogue of BPE:

- for each element, collect its Wyckoff-orbit multiset under one space group;
- learn frequency-based pair merges from MP-20 train only, without sequence adjacency;
- each merge is an exact multiset of at most four Wyckoff orbits under the same space group;
- retain all singleton WP symbols as fallback, so decoding remains lossless and multiplicity counts are exact;
- compare against ordinary `(WP, count)` run-length encoding;
- pilot vocabulary: 256 merge rules; reported extra compression on held-out units is about 6%; no model-quality gain is claimed yet.

The question is whether this directly repeats existing crystal or molecular tokenizers, and which part is genuinely specific to unordered Wyckoff multisets.

## Closest prior art (ranked)

### 1. GraphBPE: molecular graph pair merging — closest general algorithm

Shen and Póczos, *GraphBPE: Molecular Graphs Meet Byte-Pair Encoding*, arXiv:2407.19039 / ICML 2024 AI for Science Workshop, adapts BPE to molecular graphs. Its official repository describes the procedure as initializing atom-level tokens, finding frequent graph substructures/pairs, merging them into larger substructures, and using the learned merge vocabulary as a model-independent preprocessing step. The paper’s central abstraction is not an ordered SMILES adjacency pair: graph neighborhoods and substructures are identified from graph connectivity and can be canonicalized across graph isomorphism.

Overlap with the Wyckoff proposal:

- frequency-trained pair/fragment merges;
- singleton fallback atoms/fragments;
- a finite learned vocabulary used losslessly to encode/decode larger structures;
- compression/token-count reduction separated from the downstream model.

Difference:

- GraphBPE merges connected graph fragments, while the proposal merges an unordered multiset of crystallographic orbit records;
- graph fragments carry connectivity and boundary structure, while a WP merge carries SG, WP identity, multiplicity and occupancy counts;
- GraphBPE is not a composition-exact crystallographic encoding and does not enforce multiplicity arithmetic or fixed-site exclusivity.

Verdict: the merge-learning algorithm is not new. The possible contribution is the crystallographic set domain and exact multiplicity-preserving codec, not “BPE for structures” in general.

### 2. SAFE: unordered fragment blocks for molecular generation — closest unordered representation

Noutahi et al., *Gotta be SAFE: A New Framework for Molecular Design*, arXiv:2310.10773 / Digital Discovery 2024, represents a molecule as an unordered collection of interconnected fragment blocks while retaining a lossless mapping to a valid SMILES graph. The official SAFE documentation describes fragment blocks as order-insensitive at the molecular level and supports scaffold decoration, fragment linking and other fragment-constrained generation tasks. The paper reports a BPE tokenizer over SAFE representations with a finite vocabulary (1180 tokens in the published setup), but the chemically meaningful unit is the fragment block and attachment-point notation.

Overlap:

- explicitly removes arbitrary sequence order from the semantic representation;
- keeps a lossless singleton/fragment decomposition;
- uses a finite fragment vocabulary for generative modeling;
- treats token-count reduction as an efficiency benefit, not proof of model quality.

Difference:

- SAFE fragments are connected molecular subgraphs with attachment points; the proposed units are same-SG WP orbit multisets with no sequence adjacency;
- SAFE’s ordering issue concerns equivalent fragment arrangements and SMILES serialization, not Wyckoff multiplicity equations;
- SAFE does not provide a crystal-space-group/multiplicity legality layer.

Verdict: unordered fragment vocabulary is established. The crystal-specific distinction can be real only if the merge rule is defined as an exact multiset operation with canonical SG/WP/multiplicity semantics and is shown to preserve all legal decodings.

### 3. Kong et al. Principal Subgraph Extraction — earlier molecular precedent

GraphBPE §2 (Related Work) explicitly identifies Kong et al., *Molecule generation by principal subgraph mining and assembling*, NeurIPS 2022, as the earlier BPE/PSE line. GraphBPE Appendix B (§B, especially the PSE contextualizer discussion) says PSE can be recovered by skipping GraphBPE preprocessing and using a contextualizer that does not include neighborhood context. Kong’s method initializes atom-level subgraphs, mines frequent principal subgraphs and assembles molecules from the learned vocabulary. This is the more appropriate original citation for the PSE/“principal subgraph” idea; *Fragment-based Pretraining and Finetuning on Molecular Graphs* (Luong & Singh 2023) is a later downstream use of PSE, not the origin.

Overlap:

- data-driven frequency merge rules;
- vocabulary size fixed in advance;
- singleton atoms retained as the base vocabulary;
- fragments can be encoded and reconstructed through a learned decomposition.

Difference:

- principal subgraphs are topological graph fragments and may overlap during vocabulary mining;
- the proposed WP merge is a disjoint, exact multiset aggregation with a hard upper bound of four orbit records and same-SG scope;
- no crystallographic multiplicity/DOF semantics exist in the molecular method.

Verdict: frequency-based fragment mining is general prior art; the exact multiset codec and crystallographic legality are the only plausible domain-specific difference.

### 4. HME / Molecular Fragment Tokenizer — recent explicit fragment vocabulary

The Nature Communications article *Navigating chemical-linguistic sharing space with heterogeneous molecular encoding* describes a Molecular Fragment Tokenizer that starts from atoms and repeatedly merges the most statistically frequent adjacent substructures until a predefined vocabulary size (800), then adds fragment entries to an LLM vocabulary. This is a direct modern example of “frequent pair/substructure merge → finite learned vocabulary → tokenizer,” but it remains graph-adjacent and ordered by molecular connectivity. In this audit I checked the publisher’s article page and method description/search extract, not the full supplementary derivation; treat this item as corroborating recent evidence rather than a full-paper audit.

Verdict: a 256-rule frequency vocabulary is not novel as a vocabulary-learning procedure. The proposal’s potential novelty is the absence of adjacency and the exact WP multiset semantics, which need to be isolated in ablations.

## Crystal-specific controls

### Wren / WyFormer

Wren (Goodall et al., Sci. Adv. 2022) treats a crystal’s Wyckoff representation as a multiset of `(WP/letter, element)` members and applies message passing followed by permutation-invariant pooling. This is the strongest existing crystal precedent for unordered WP-set encoding, but it does not learn merge rules or a variable-length codec. WyFormer (Kazeev et al., ICML 2025, arXiv:2503.02407) uses separate categorical fields such as element, site symmetry and enumeration, random WP-order augmentation and no positional encoding; its official code has no BPE/set-merge vocabulary. These works make the unordered WP domain established, while the merge grammar is new only in the narrow engineering sense.

### WyckoffDiff

WyckoffDiff (arXiv:2502.06485; official `httk/wyckoffdiff`) represents all WP slots for a space group and diffuses per-element occupancy counts, with separate constrained and unconstrained position variables. It is an occupancy-set representation with exact crystallographic semantics, but not a merge tokenizer. It is therefore the closest control for representation legality and symmetry, not for pair-merge learning.

### CrystalFormer / Matra-Genoa

CrystalFormer uses an ordered Wyckoff-letter sequence plus continuous coordinates and lattice parameters; Matra-Genoa uses a reversible ordered mixed sequence. Neither learns unordered frequency merge rules. Their existence does not establish the merge idea, but they are relevant baselines for whether a 6% token reduction changes any actual generation task.

## What is and is not a method difference

### General algorithm already known

The following claims would overstate novelty:

- “first frequency-based pair merge vocabulary for chemical structures” — GraphBPE, principal subgraph mining, SAFE-related tokenizers and HME already do this;
- “first unordered fragment/set representation” — SAFE and molecular set methods establish unordered chemical representations;
- “first learned finite fragment vocabulary with singleton fallback” — standard in graph/molecular fragment tokenizers;
- “first Wyckoff multiset representation” — Wren, WyFormer and WyckoffDiff already use unordered or slot-based WP representations.

### Narrow crystallographic difference that may be defensible

The following combination was not found as a direct prior implementation in the bounded search:

1. merge candidates are **multiset** pairs, not adjacent sequence or connected graph pairs;
2. both members are tied to the same SG and exact WP orbit/multiplicity metadata;
3. merged tokens have a bounded number of orbit records (≤4) and retain exact additive multiplicity/count semantics;
4. singleton WP symbols remain available for lossless fallback and unseen combinations;
5. the merge grammar is trained only on MP-20 train and evaluated on held-out crystal units under canonical set encoding.

This is a domain-specific set codec, not a new BPE algorithm. Its value is an empirical question: the reported ~6% held-out compression is small, and no model benefit has yet been shown. The correct novelty language is “a Wyckoff-multiset merge codec / bounded set vocabulary” rather than “a new BPE.”

## Strongest objections

1. **Frequency does not imply semantic utility.** A frequent pair of WP records may only reflect dataset composition or common SGs; it may not improve conditional generation, legality, OOD composition, or sample efficiency.
2. **Six percent compression is modest.** If ordinary `(WP,count)` RLE already captures repeated orbits, the merge rules may only remove a small amount of redundancy. The comparison must equalize integer coding, metadata overhead, and singleton fallback costs.
3. **Canonicalization is decisive.** The same crystal can have equivalent WP settings/origin choices and arbitrary multiset order. Merge counts must be computed after canonicalization, and decoding must be invariant to equivalent representations; otherwise frequency rules learn coordinate-setting artifacts.
4. **Same-SG restriction limits transfer.** A merge rule tied to one SG/WP vocabulary may not transfer to another SG even when site symmetry and multiplicity are analogous. Cross-SG sharing needs an explicit canonical WP identity scheme; otherwise the vocabulary fragments by SG.
5. **No model gain yet.** Compression is not evidence that a tokenizer improves a generator. Required tests include matched-context baselines, set permutation robustness, exact reconstruction, legal occupancy rate, OOD compositions/SGs, infilling, latency and memory.
6. **It may still be a ranker/codec, not a tokenizer contribution.** If the merge vocabulary is only used to compress an already chosen WP multiset and the model’s prediction unit/likelihood remains unchanged, the contribution is a serialization codec. It becomes a tokenizer/model method only when merged units participate in prediction and improve a specified task.

## Bottom-line assessment

The general ingredients—frequency pair merging, fragment vocabularies, unordered chemical sets, lossless singleton fallback, and WP multiset representations—are all established in nearby work. The closest two are **GraphBPE** for the merge algorithm and **SAFE** for unordered lossless fragment blocks; **Wren/WyFormer/WyckoffDiff** are the closest crystal representation controls. The potentially specific part is applying a bounded, exact multiset merge grammar to same-SG Wyckoff orbit records while preserving multiplicity and decoding. With only ~6% compression and no model result, it should be presented as a well-defined crystallographic tokenizer hypothesis and diagnostic codec, not as a demonstrated generative-method breakthrough.

## Sources checked and reading scope

- **GraphBPE (full primary-text audit):** arXiv HTML <https://arxiv.org/html/2407.19039>, plus official code <https://github.com/A-Chicharito-S/GraphBPE>. Checked §1 (motivation), §2 (Related Work, including the explicit Kong/PSE distinction), §3.1 (BPE), §4.2 (GraphBPE algorithm), §5.2 (tokenization), Appendix B (PSE contextualizer), and the result discussion in Appendix C. GraphBPE itself says PSE is recovered by skipping preprocessing and using the non-contextual PSE contextualizer.
- **Kong et al. PSE (direct cited primary source):** NeurIPS 2022 OpenReview page <https://openreview.net/forum?id=ATfARCRmM-a>, cited by GraphBPE as “Molecule generation by principal subgraph mining and assembling.” This audit used the direct citation and GraphBPE’s Appendix B reconstruction statement; the Kong PDF was not independently read end to end here, so claims about its exact implementation are limited to that cited scope.
- **SAFE (full primary-text audit):** arXiv HTML <https://arxiv.org/html/2310.10773>, code link <https://github.com/datamol-io/safe/>. Checked §3 (SAFE algorithm), §3.1 (SMILES-to-SAFE fragmentation and unordered fragment arrangement), §3.2 (fragment-constrained tasks), §4.1 (BPE tokenizer, 1180-token vocabulary), and §4.3 (fragment-constrained evaluation).
- **Principal-subgraph downstream use (partial primary-text check):** *Fragment-based Pretraining and Finetuning on Molecular Graphs*, OpenReview PDF <https://openreview.net/pdf?id=77Nq1KjmLl>, checked the fragment-vocabulary/PSE excerpt only. It is cited as a later application, not as the original PSE source.
- **HME (partial publisher-page check):** *Navigating chemical-linguistic sharing space with heterogeneous molecular encoding*, Nature Communications <https://www.nature.com/articles/s41467-026-74666-w>. Checked the publisher’s Molecular Fragment Tokenizer description and iterative frequency-merge statement; this item was not read as a complete paper/supplement, so it is corroborating recent evidence only.
- **Wren (full primary-text audit in the earlier task):** Goodall et al., Sci. Adv. 8, eabn4117 (2022), official PDF <https://api.repository.cam.ac.uk/server/api/core/bitstreams/f4802430-f2ba-4b55-af97-d449ed2616ce/content>; checked pp. 1–2 (Wyckoff multiset representation and message-passing model) and pp. 6–7 (embedding and equivalent-representation augmentation).
- **WyFormer (full paper/code audit in earlier tasks):** arXiv <https://arxiv.org/abs/2503.02407>, official repository <https://github.com/SymmetryAdvantage/WyckoffTransformer>; checked §2.1–2.3, Appendix R and tokenizer YAML/source. No set-merge/BPE vocabulary was found.
- **WyckoffDiff (full paper/code audit in earlier tasks):** arXiv <https://arxiv.org/abs/2502.06485>, official repository <https://github.com/httk/wyckoffdiff>; checked §3.1–3.4 and Appendix A. Occupancy diffusion is not a merge tokenizer.
- **CrystalFormer / Matra-Genoa (full paper audits in earlier tasks):** <https://arxiv.org/abs/2403.15734> and <https://arxiv.org/abs/2501.16051>; checked their representation and sampling sections. Neither uses unordered frequency merge rules.

No code was changed and no experiment was run.
