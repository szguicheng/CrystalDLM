# Sequence representations: source checks, 2026-10-17

Scope: source evidence for survey section 4.3.5, not a new benchmark. The manuscript is based on Overleaf commit 60733e0119d3344d6be788aaaa5f130a98939a2a.

## CrystaLLM

- Primary publication: Luis M. Antunes, Keith T. Butler, Ricardo Grau-Crespo. Crystal structure generation with autoregressive large language modeling. Nature Communications 15 (2024). DOI: https://doi.org/10.1038/s41467-024-54639-7
- Source checked: https://www.nature.com/articles/s41467-024-54639-7 ; earlier methods text https://arxiv.org/html/2307.04340v2 section 2.2.
- Rule-based token identities include CIF tags, elements, space-group symbols, digits and punctuation. The language-model embedding table does not make lexical token formation neural.
- Standardized CIF stores explicit lattice and fractional-coordinate information. Decoding tokens recovers the serialized record; generated records still require structural and symmetry checks.
- Discussion explicitly states that the training set has no fractional site occupancies and the model cannot generate site-occupancy-disordered structures. Do not infer supported disorder from the general CIF file format.

## SLICES

- Hang Xiao, Rong Li, Xiaoyang Shi, Yan Chen, Liangliang Zhu, Xi Chen, Lei Wang. An invertible, invariant crystal representation for inverse design of solid-state materials using generative deep learning. Nature Communications 14, 7027 (2023). DOI: https://doi.org/10.1038/s41467-023-42870-7
- Primary paper, Results: SLICES representation; Encoding crystal structures; Reconstruction; Benchmark.
- String stores atom symbols and quotient-graph edges (source node, destination node, integer cell translation). Original encoding uses EconNN connectivity.
- It does not directly store input lattice metric or atomic coordinates, or explicit space-group orbit compression. Raw strings can differ in node/edge order; graph equivalence must not be described as identical strings.
- SLI2Cry builds an embedding, optimizes geometry with modified GFN-FF, then refines with pretrained M3GNet. Geometry recovery is approximate and prior-dependent, not an exact inverse of all input coordinates.
- Source restricts original decoder evaluation to atomic numbers <=86 and connected 3D nets; low-dimensional structures and MOFs are not universally reconstructed. No rates will be ranked across incomparable datasets in this section.

## Mat2Seq

- Keqiang Yan, Xiner Li, Hongyi Ling, Kenna Ashen, Carl Edwards, Raymundo Arroyave, Marinka Zitnik, Heng Ji, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji. Invariant Tokenization of Crystalline Materials for Language Model Enabled Generation. NeurIPS 2024. DOI: https://doi.org/10.52202/079017-3971
- Official record: https://proceedings.neurips.cc/paper_files/paper/2024/hash/e23133d34964a0a09f6d076fc4b922a4-Abstract-Conference.html
- Full text checked: https://arxiv.org/html/2503.00152v1 , sections 3.2-3.5.
- Niggli reduction, origin selection and ordering define the sequence; six lattice parameters, space-group information, irreducible sites and fractional coordinates allow structure recovery.
- The paper proves invariance for its defined SE(3) and periodic transformations. Attribute the result to that construction; do not generalize to arbitrary defective/disordered inputs or floating-point implementations without tests.
- Section 3.5 rounds real values to four decimal places. Structural completeness of the mathematical representation must be distinguished from exact numeric recovery after serialization.

## Draft architecture

1. Periodic crystal identity and equivalent descriptions.
2. Atom/graph/continuous geometric units, including learned latent reconstruction.
3. Explicit structure strings and topology strings.
4. Symmetry-orbit units and prototype versus geometry-complete descriptions.
5. Feature matrix with defined checkmarks, plus scientific coverage/evaluation synthesis.

No method is labeled a new tokenizer merely because it has a Transformer, diffusion decoder or learned task embedding.
