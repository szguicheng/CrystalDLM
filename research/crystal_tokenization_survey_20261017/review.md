# Local chapter review — 2026-10-17

## Scope and source of truth

Initial local survey base: Overleaf commit `60733e0119d3344d6be788aaaa5f130a98939a2a`. The preceding version was uploaded as Overleaf commit `8352fefd4d68e4cba879fa18b6cb7d47c4d8cd85`. The advisor revision on 2026-10-18 is local only; see `revision-20261018.md` for the current outline and validation.

The canonical section is `survey-overleaf-6a9988f200d1def8f07b7fbc/sections/4_3_5_crystals.tex`. Both `main.tex` and `crystals-preview.tex` include this file. References are in `bib/4_3_5_crystals.bib`.

## Previous outline (2026-10-17)

1. Crystal identity: finite descriptions of a periodic structure and equivalent records.
2. Atomic graphs and explicit geometry: distinguish CGCNN features, coGN symmetry-reduced graph edges, DiffCSP state variables, and CDVAE learned latent reconstruction.
3. Sequences: distinguish numerical structure records, canonicalized complete descriptions, periodic topology requiring geometry inference, and symmetry-augmented graph strings.
4. Symmetry: distinguish geometry-complete orbit descriptions, coordinate-free protostructures, text tokens over Wyckoff fields, and mixed discrete-continuous field sequences.
5. Coverage: defects, supercell concentration, occupancy, symmetry tolerance, equivalent-input tests, and diffraction as a linked measurement.

## Claim and evidence audit

| Claim | Primary evidence | Resolution |
| --- | --- | --- |
| CGCNN is a prescribed graph input with learned property features, not a reversible crystal decoder | CGCNN paper and official `data.py`/`model.py`; geometry_evidence.md | Supported. The value-type column records mixed discrete element and continuous distance features. |
| coGN reduces repeated sites through asymmetric-unit nodes and affine image edges | Digital Discovery paper and official construction code | Supported. The table marks symmetry reduction but no structure decoder. |
| DiffCSP directly models lattice and fractional coordinates | NeurIPS 2023 paper, official diffusion implementation | Supported. Table refers to the explicit state, not a tokenizer autoencoder. |
| CDVAE is a continuous latent encoder with approximate reconstruction | ICLR 2022 paper and official encoder/decoder code | Supported. No unsupported reconstruction-rate comparison is included. |
| CIF text preserves numeric fields but does not uniquely canonicalize crystal identity | CrystaLLM paper and Mat2Seq analysis | Supported. Numeric precision remains explicit. |
| Mat2Seq canonicalizes and uses symmetry-reduced sites; numeric output is rounded | Mat2Seq §§3.2–3.5 | Supported within the paper's stated transformation setting. |
| SLICES stores periodic topology and requires geometric inference | SLICES representation, encoding and SLI2Cry sections | Supported. Geometric reconstruction is not labeled exact. |
| SLICES-PLUS adds a separate symmetry-operation sequence alongside the quotient graph | SLICES-PLUS paper, representation design | Supported. The table treats graph and symmetry streams as the represented units. |
| CrystalFormer retains coordinates and lattice; WyFormer does not | Original papers and author implementations | Supported. WyFormer global space-group token added after independent review. |
| PLaID++ uses text serialization of geometry-complete Wyckoff fields | PMLR 2026 paper, appendix and official parser | Supported. One crystallographic site is not equated to one LLM token. |
| Matra-Genoa uses an invertible mixed alphabet of discrete fields and continuous geometry | Matra-Genoa arXiv v2, representation and experiments | Supported. The table distinguishes it from text-only and continuous-only representations. |

## Five review questions

- **Contribution:** Is this a comparative synthesis? Yes: twelve table methods share defined feature columns, with four additional sources covering graph, symmetry-string, and historical Wyckoff context.
- **Clarity:** Is the unit being compared clear? Yes: prescribed inputs, learned latent and symbolic sequences are identified in both prose and table notes.
- **Empirical claims:** Are new performance results implied? No; this is a source-based survey section, with no new training or numerical benchmark claims.
- **Evaluation:** Are tokenizer properties conflated with whole-model utility? No; reconstruction target, geometry retention and equivalence mechanisms are distinguished.
- **Method boundaries:** Are a protostructure, a complete structure and a measurement distinct? Yes; geometry completion, numerical precision and diffraction are discussed separately.

## Local build

From the survey checkout:

```sh
tectonic --keep-logs --keep-intermediates --outdir ../research/crystal_tokenization_survey_20261017/build/preview crystals-preview.tex
tectonic --keep-logs --keep-intermediates --outdir ../research/crystal_tokenization_survey_20261017/build/full main.tex
```

Build logs and rendered review pages are retained under this research directory. The full manuscript retains its existing overview figure, for which Tectonic reports a PDF-version/tag warning; the added section itself has no LaTeX or bibliography warnings.
