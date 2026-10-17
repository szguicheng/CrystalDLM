# Local chapter review — 2026-10-17

## Scope and source of truth

Survey base: Overleaf commit `60733e0119d3344d6be788aaaa5f130a98939a2a`. Only the local checkout was edited; no push or Overleaf upload was performed.

The canonical section is `survey-overleaf-6a9988f200d1def8f07b7fbc/sections/4_3_5_crystals.tex`. Both `main.tex` and `crystals-preview.tex` include this file. References are in `bib/4_3_5_crystals.bib`.

## Reverse outline

1. Crystal identity: finite descriptions of a periodic structure and equivalent records.
2. Atomic graphs and explicit geometry: distinguish CGCNN features, DiffCSP state variables, and CDVAE learned latent reconstruction.
3. Sequences: distinguish numerical structure records, canonicalized complete descriptions, and periodic topology requiring geometry inference.
4. Symmetry: distinguish geometry-complete orbit descriptions, coordinate-free protostructures, and text tokens over Wyckoff fields.
5. Coverage: defects, supercell concentration, occupancy, symmetry tolerance, equivalent-input tests, and diffraction as a linked measurement.

## Claim and evidence audit

| Claim | Primary evidence | Resolution |
| --- | --- | --- |
| CGCNN is a prescribed graph input with learned property features, not a reversible crystal decoder | CGCNN paper and official `data.py`/`model.py`; geometry_evidence.md | Supported. Table explicitly selects the released Gaussian-distance features for the discreteness column. |
| DiffCSP directly models lattice and fractional coordinates | NeurIPS 2023 paper, official diffusion implementation | Supported. Table refers to the explicit state, not a tokenizer autoencoder. |
| CDVAE is a continuous latent encoder with approximate reconstruction | ICLR 2022 paper and official encoder/decoder code | Supported. No unsupported reconstruction-rate comparison is included. |
| CIF text preserves numeric fields but does not uniquely canonicalize crystal identity | CrystaLLM paper and Mat2Seq analysis | Supported. Numeric precision remains explicit. |
| Mat2Seq canonicalizes and uses symmetry-reduced sites; numeric output is rounded | Mat2Seq §§3.2–3.5 | Supported within the paper's stated transformation setting. |
| SLICES stores periodic topology and requires geometric inference | SLICES representation, encoding and SLI2Cry sections | Supported. Geometric reconstruction is not labeled exact. |
| CrystalFormer retains coordinates and lattice; WyFormer does not | Original papers and author implementations | Supported. WyFormer global space-group token added after independent review. |
| PLaID++ uses text serialization of geometry-complete Wyckoff fields | PMLR 2026 paper, appendix and official parser | Supported. One crystallographic site is not equated to one LLM token. |

## Five review questions

- **Contribution:** Is this a comparative synthesis? Yes: nine methods share defined feature columns, with equivalence and reconstruction treated explicitly.
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
