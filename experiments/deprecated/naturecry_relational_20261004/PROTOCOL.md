# NatureCry Relational CNL v1

Date: 2026-10-04. Status: implementation and frozen-cohort offline validation
completed. The qualifying pilot is `pilot30_002` (30/30); the full run is
`run100_001` (100/100). See `RESULTS.md` for evidence and claim boundaries.

## Contract

This experiment implements the program-encoded, program-decoded relational
controlled natural language discussed with the user. Text normalization is
allowed: equivalent supported wording need not survive verbatim. Geometry must
recover the original input cell, elements and periodic atom positions.

This is a separate representation from the earlier fixed English field wrapper
and the LLM inline-parameter pilot. It does not overwrite either experiment.
There are no LLM calls, training, GPU jobs, new datasets or external uploads.

## Representation

The encoder consumes the existing hash-pinned structural facts. Its output is
one complete English document, without XML parameter tags, JSON, a CIF appendix
or a second complete atom list. Numeric values use Python float64 `repr`, not
rounded display values. Public symmetry tables are shared grammar knowledge,
not a material-specific side channel.

Orbit sentences express independent offsets, fixed-coordinate planes and
dependent-coordinate relations. Shared, multiplied and translated offsets are
geometric construction rules. The text does not state a Wyckoff label or carry
a separate orbit parameter vector. The parser builds an affine representative
relation and matches it, together with the standard-cell orbit multiplicity,
against the frozen public definitions of the stated space group. An ambiguous
or unsupported relation is rejected, not repaired from the source.

The representative is a single seed position, not every expanded atom. Orbit
multiplicity refers to the standard cell; an original primitive cell can contain
fewer atoms. Fixed representative coordinates describe the nominal symmetry
construction before any explicitly stated precision correction.

Cell dimensions, original-to-standard fractional frame transformation, rounding
order, stable correction targets, and residual displacements remain explicit
technical sentences. These details are necessary for faithful reconstruction,
even when they are less fluent than the spatial orbit descriptions. No tiny
frame coefficient, origin shift, lattice difference or residual is idealized.

## Decoder Boundary

The decoder receives only the saved English document and the frozen public
table. It does not receive the source CIF, encoder facts, original orbit label,
oracle plan, legacy correction-address map or encoder context. Source facts are
used only by the encoder and by the independent exact-plan comparison.

The new parser delegates validated plans to the existing read-only geometry
engine in `../naturecry_inline_20261003/geometry.py`. Its public table hash is
`ca56602252b7896121253c910739ef2fccf0ba947f2940358feb941f30c5e13d`.
The original expansion, image selection, deduplication, stable site ID,
rounding and correction order is unchanged.

## Validation

Use the unchanged ordered 100-case selection from
`../naturecry_roundtrip_20261001/run_001/selection.json` and the pinned facts in
`../naturecry_inline_20261003/prepared_final`. Start with its first 30 cases.
Expand to 100 only after all 30 pass every contract below under the same runtime
and selection hashes. Keep every selected case in the denominator; no filtering,
replacement, model retries or source-informed repair.

For each case:

1. Verify frozen source, facts and shared runtime provenance.
2. Encode facts into English and save the document.
3. Parse that document alone into a construction plan.
4. Require exact equality with the fact-derived reference plan, including all
   float64 values, frame information and residuals.
5. Reconstruct through the public geometry engine and export a CIF using 14
   significant figures.
6. Reopen both the original and reconstructed CIFs. Require original-cell
   lattice matrix component error strictly below `1e-8 angstrom`, and periodic,
   element-aware matched position error strictly below `1e-6 angstrom`.
7. Re-render the parsed plan into canonical text and parse it again. Require
   canonical text stability and exact plan equality.

Save per-case texts, parsed plans, reconstructed CIFs, full-denominator records,
summary, source snapshots and hashes. Recheck runtime hashes after execution.
No result from a changed implementation may qualify a later implementation's
30-to-100 gate.

## Claim Boundary

The reference-table audit found 230 groups and 1,731 definitions. Every free
axis has an exact identity-coordinate row with zero constant, each coordinate
row has at most one nonzero variable coefficient, and the signature
`(multiplicity, representative A, representative b)` is unique within a group.
There are 25 distinct coordinate-row forms. These properties were independently
checked against the hash-verified table before implementation.

They give a concrete inverse construction: parse direct numeric free offsets,
parse the remaining coordinate relations into `A` and `b`, and select the unique
public definition. No numerical representative fitting is needed. A compact
synthetic closure test covers all public definitions; it is a grammar test, not
an additional real-structure cohort or a physical-validity benchmark.

`decode(encode(facts)) == reference_plan` is the formal target for facts whose
public representative relations are supported and uniquely identifiable. Exact
decimal float transcription does not imply bitwise-exact physical reconstruction
after floating-point arithmetic and CIF export; that boundary uses the stated
geometric tolerances.

A measured 100/100 result supports this frozen cohort, not every crystal or every
English sentence. The parser accepts a finite controlled grammar, not unrestricted
prose. Supported synonyms may normalize to one canonical wording. Arbitrary LLM
generations are not covered. Readability and scientific usefulness are separate
from numerical round-trip success and have not received a human evaluation.
