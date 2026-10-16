# NatureCry Inline Protocol v1

## Scope

This experiment keeps the frozen 100-case selection and adds typed inline values to an LLM-written spatial narrative. It does not train a model. The supported input domain is the audited cohort: 1-32 atoms, full occupancy, ordinary elements, no extra site or structure properties, with the source audit's original frame, rounding, and residual behavior.

The evidence boundary is deliberate: preparation and offline preflight validate facts, exact parameter binding, decoder JSON shape, public geometry execution, and strict source matching. They do not test whether an LLM understands natural spatial language. A legal `<param>` tag proves only that its name, type, unit, frame, and numeric value passed the exact inventory gate. It does not prove that surrounding prose describes the crystal correctly. Any first-pass encoder prose must be reviewed for unsupported or false spatial claims before results are interpreted.

## Encoder Input and Output

The encoder receives the prepared crystallographic facts and typed inventory. It chooses the narrative wording and order. Each inventory reference must occur exactly once in the prose as `[[param:NAME]]`; the local binder replaces that reference at the same position with:

```xml
<param name="cell.a" type="float64" unit="angstrom" frame="input">7.79230124</param>
```

The tag body is compact JSON for its declared type. Floating-point values use Python's float64 shortest-roundtrip decimal representation. Strings are XML-escaped. Attributes are fixed to `name`, `type`, `unit`, and `frame`.

Facts include the six input-cell parameters, space-group number and literal Hall symbol, input atom count and composition, each independent orbit's stable ID/element/Wyckoff label/free parameters, the explicit frame matrix and shift, `round_digits` including explicit `null`, and every correction's stable site ID/element/full three-component delta. There is no full atom table, CIF, or trailing fact appendix in the decoder document.

Encoder gates reject missing, repeated, and unknown refs; malformed, injected, nested, or duplicate tags/attributes; missing inventory tags; wrong types, units, frames, shapes, and values; any untagged Unicode numeric character; and common CIF, JSON, key-value, Markdown, or tabular appendix structure. This keeps exact numerals in tags while leaving natural prose free to describe relationships in words. These checks cannot prove that a qualitative sentence is true or rule out unsupported bonding/coordination/stability claims expressed in words. The decoder prompt asks for contradictions between prose, tags, and public rules to be returned as `conflict`, but no independent semantic prose audit is run locally. First-pass prose needs a separate review.

## Blind Decoder

The decoder receives only its fixed prompt and the completed inline document. The request builder accepts one argument, the inline document; it does not load source facts, source CIFs, residual-index maps, or oracle plans for that request. The decoder returns one unfenced JSON object with exactly:

- `status`, `space_group_number`, `hall_symbol`, `lattice`, `input_atom_count`, and `composition`
- `orbits`: `id`, `element`, `wyckoff`, and exact public-table-order `parameters`
- `frame_transform`: both `matrix` and `origin_shift`
- `round_digits` and `precision_corrections`: each with `site_id`, `element`, and three-component `delta`

Missing round/correction fields are not defaulted. Equivalent Wyckoff representatives are not refit. The parser rejects Markdown fences, duplicate JSON keys, NaN/Infinity, trailing text, unknown keys, booleans as numbers, numeric strings, fractional counts, duplicate orbit IDs/correction targets, wrong shapes, and plan values that do not match the inline tags. Hall, space group, and Wyckoff labels must agree with the frozen public table. Unknown correction IDs and element mismatches fail during public geometry execution.

## Public Geometry Rules

The frozen order is recorded in `rules.json` and implemented by `geometry.py`:

1. Expand each independent orbit in stable orbit-ID order, using its exact free parameters in public-table axis order and the table's operation order.
2. Bound image translations by `low=floor(sum(min(P,0),axis=1)+s-1)` and `high=ceil(sum(max(P,0),axis=1)+s+1)`, inclusive, with the last translation axis changing fastest. Transform by `q_input = inverse(P) (q_standard + image - s)`, retain boundary candidates in `[-1e-9, 1+1e-9]`, apply modulo one, snap values within `1e-12` of one to zero, and deduplicate first-retained sites by species and periodic max-component distance `<1e-8`.
3. Assign IDs as `<orbit_id>_<1-based retained ordinal within orbit>`.
4. Round fractional coordinates with `numpy.round` (ties to even) only when `round_digits` is present, then apply modulo one.
5. Resolve corrections only by stable ID, check element and target uniqueness, add the complete input-frame fractional delta, and apply modulo one. Do not reorder or deduplicate after correction.
6. Check composition and original input atom count before constructing the structure.

Preflight preparation may translate legacy residual indices into these IDs. That map and the source-derived oracle plan live only under `prepared_final/preflight_only/`; neither is read by the API runner's decoder message builder.

## Scoring and Failures

The strict match uses the same input cell and composition, element-aware periodic coordinate matching, coordinate error `<1e-6 A`, and lattice-matrix component error `<1e-8 A`. API outputs are saved as a plan and a 14-significant-figure CIF; scoring reparses that CIF before comparison.

The first-pass denominator is always 100. Encoder transport/format/inventory failures, decoder transport/format/schema failures, geometry failures, and original-structure numerical mismatches remain separate failure stages. An encoder failure skips that case's decoder request. There are no retries, repair calls, or oracle repairs. Every attempted request is reconciled against the per-case ledger; HTTP redirects are not followed. Only returned token usage is recorded, and cost is left unset without a verified price.

## Local Commands

Use the repository's existing environment; no software is installed. The following commands are local-only and never call a generation API:

```bash
rtk proxy research/plaid_20260930/.venv/bin/python experiments/naturecry_inline_20261003/prepare.py
rtk proxy research/plaid_20260930/.venv/bin/python experiments/naturecry_inline_20261003/preflight.py
rtk proxy research/plaid_20260930/.venv/bin/python -B -m pytest -p no:cacheprovider -q experiments/naturecry_inline_20261003/test*.py
```

Preparation and preflight refuse to overwrite existing outputs. The API runner defaults to `prepared_final` and `api_runs/firstpass_001`, refuses to run without `--execute-api`, requires the existing `SCIENCEPAL_API_KEY` environment variable, and caps requests at 200. After human review of the protocol and first-pass prose handling, the authorized API command is:

```bash
rtk proxy research/plaid_20260930/.venv/bin/python experiments/naturecry_inline_20261003/api_runner.py --execute-api
```

The runner does not prompt for or print credentials. It writes payloads without headers, redacts the known key from persisted strings, records each failed request without retry, and continues independent cases while preserving all 100 cases in its report.
