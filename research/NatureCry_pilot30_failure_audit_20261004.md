# NatureCry K3 Low Pilot Failure Audit

## Snapshot

This is a read-only snapshot of `experiments/naturecry_inline_20261003/api_runs/pilot30_k3_stream_low_001`. At capture, `records.json` contained 19 completed case records and was stable during the read (`SHA-256: 6f59559468712fdb0d84e88c57dc1f29a1533631226294a430ffb31c3eb84c49`). There was no final `summary.json`, so this is not a full-pilot result. The request directory had 22 `.request.json` files and 21 `.response.json` files; the 21 attempts in the 19 records reconcile as 19 encoder plus 2 decoder requests, with one additional request lacking a response/record at the snapshot. That unmatched request is left unclassified.

Record outcomes at the cutoff:

| Result | Cases |
| --- | ---: |
| `encoder_untagged_number` | 12 |
| `encoder_inventory` | 4 |
| `encoder_format` | 1 |
| Strict source-match pass | 2 |

Each recorded case had at most one encoder call; the two passes each had one decoder call. Across all 19 records, the encoder system message hash in the request matches both `prepared_final/prompts.json#/encoder` and the corresponding `encoder_prompt_sha256` in `records.json` (19/19). No reasoning stream files were read or used for this audit; excerpts below are from `*.encoder.response.json#/choices/0/message/content` only.

## REF And Format Findings

The inline protocol recognizes only `[[param:<field-name>]]` (`codec.py:44`), requires every inventory name exactly once (`codec.py:201` and `codec.py:203`), and rejects any numeric character left in plain prose (`codec.py:188` and `codec.py:191`). Numeric failures therefore indicate format noncompliance; they do not by themselves establish that a raw input value was disclosed.

Across the 12 `encoder_untagged_number` records, the recognized REF set was complete and unique: no missing, duplicated, or unknown REF names. The failing characters were outside those references. The four inventory failures had no duplicated or unknown recognized references, but did have these omissions:

| Case | Inventory result | Evidence |
| --- | --- | --- |
| `anchor_Li8V4Ni4O16` | 0/36 recognized; all 36 inventory names are missing. The prose uses `[[cell.a]]`-style placeholders without the required `param:` prefix. | `requests/anchor_Li8V4Ni4O16.encoder.response.json#/choices/0/message/content`; expected names in `prepared_final/cases/anchor_Li8V4Ni4O16/inventory.json#/fields` |
| `mp-21371` | 0/27 recognized; all 27 inventory names are missing. It similarly uses `[[orbit.o000.id]]` rather than `[[param:orbit.o000.id]]`. | `requests/mp-21371.encoder.response.json#/choices/0/message/content`; expected names in `prepared_final/cases/mp-21371/inventory.json#/fields` |
| `mp-1173947` | 66/87 recognized; exactly 21 missing: `orbit.o000.element` through `orbit.o010.element`, and `correction.c000.element` through `correction.c009.element`. No duplicates or unknown names. | `records.json#/12`; response content and `prepared_final/cases/mp-1173947/inventory.json#/fields` |
| `mp-19419` | Three missing: `orbit.o000.element`, `orbit.o001.element`, `orbit.o002.element`. No duplicates or unknown names. The prose also includes the raw composition formula `LiFeO2`; inventory validation fires before the numeric-prose gate. | `records.json#/15`; `requests/mp-19419.encoder.response.json#/choices/0/message/content` |

The remaining `encoder_format` case, `mp-1079976`, had all 31 expected references exactly once and failed the markup check (`GateError: Encoder prose contains markup`; `codec.py:199`). The response also mentions raw expansion identifiers `o002` and `o003` outside parameter references.

## Numeric And Identifier Audit

The numeric gate treats source values, public labels, and conventional crystallographic notation alike. Inspection of final content against the frozen inventories gives these distinctions:

- Clear literal copies of input geometry appear in `mp-1095422`: `5.47073346` for `cell.a` and `cell.b`, `7.552314` for `cell.c`, `90.0` for `cell.alpha` and `cell.beta`, `119.99999349000001` for `cell.gamma`, and `0.14765799999999993` for `orbit.o002.parameters`. The sentence repeats some of these values next to their typed references. This is a format failure and a raw-value appearance, not a successful decoded geometry. Evidence: `requests/mp-1095422.encoder.response.json#/choices/0/message/content` and `prepared_final/cases/mp-1095422/inventory.json#/fields`.
- `mp-1079976` writes `90` for right angles, matching the prepared `cell.alpha` and `cell.beta` values. `mp-4047` writes public Wyckoff multiplicities/labels `8c`, `4a`, and `24e`; its `60`-degree statement rounds the prepared angles near `59.999999999999986`. These are still rejected by the plain-prose numeric gate, but they are not equivalent to copying free positional parameters.
- `mp-1219591` writes public space-group number `74` beside its typed reference, and repeats raw orbit IDs `o000` through `o005` outside references. Its raw orbit labels (`16j`, `4a`, `4d`, `4e`) are public-table labels. The literal `18` atom count also matches the prepared input count. Evidence: `requests/mp-1219591.encoder.response.json#/choices/0/message/content` and `prepared_final/cases/mp-1219591/inventory.json#/fields`.
- `mp-19419` includes `LiFeO2`; this exposes composition in ordinary prose, not lattice or free-coordinate values. It is a format discrepancy, while the specific inventory failure is the three absent orbit-element references above.
- Other observed numerals are often public Wyckoff multiplicities or fixed-site relations, for example `4c` in `anchor_Tb8Mg4`, `0, 0, 0` and `0, 0, z` in `mp-30009`/`mp-1019259`, and `1/3, 2/3, z` in `mp-1205902`. `mp-1188285`, `mp-1205902`, and `mp-757163` describe the conventional 120-degree hexagonal metric; those statements do not copy the near-120-degree source angle exactly. These remain encoder-gate failures, not proof of private numeric leakage.

## Representative Final-Content Excerpts

Each excerpt is one sentence from the final encoder content, not from a reasoning chunk.

1. `mp-21371` shows the missing `param:` marker: “The vanadium atoms form orbit `[[orbit.o000.id]]`, with element assignment `[[orbit.o000.element]]`, Wyckoff label `[[orbit.o000.wyckoff]]`, and independent standard fractional parameters `[[orbit.o000.parameters]]`.”
2. `mp-1173947` has element prose but omits the typed element references: “Two lithium ions occupy fixed inversion centers and therefore have no free positional parameters: orbit `[[param:orbit.o000.id]]` is Li on Wyckoff site `[[param:orbit.o000.wyckoff]]`, and orbit `[[param:orbit.o001.id]]` is Li on site `[[param:orbit.o001.wyckoff]]`.”
3. `mp-1095422` repeats source lattice values outside the references: “The input cell has lattice parameters a and b both equal to 5.47073346 angstroms `[[param:cell.a]][[param:cell.b]]`, with c = 7.552314 angstroms `[[param:cell.c]]`.”
4. `mp-1219591` repeats a public SG number beside its typed reference: “In the conventional public description the structure is assigned to orthorhombic Imma, space-group number 74 `[[param:space_group.number]]`, with Hall symbol “-I 2b 2” `[[param:space_group.hall]]`.”
5. `mp-19419` includes a raw composition formula: “This primitive cell contains `[[param:input.atom_count]]` atoms in total, comprising `[[param:composition.Li]]`, `[[param:composition.Fe]]`, and `[[param:composition.O]]`, consistent with the formula LiFeO2.”

## Decoder Source Comparison (19-Record Snapshot)

Two cases completed decoder and strict original-structure comparison:

| Case | Status | Atoms | Coordinate error | Lattice error | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| `mp-1205515` | pass | 10 | 0.0 Å | 0.0 Å | `records.json#/3` |
| `mp-1207068` | pass | 9 | 0.0 Å | 0.0 Å | `records.json#/6` |

The other 17 recorded cases failed before decoder execution, so that cutoff supported conclusions about prompt/hash integrity and encoder-format adherence, not a general claim about K3 geometry-decoding capability. The two exact matches were positive case-level evidence only; the final 30-case result is appended below.

## Final 30-Case Snapshot

The completed run summary at `experiments/naturecry_inline_20261003/api_runs/pilot30_k3_stream_low_001/summary.json` reports all 30 selected cases attempted, 30 encoder calls, 4 decoder calls, and 34 total calls, reconciled to case records with zero automatic retries. `transport_abort_reason` is null and no selected case was left unattempted. Prior attempts are accounted separately: 53 prior + 34 in this pilot = 87 global attempts, below the 200-attempt ceiling. The pilot score remains denominator 30; prior calls and their outputs were not included.

Final case outcomes:

| Result | Cases |
| --- | ---: |
| `encoder_untagged_number` | 20 |
| `encoder_inventory` | 5 |
| `encoder_format` | 1 |
| Strict source-match pass | 4 |

All five inventory failures have no duplicate or unknown REF names. Their exact omissions are:

| Case | Inventory result |
| --- | --- |
| `anchor_Li8V4Ni4O16` | 0/36 recognized; all 36 missing due to `[[field]]` placeholders lacking `param:` |
| `mp-1173947` | 66/87 recognized; missing 11 `orbit.o000.element` through `orbit.o010.element` and 10 `correction.c000.element` through `correction.c009.element` |
| `mp-21371` | 0/27 recognized; all 27 missing due to placeholders lacking `param:` |
| `mp-19419` | Missing `orbit.o000.element`, `orbit.o001.element`, and `orbit.o002.element`; raw formula `LiFeO2` also appears in prose |
| `mp-754971` | 33/39 recognized; missing `orbit.o000.element` through `orbit.o005.element` |

Evidence is in `records.json` entries for those cases, their `requests/<case>.encoder.response.json#/choices/0/message/content`, and `prepared_final/cases/<case>/inventory.json#/fields`. I also counted REF sets for the 20 `encoder_untagged_number` outputs: all 20 had the expected inventory exactly once, with no missing, duplicate, or unknown REF names. Their failures are untagged prose numerals, not REF accounting errors. The one `encoder_format` case remains `mp-1079976`: all 31 expected references occur once, but markup is rejected by the format gate.

The 19-record numeric/identifier examples above remain present in the final 30. They distinguish three classes that the current numeric gate intentionally treats the same way: public labels/relationships (`4c`, `8c`, `1/3`, `2/3`), source facts written literally (the exact lattice/orbit values in `mp-1095422`), and raw identifiers/formula text (`o000`-style identifiers and `LiFeO2`). The latter should be reported separately from literal source-coordinate matches; neither a public Wyckoff multiplicity nor a formula is evidence of geometry recovery. The final five inventory failures add missing REF cases; their omission list above is the actionable reference-accounting result.

The four final strict source-match records report zero coordinate and lattice error in `records.json`:

| Case | Atoms | Coordinate error | Lattice error | Record |
| --- | ---: | ---: | ---: | --- |
| `mp-1205515` | 10 | 0.0 Å | 0.0 Å | `records.json#/3` |
| `mp-1207068` | 9 | 0.0 Å | 0.0 Å | `records.json#/6` |
| `mp-646926` | 6 | 0.0 Å | 0.0 Å | `records.json#/22` |
| `mp-2582` | 6 | 0.0 Å | 0.0 Å | `records.json#/24` |

These are runner-recorded source-comparison metrics; this audit did not independently reopen or rescore the four CIFs. That independent verification is outside this report.

## Recommendations

1. In a future authorized protocol revision, keep the exact `[[param:<inventory-name>]]` form prominent in examples and explicitly require every listed inventory key once, including every orbit/correction `.element` field. Add contrasting examples for `[[param:orbit.o000.element]]` versus malformed `[[orbit.o000.element]]`.
2. Decide explicitly whether public SG/Wyckoff labels, fixed-site notation, formulas, and conventional angles may appear as plain prose. If the blanket numeric gate remains, instruct the encoder to keep all numeral-bearing values inside typed references; if public notation is allowed, revise the gate and its frozen hash-pinned protocol together rather than treating these outputs as raw-value leaks.
3. Keep decoder/geometric scoring separate from encoder format compliance. Here, 26/30 cases stop at encoder format gates and only four reach strict source comparison; do not interpret 4/30 as an isolated geometry-capability estimate.
4. The current continuation rule requires 30/30 before extending to 100. The observed 4/30 does not meet that threshold, so this snapshot supports stopping at the pilot boundary.
