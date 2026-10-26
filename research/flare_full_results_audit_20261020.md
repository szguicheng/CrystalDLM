# FLARE full-run results audit

Audited the synced outputs under `experiments/flare_crystal_full_20261019/` on 2026-10-20. The initial audit recounted per-attempt generation and GRACE score receipts. The 2026-10-20 correction pass subsequently regraded saved readback responses and refreshed canonical summaries locally and remotely, without model inference or GPU work.

Both arms have 700/700 generation receipts and 700/700 score receipts. Generation plan indices cover 0–699 exactly once per arm; each receipt’s request, attempt, flat index, and seed matches its manifest plan, and every score receipt joins to its generation seed. The AR and FLARE manifests share the same 700-attempt plan hash (`8acfff65…45ddd`), request set, and evaluation source/code fingerprints.

| Arm | Strict-valid structures | Requested-SG matches¹ | Strict-valid + requested SG | Condition-qualified proxy ≤80 meV/atom | Global-unique, novel, ≤80 meV/atom² | Scored | Score errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pure AR | 169/700 (24.1%) | 269/700 (38.4%) | 164/700 (23.4%) | 26/700 (3.7%) | 25/700 (3.6%) | 169 | 0 |
| FLARE | 121/700 (17.3%) | 209/700 (29.9%) | 117/700 (16.7%) | 25/700 (3.6%) | 24/700 (3.4%) | 121 | 0 |

¹ Requested-SG matches can include structures that fail strict structural validation, so the intersection is listed separately. ² Counted from each scoring receipt’s global-unique threshold set; the score is the MP2020-corrected GRACE energy minus the fixed MP2023 hull proxy, not DFT-confirmed ehull.

Each space-group request has 100 attempts. Per row, values are `strict-valid / requested-SG match / strict-valid and requested-SG / global-unique novel ≤80 meV`:

| Space group | Pure AR | FLARE |
|---:|---:|---:|
| 1 | 17 / 24 / 17 / 2 | 10 / 10 / 10 / 1 |
| 15 | 8 / 41 / 8 / 0 | 7 / 28 / 7 / 1 |
| 38 | 45 / 73 / 45 / 4 | 22 / 48 / 22 / 2 |
| 119 | 9 / 21 / 9 / 0 | 5 / 20 / 3 / 0 |
| 143 | 5 / 0 / 0 / 0 | 2 / 0 / 0 / 0 |
| 194 | 53 / 70 / 53 / 18 | 58 / 78 / 58 / 20 |
| 216 | 32 / 40 / 32 / 1 | 17 / 25 / 17 / 0 |

All 700 scoring receipts completed in each arm. The only statuses are `scored` (169 AR; 121 FLARE) and `skipped_invalid_structure` (531 AR; 579 FLARE); there are no scorer-error receipts. The proxy’s post-relax requested-condition match counts are 159 AR and 111 FLARE. These counts remain proxy results rather than DFT ehull measurements.

The paired setup uses the same Qwen3-8B source revision (`b968826…`), source checkpoint identity and trainable state, the 7,272-record training split and 900-record validation split (identical train/validation hashes and validation IDs), and training seed 42. Both completed 909 updates with 8 materials per update, fresh AdamW, learning rate `5e-5`, weight decay `0.01`, gradient clipping `1.0`, and BF16. The designed objective differs by arm. Training world size was 1 for AR and 2 for FLARE. Validation covered all 900 items: AR loss `0.21003`; FLARE total loss `0.30686` (`0.21094` clean AR + `0.09592` denoising).

Generation timing receipts report total-time median/p95 of `16.55/33.37 s` for AR and `20.20/58.52 s` for FLARE; crystal-generation median was `7.70 s` and `11.69 s`, respectively. Summing the 700 per-attempt totals gives `13,257 s` and `17,544 s`. Those sums are accumulated receipt durations across workers, not elapsed wall time; the synced artifacts do not provide a reliable end-to-end wall-clock duration.

The legacy formula grader rejected space-separated formulas. Historical AR and FLARE readbacks have been regraded through the shared `experiments/evaluation_common/readback.py` grader, with canonical summaries refreshed from each saved response and validated reference facts. The corrected formula counts are 169/169 AR and 121/121 FLARE; all four readback fields are correct for 162/169 and 115/121, respectively. These readback results are separate from the structural-validity and property-score counts above; the valid-structure counts (169/121) and final novel ≤80 meV counts (25/24) are unchanged.
