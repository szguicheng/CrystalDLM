## Material Passport

- Artifact: `pilot002_failure_audit`
- Representation: `relational_cnl_v1`
- Status: `ANALYZED` from the four `pilot_002` evaluation JSONL files; no GPU run or new sampling
- Denominator: 48 G rows and 48 `E_reference` rows, 8 successes and 40 failures for each task

Hybrid checkpoint reload is verified in `hybrid_shift` mode at prediction offset 1. AR uses offset 0. Both arms share initial fingerprint `6bffa202…d9d1429`; the run completed all 512 Hybrid updates.

| G split | n | Pass | END count | PAD boundary | CNL parse | Condition |
|---|---:|---:|---:|---:|---:|---:|
| AR seen8 | 8 | 8 | 0 | 0 | 0 | 0 |
| AR heldout16 | 16 | 0 | 0 | 0 | 6 | 10 |
| Hybrid seen8 | 8 | 0 | 6 | 2 | 0 | 0 |
| Hybrid heldout16 | 16 | 0 | 13 | 2 | 1 | 0 |
| **Total** | **48** | **8** | **19** | **4** | **7** | **10** |

| `E_reference` split | n | Pass | END count | PAD boundary | CNL parse | Condition | Geometry | Physical | Length only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AR seen8 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| AR heldout16 | 16 | 0 | 0 | 0 | 4 | 3 | 2 | 1 | 6 |
| Hybrid seen8 | 8 | 0 | 5 | 2 | 1 | 0 | 0 | 0 | 0 |
| Hybrid heldout16 | 16 | 0 | 15 | 1 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **48** | **8** | **20** | **3** | **5** | **3** | **2** | **1** | **6** |

The counts are mutually exclusive primary causes. For E grading, six AR heldout rows are length-only; two other rows also have `lengths_exact=false` but are assigned to condition (one) or physical audit (one), so there are 8 length-mismatched E rows in total. G has zero trigger, length, geometry, physical, and program-exception failures; E has zero trigger and program-exception failures. Hybrid's 23 `generation_error` and 23 `edit_error` rows are expected `unpack_block` validation rejections, not run crashes. `Q2` exposure is a separate gap and is not used to explain G/E.

AR heldout G's ten condition failures are seven orbit-definition selection failures, one missing relational orbit, and two Hall-setting mismatches. The six CNL parse failures are five invalid/unknown element names and one missing composition. AR heldout E has four malformed CNL sentences, two orbit-definition condition failures, two geometry expansion composition mismatches, one minimum-pair-distance failure below 0.5 Å, and six length-only mismatches.

Representative raw classifications: Hybrid `anchor_Tb8Mg4` G reports “crystal block requires exactly one END”; Hybrid `mp-1205515` G reports “invalid crystal PAD boundary”; Hybrid heldout `mp-1095422` G returns a block but CNL decode reports a missing/malformed input-cell sentence. AR heldout `mp-1019259` G fails the orbit-definition condition. AR heldout `mp-1079976` E fails geometry expansion (`expected Ge4Ni2Tm2`, observed `Ge2Ni2Tm2`); this is a new replay observation because the predecessor's raw output for that row was missing. AR heldout `mp-1225818` E fails physical audit at minimum pair distance below 0.5 Å.

For Hybrid boundary failures, JSONL rows contain an error object without structured `value.tokens`; the exact token-ID sequence is unavailable. Counts above use only the stored validator error class. The one Hybrid G CNL-decode failure and one Hybrid E CNL-decode failure do retain a 2560-token array because generation/unpacking returned successfully. `pilot_002` AR heldout replay has 16 raw token arrays; only two predecessor heldout raw rows existed, so the replay must not be presented as recovered predecessor output.

Hybrid mixed loss fell from 11.1574 at update 1 to 0.2539 at update 512, while update-512 task losses were G 0.9679, E 0.0475, and Q1 0.000043. The loss averages token prediction across tasks; Hybrid predicts masked crystal positions and weights PAD at 0.1. It does not report separate body/END/PAD accuracy or establish free-generation success.

The original `pilot_002` training source snapshot has no explicit `noise_step` on G/E examples, so `model.pack` draws uniformly from 1..32. A full-mask example therefore had theoretical probability 1/32; 1,024 G exposures imply expectation 32, but the curve does not log per-example mask steps, so the realized count is unknown.

`model.py:47-53` defines the one-END and PAD-suffix validation that classifies the Hybrid boundary failures. `model.py:109-143` constructs body+END+PAD targets and applies the PAD loss weight; `model.py:212-250` samples Hybrid mask positions and validates the completed block. The frozen runner's `update_examples` (`q2_coverage_preparation/source_snapshot/run.py:228-239`) omits explicit `noise_step` for G/E, and its train loop (`:602-612`) passes those targets to `model.loss`; `model.py:122-128` therefore selects k uniformly from 1..32. `score.py:63-145` separates decode, geometry, and physical audit failures; `score.py:263-324` grades edited lengths, plan fields, scaled geometry, and physical validity.

The smallest follow-up diagnostic is teacher-forced inference on the same Hybrid checkpoint and the same eight seen cases, without optimizer updates: compare argmax accuracy for masked body, END, and PAD positions at full-mask and intermediate-mask levels, then count per-sequence argmax ENDs and exact PAD-boundary validity. This isolates boundary learning from reverse-unmask error propagation; the current evidence does not separate those causes.
