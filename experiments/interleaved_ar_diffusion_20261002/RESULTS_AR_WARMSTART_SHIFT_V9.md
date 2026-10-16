# V9 AR warm-start / predictor-alignment continuation results

Slurm job **601979** (`predictor_align_1024_003`) completed on `gpu-a100-04`: `COMPLETED/00:49:57/0:0`. The final summary is complete. The run inherited the completed AR and same-position hybrid training from job 601858; it replayed the same-position hybrid evaluation and trained only the shifted-predictor hybrid. It did not use `test256`.

All three arms share V8 AR initial fingerprint `7567d5dd…b21c85`, seed 17, data manifest SHA `e3a5a86e…c0004`, and the same 16,384-row exposure stream (`d53173eb…c24f6f`, 1,024 materials once each, 8G/4E/4Q). The reader remains `named_axis_v2`, physical audit `orbit_membership_v3`, and the requested sampling/noise settings remained fixed.

| Arm | Training provenance | Offset | Final fingerprint | Core G / Qref / Eref / closed loop / flow / swap | Extra reading | Old8 G / Eref / closed loop / swap |
|---|---|---:|---|---|---:|---|
| AR control | Job 601858, 1,024 updates, inherited | 0 | `45684803…f036242` | 51 / 128 / 113 / 47 / 51 / 126 | 97/128 | 6 / 8 / 6 / 8 |
| Hybrid same-position | Job 601858, 1,024 updates, inherited checkpoint; full evaluation replayed | 0 | `b5ed6bff…91b0` | 0 / 128 / 0 / 0 / 0 / 122 | 118/128 | 0 / 0 / 0 / 7 |
| Hybrid shifted predictor | Job 601979, 1,024 updates from the shared V8 AR start | 1 | `f784f004…b8c1c` | 0 / 128 / 0 / 0 / 0 / 122 | 122/128 | 0 / 0 / 0 / 7 |

The offset-1 run did not improve G or closed-loop success over the same-position hybrid: paired across the same 128 validation IDs, G had **0 gains / 0 losses** and closed loop had **0 gains / 0 losses**. Both hybrid arms had G=0/128 and closed loop=0/128, so no valid generated crystal entered the actual read/edit flow. The shift arm’s extra-reading score was four higher (122 versus 118); the full validation denominator and every failed attempt remain included.

Both validation strata are reported separately:

| Arm | Stratum | n | G | Qref | Eref | Closed loop | Flow complete | Swap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AR | original64 | 64 | 22 | 64 | 59 | 20 | 22 | 63 |
| AR | later64 | 64 | 29 | 64 | 54 | 27 | 29 | 63 |
| Hybrid same | original64 | 64 | 0 | 64 | 0 | 0 | 0 | 61 |
| Hybrid same | later64 | 64 | 0 | 64 | 0 | 0 | 0 | 61 |
| Hybrid shift | original64 | 64 | 0 | 64 | 0 | 0 | 0 | 62 |
| Hybrid shift | later64 | 64 | 0 | 64 | 0 | 0 | 0 | 60 |

The AR arm’s repaired-scorer recount was run from the inherited raw data; its core, extra-reading, and old8 counts matched the stored AR counts. The two hybrid arms each read all 128 core IDs, all 128 extra-reading IDs, and all eight old8 controls. Their offset-aware checkpoints were strictly restored and checked against their recorded fingerprints. The Hsame evaluation was replayed from its completed offset-0 checkpoint after the 601858 scorer exception; the first 66 replayed records match the predecessor’s scientific payload exactly. The comparison recorded equal validation IDs, `first_mismatch=null`, no forward-call differences, and raw hashes `420f3914…6c9e17` (prior 66) and `87b024e7…c1eaa7a1` (replayed 128). Only timing fields changed.

Failure stages were kept on the full denominator:

| Arm | Generation call/decode failures | Returned crystals failing physical audit | Reference-read call failures | Reference edits: call failures / incorrect grades | Actual reads/edits attempted | No-crystal read call failures |
|---|---:|---:|---:|---:|---:|---:|
| AR | 0 | 77 | 0 | 0 / 15 | 51 / 51 | 0 |
| Hybrid same | 79 (48 END-boundary, 31 PAD-boundary) | 49 | 0 | 58 (50 END, 8 PAD) / 70 | 0 / 0 | 1 |
| Hybrid shift | 89 (63 END-boundary, 26 PAD-boundary) | 39 | 0 | 54 (44 END, 10 PAD) / 74 | 0 / 0 | 6 |

The audit reason labels can overlap within one invalid generated structure. Swap reading calls succeeded on all 128 inputs in each arm; swap success counts above show the factual-match results. Hsame and Hshift `stage0_passed=false`.

Runs 601839 and 601858 remain separate failed executions, not algorithm results. Job 601839 was cancelled after a CPU-placement error before any validation. Job 601858 completed AR and Hsame training, then stopped during Hsame evaluation when an invalid numeric token made the scorer raise. Its raw outputs and checkpoints remain under `predictor_align_1024_002/`; job 601979 used its valid AR/Hsame checkpoints, preserved the 66 existing Hsame raw rows, and completed the prescribed continuation. The scorer fix marks only an unconvertible lattice-length field false, without changing raw text or tolerance.

This remains a single-seed, reused-validation mechanism pilot. The same 128 validation and eight old8 materials were used in earlier runs; this result does not establish a general causal effect. `test256` was not read, trained on, or used for selection. See [the fixed-stage continuation protocol](PROTOCOL_AR_WARMSTART_SHIFT_V9.md), [the 601979 summary](predictor_align_1024_003/summary.json), and [the Hsame 66-row replay comparison](predictor_align_1024_003/hybrid_same/validation_128/prefix_replay_comparison.json).
