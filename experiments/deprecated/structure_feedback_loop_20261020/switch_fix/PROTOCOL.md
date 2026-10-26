# Retry input serialization fix

User authorized diagnosis and repair of the missing AR-to-diffusion switch on 2026-10-20. The frozen model, original requests, temperature, token caps, B=4 diffusion sampler, strict auditor and no-GRACE scope remain unchanged. No training and no forced START token.

## Diagnosis

The original runner appends checker feedback as a second user turn after an assistant-generated crystal. Training contains G generation followed by Q factual readback; it does not supervise a second G after feedback. Fourteen deterministic diagnostic contexts (the first two initial failures per space group) exactly replayed the missing-trigger failures.

| Probe | Natural START / 14 |
|---|---:|
| Original FLARE retry context, cap256 | 0 |
| Matched pure-AR checkpoint, same context/cap | 0 |
| FLARE, only raise cap to768 | 1 |
| Pure AR, only raise cap to768 | 0 |
| FLARE, repeat original G request after history | 0 |
| FLARE, concise feedback after history | 0 |
| FLARE, add EOS between history and feedback | 0 |
| FLARE, remove old crystal control markers only | 2 |
| FLARE, fresh single-user G request with failed structure as input data | 14 |

The last variant keeps the original structural/property request and the actual failed crystal, includes the same checker feedback, and ends with the normal `Assistant:` generation prefix. The previous crystal is decoded without modality delimiters and placed in an explicitly labeled input-data section. Previous generated explanations are omitted when a crystal is available; when no crystal was generated, the failed AR response is retained as input data. Raw evidence is unchanged.

This is a context-contract repair, not evidence that shared parameters inherently prevent modality switching. The fresh-request result does not isolate every feature of the serialization change; the marker-only and prompt-only ablations did not reproduce its recovery.

## Regression and outcome verification

- Verify `context.py` reconstructs exactly the 14 successful probe contexts, without appending START or any reference answer.
- Keep the 14 diagnostic groups separate from the remaining 56 groups in reporting. The latter contain 47 initially failed groups and 9 initial successes.
- Reuse the original shared first candidate and independent-retry baseline for all 70 groups, whose exact source files and hashes are recorded. These paths are unchanged by the fix and use identical frozen weights, sampling and seeds. Old receipt provenance is retained, never relabeled as newly generated.
- Generate only the corrected feedback retries, at most two after the reused first candidate, with the original paired retry seeds. Stop on strict validity plus agreement with the original requested space group. A missing START remains a failure; do not force a transition.
- Save every new attempt atomically. Resume verifies fixed code, parent run, per-group source hashes, checkpoint state and attempt order.
- Report natural switching on actual retries, success within three candidates, paired outcomes against direct retries, and the held-out 56-group subset. Structural success is separate from ehull, which is unevaluated.

Use project mamba on LiuGangGPU, two selected idle H20 GPUs, project files only. Previous runs and weights remain intact.
