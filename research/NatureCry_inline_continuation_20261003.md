# NatureCry inline continuation, 2026-10-03

Source requirements: `CrystalDLM_NatureCry_HANDOFF_20261003.txt`.

## Current State (2026-10-04)

Latest real experiment: the user narrowed the next run to 30 cases, requiring
30/30 before expansion. `api_runs/pilot30_k3_stream_low_001/` completed the first
30 frozen cases with K3 low-effort streaming: 4/30 strict matches. The other 26
failed encoder format/inventory gates; all four cases sent to the blind decoder
matched their original structures. No expansion to 100 was performed.

See `experiments/naturecry_inline_20261003/api_runs/pilot30_k3_stream_low_001/RESULTS.md`
and `research/NatureCry_pilot30_failure_audit_20261004.md`. This pilot made 34
requests and reported 130158 tokens. Separate earlier transport/contract trials
are preserved; total recorded/reserved client attempts are 87, not 34. Current
credentials are read from the user's project `.env` without copying key values.
The historical zero-API/credential-blocked notes below are superseded.

The user's final model choice is `gpt-6-luna/max` for implementation and review.
Local implementation, focused tests, frozen-cohort preflight, and independent
scoped review are complete. The historical notes below describe earlier states.

- Final preparation: `experiments/naturecry_inline_20261003/prepared_final/`.
- Final preflight: `experiments/naturecry_inline_20261003/preflight/final_002/`.
- Delivery report: `experiments/naturecry_inline_20261003/IMPLEMENTATION_REPORT.md`.
- Combined tests: 14 passed, with installed-spglib deprecation warnings.
- Offline preflight: 100/100, 100 unique cases and 100 written/reparsed CIFs;
  maximum coordinate error 9.381384558082572e-16 angstrom and maximum lattice
  matrix component error 3.019806626980426e-14 angstrom.
- Fresh post-interruption `verify_prepared(prepared_final, include_preflight=True)`
  passed all recorded hash categories. Archived v2/final001 are not current.
- Independent Luna/max review: Spec PASS, Quality PASS; original P1/P2 addressed,
  no new important findings. Review agent: `01a10257-2f3f-7231-95e6-07b39b9432f2`.

Real generation API requests remain zero. The current environment has no
`SCIENCEPAL_API_KEY`; the existing credential entry point is still needed before
the already-authorized first-pass experiment. The runner remains opt-in, capped
at 200 requests, and permits no retries or repair calls. No training was started.

The 100/100 result is deterministic offline protocol/geometry validation, not an
LLM recovery rate. Arbitrary qualitative prose has not received an independent
semantic audit. The encoder gate rejects untagged numbers and common structured
appendices, and the decoder prompt requires conflict checks; those do not prove
that every spatial sentence is true. Review actual encoder prose before drawing
semantic conclusions from a future run.

## Status

The requested Gemini implementation subagent was dispatched through the supported
`multi_agent_v1__spawn_agent` interface in this local project, with model
`antigravity/gemini-3.8-flash-high`, reasoning effort `high`, and no inherited
conversation. Agent `01a10226-01f0-7e02-a3cd-9431df02852e` failed with
`exceeded retry limit, last status: 429 Too Many Requests` and was closed.
The implementation directory `experiments/naturecry_inline_20261003/` was still
absent after failure. No implementation, tests, new preflight, or generation API
calls were completed. The historical V2 encryption problem was not the error
reported by this attempt.

A read-only GPT review was completed by agent
`01a10226-fcdb-7542-8b94-458972200d7a`, using `gpt-6.1-sol/high`.
No source files were changed by that review.

Pending user inputs: whether GPT implementation is permitted as a fallback,
and the existing credential entry point for SciencePal. The current shell has
no `SCIENCEPAL_API_KEY`. No credential values were requested in chat or retained.

The user subsequently requested another Gemini attempt after confirming that
the side chat worked. A second V1 Gemini/high implementation agent,
`01a1022d-36a5-7782-9fb0-26a9ff685d0a`, was dispatched with the same local scope
and the completed GPT review checklist. GPT implementation has not been
authorized. This dispatch is pending execution evidence and does not establish
that implementation or preflight has succeeded.

The second attempt subsequently created
`experiments/naturecry_inline_20261003/progress.json` in the local project.
This confirms local execution has started; implementation and preflight remain
in progress until verified from the actual artifacts.

The second agent later failed with the same exhausted-retry 429 error. Its
partial output includes `geometry.py` and `progress.json`. No `codec.py`, focused
tests, final shared-entry preflight, or API runner had been saved at inspection.
An ad hoc geometry prototype check in the agent transcript passed 100 cases,
but that duplicated scratch function is not the required shared public-entry
preflight and must not be reported as protocol acceptance. Generation API calls
remain zero. A GPT reviewer is examining the saved geometry module.

The user then explicitly selected `gpt-6-luna/max` for subagents. This supersedes
the prior Gemini-only implementation preference. A third, smaller Gemini attempt
(`01a1023d-50aa-7ba1-b041-7e91b6628604`) was stopped before saving its assigned
protocol files. Luna/max agent `01a1023f-4e9a-7861-b02e-54eb9590b60e` now owns
remaining implementation in the existing experiment directory. The SciencePal
credential entry point remains pending; no generation API requests have occurred.

The saved geometry module's independent review confirmed core expansion/frame/ID
and correction sequencing semantics. One comparator fix is required: use positive
strict acceptance conditions (`not error < tolerance`) so NaN cannot pass a
comparison. A synthetic NaN scoring reproduction exposed this difference from the
legacy comparator; it was not a frozen-cohort failure. This fix and its focused
check have been assigned to the Luna implementer. Integrated review is pending.

## Verified Local State

- Project root is not a Git repository.
- Existing Python environment imports numpy, pymatgen, and spglib.
- Frozen selection contains 100 cases and 100 unique IDs. All 100 input hashes
  match the manifest; no cases were replaced.
- Selection SHA-256: `03527304a32e11c15414e0f368e72c694e754b3add76f660701437b6bf22d54a`.
- Historical audit covers 1-32 atoms, 71 nonidentity frames, 17 correction cases,
  and 114 corrected atoms.
- The old LLM `run_001/summary.json` reports 84/100 strict matches, 198 attempted
  requests, 195 successful responses, and 343797 reported tokens.
- No existing NatureCry runner was found in the process list at inspection.

Read-only baseline source hashes:

| File | SHA-256 |
| --- | --- |
| `experiments/naturecry_llm_100_20261002/cohort.py` | `f8b47626030b77592a3e2105544576f1b2103bd77a4f9eb9e7d138416b19a1d8` |
| `experiments/naturecry_llm_pilot_20261002/pilot.py` | `0aad3088782cb38abf80949f439d915cef407075aaad3697cf6fd6fc3f626b81` |
| `experiments/explicit_full_mp20_20260923/codec/wyckoff_baseline.py` | `1ccdc829f301f1279f78751dc5138fcac1e67680dfdaebefdbf72c9ca3ba758a` |

## Preimplementation Review Obligations

These are requirements for the forthcoming implementation, not findings against
new inline code, which does not yet exist.

1. Decode exact free parameters in the public table's axis order. The legacy
   cohort executor refits representatives using a pseudoinverse and tolerant
   candidate selection. Equivalent representatives can change floating-point
   evaluation and expansion order. Do not allow arbitrary equivalent
   representatives in this inline protocol.
2. Define stable site IDs on retained original-cell sites, preserving orbit
   provenance. Freeze orbit ordering by ID, public table operation order,
   translation enumeration, boundary selection, and species-aware first-retained
   deduplication. A standard-operation index alone is insufficient for cell
   expansion or collapse.
3. Preserve the actual sequence: standard expansion, input-frame image
   selection/mod/dedup, round, mod, residual addition, mod. Preparation may map
   legacy residual indices to IDs; runtime must resolve IDs using only public
   execution. Validate unique correction targets and element agreement. Do not
   round, reorder, or deduplicate after applying corrections.
4. Encoder gates require exact inventory coverage and decimal value equality.
   Reject missing, duplicate, unknown, or leftover refs; injected/nested tags;
   duplicate or unknown attributes; incorrect type/unit/frame; and incomplete
   vectors. Escape Hall symbols and string values through structured serialization.
   Legal tags do not prove that surrounding spatial prose is correct.
5. JSON gates reject fences, duplicate keys, NaN/Infinity, unknown keys, wrong
   shapes, boolean integers, numeric strings, fractional counts, duplicate orbit
   IDs, and unknown correction refs. Validate Hall/space-group/Wyckoff agreement.
   Require rounding and correction fields explicitly rather than inferring them.
6. Local preflight and blind decoding must use the same executor. Keep facts,
   old-index mappings, and scoring inputs outside runtime geometry. Cover orbit
   and correction reordering, boundary/dedup behavior, malformed gates, and all
   100 frozen cases under the existing comparator: equal atom count/composition,
   lattice matrix error <1e-8 angstrom, coordinate error <1e-6 angstrom.

## Continuation

Resume implementation only through a user-allowed model route. Keep writes in
the new experiment directory and preserve old artifacts. Complete focused tests
and the frozen 100-case local preflight, then obtain an independent code review.
Only after those gates pass should the authorized first-pass runner execute up
to 100 encoder and 100 decoder requests, without retries or repairs. All failures
remain in the N=100 denominator. No training, GPU work, push, or publication.
