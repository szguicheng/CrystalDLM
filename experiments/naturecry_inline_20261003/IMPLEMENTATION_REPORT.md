# NatureCry Inline Implementation Report

**Status:** local implementation and offline preflight complete. The first-pass generation API was not called.

## Final Artifacts

- `codec.py`: typed inline inventory, placeholder binder, plain-prose/tag validation, exact value and metadata matching, strict decoder JSON/schema gate.
- `geometry.py`: public Wyckoff expansion, deterministic original-frame site IDs, residual lookup by ID, shared geometry executor, NaN-safe strict structure thresholds.
- `prepare.py`, `preflight.py`, `api_runner.py`: frozen source preparation, hash-pinned entry verification, offline shared-entry preflight, and the explicit first-pass runner.
- `decoder_schema.json`, `inventory_schema.json`, `rules.json`, `PROTOCOL.md`, and `prepared_final/prompts.json`: final protocol, execution order, prompt and schema artifacts.
- `prepared_final/`: final case facts/inventories and isolated preflight oracle plans/residual maps. The decoder request builder reads only the bound inline document.
- `preflight/final_002/`: final per-case records, decoded plans, 14-significant-figure CIFs, summary, and a deterministic binding example.
- `source_hashes.json` and `manifest.json` under `prepared_final/`: exact source, input, facts, inventory, runtime, prompt, schema, protocol, rules, oracle-plan, and mapping hashes.

Earlier `prepared/`, `prepared_v2/`, `preflight/run_001/`, and `preflight/final_001/` artifacts were preserved. `prepared_final/` and `preflight/final_002/` are the hash-consistent final pair.

## Offline Preflight

The unchanged selection contains 100 cases; selection SHA-256 is `03527304a32e11c15414e0f368e72c694e754b3add76f660701437b6bf22d54a`. Preparation verified all 100 input hashes and retained all cases, including the audited 71 nonidentity frames, 17 correction cases, and 114 corrected atoms.

Entry verification recorded matching expected/verified counts for selection files (1/1 each), selected-dataset hash reference (1/1), frozen CIFs (100/100), source facts (100/100), inventories (100/100), runtime code/protocol/rules (12/12), prompt/schema constants (3/3), prepared prompt file (1/1), oracle plans (100/100), and legacy residual maps (100/100).

All 100 deterministic fixtures passed the same decoder schema gate and `execute_geometry_plan` used by the runtime. Each result was written as a 14-significant-figure CIF, reparsed, and compared to the frozen source with the existing strict thresholds: coordinate error `<1e-6 A`, lattice-matrix component error `<1e-8 A`. Maximum errors were `9.381384558082572e-16 A` and `3.019806626980426e-14 A`, respectively.

This 100/100 result is an offline deterministic oracle/preflight result, not an LLM reconstruction rate or semantic-language result. API request counts are encoder 0, decoder 0, total 0.

## Runner Checks

The API runner requires `--execute-api` and `SCIENCEPAL_API_KEY`, hard-caps requests at 200, permits at most one encoder and one decoder attempt per case, makes no retries or repair calls, and does not follow redirects. Encoder failure skips that case's decoder. Requests and responses are stored without headers, known-key strings are redacted, and every failure remains in the N=100 report. Non-`stop` completion reasons fail the case. Decoded plans and CIFs are saved; scoring reparses the CIF.

The offline fake-transport test injected one encoder transport failure with a one-request budget. It verified a reconciled count of one attempted request, zero decoder attempts, all 100 cases in the summary, and no credential string or request headers in output. A separate fake truncated response with `finish_reason=length` was rejected and redacted. Running the CLI without `--execute-api` returned exit code 2 with `network disabled`; no live request occurred.

Default post-review command, when a first-pass API run is separately intended:

```bash
rtk proxy research/plaid_20260930/.venv/bin/python experiments/naturecry_inline_20261003/api_runner.py --execute-api
```

It writes to `api_runs/firstpass_001`; it refuses to overwrite existing output. The existing credential entry point is the `SCIENCEPAL_API_KEY` environment variable. It was not configured during this run.

## Semantic Boundary

The encoder prompt requires facts-grounded spatial language, forbids invented bonding/coordination/stability claims, preserves small inequalities, and describes residuals as real deviations. The binder rejects untagged Unicode numeric characters and common CIF/JSON/key-value/Markdown/table structures, while keeping free prose and escaping typed Hall strings safely.

These gates establish exact tagged field coverage and value/type/unit/frame matches. They do not prove that arbitrary qualitative spatial sentences are true. The decoder prompt asks the model to detect conflicts between prose, tags, and public rules, but no independent semantic prose audit has been performed. Any actual first-pass encoder prose needs independent review before interpreting LLM results.

## Verification

Combined command:

```bash
rtk proxy research/plaid_20260930/.venv/bin/python -B -m pytest -p no:cacheprovider -q experiments/naturecry_inline_20261003/test*.py
```

Result: **14 passed**. The run emitted 536 spglib deprecation warnings from the installed dependency; there were no test failures. The geometry agent had separately reported 6/6 geometry tests passing after correcting its supercell matrix to `diag(2,1,1)`; the combined run also passed them.

Final runtime hashes and all 100 source CIF hashes are recorded in `prepared_final/source_hashes.json`. Core final hashes:

- selection: `03527304a32e11c15414e0f368e72c694e754b3add76f660701437b6bf22d54a`
- `geometry.py`: `026d5c882c0e24b8802f9cf0f25415353cd3254e9bde45a6f1693396a72c04a2`
- `codec.py`: `8b8a4f11a949000e2c3f29b787676a6147a776946c4f7c9f272c1f33bfa110d9`
- `api_runner.py`: `2973ac9aef70dbd0f98257608227f9ebf645a688d8ce7f90cc8edfcd86e9716d`
- `PROTOCOL.md`: `89417bb698ebc8a3294aea81d2f0165aa01c823c1c77c6ff67e186f6096bac7c`
