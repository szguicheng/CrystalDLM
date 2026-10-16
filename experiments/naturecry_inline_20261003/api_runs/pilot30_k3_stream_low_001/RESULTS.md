# NatureCry: Real K3 30-Case Pilot

Date: 2026-10-04 (Asia/Shanghai).

The real first-pass pilot completed all 30 cases. Four matched their original
structures: **4/30 (13.3%)**. The user's continuation rule requires 30/30 before
expanding to 100, so no expansion or further generation was performed.

## Frozen Scope And Call Configuration

The pilot uses the first 30 entries, in order, of the previously frozen 100-case
selection. Selection SHA-256:
`03527304a32e11c15414e0f368e72c694e754b3add76f660701437b6bf22d54a`.
There was no replacement on failure. These 30 cases include 20 nonidentity
frames and five correction cases containing 18 residual vectors.

The original facts, inventories, encoder/decoder prompts, public Wyckoff table,
geometry code, decoder gates, and strict scoring thresholds were unchanged.
`prepared_final/` passed fresh entry hash verification. The controller source
used for this pilot is snapshotted as `pilot_firstpass.py` in this directory.

Actual call configuration:

```json
{
  "endpoint": "https://llm.sciencepal.ai/v1/chat/completions",
  "model": "tokenhub/kimi-k3",
  "reasoning_effort": "low",
  "max_completion_tokens": 8192,
  "allowed_openai_params": ["reasoning_effort"],
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Temperature and response_format were omitted. SDK automatic retries were
disabled. The client read credentials from the user-populated project `.env`;
the file and key were not copied into experiment artifacts. Encoder and decoder
were separate requests. Decoder messages contained only the completed inline
document and its public-rule prompt; no source facts, CIF, or oracle mapping.

K3 native parameter constraints and streaming behavior were checked against the
[official model parameter reference](https://platform.kimi.ai/docs/api/models-overview)
and [streaming guide](https://platform.kimi.ai/docs/guide/utilize-the-streaming-output-feature-of-kimi-api).
The SciencePal gateway additionally required its request-level reasoning-effort
passthrough option, as recorded in the separate HTTP-400 diagnostic. No gateway
or service configuration was modified.

## Results

| Stage or outcome | Cases |
| --- | ---: |
| Selected and actually attempted | 30 |
| Encoder gate passed | 4 |
| Decoder gate passed | 4 |
| Original-structure match passed | 4 |
| Untagged numeric characters in encoder prose | 20 |
| Missing or malformed encoder references | 5 |
| Encoder markup/format failure | 1 |
| Transport failures in this pilot | 0 |

All 26 failures stopped before decoder execution. The conditional result among
encoder-gate passes is 4/4; it must not replace the primary denominator of 30 or
be generalized to untested decoder inputs.

Each decoded structure was written to a 14-significant-figure CIF, reparsed, and
compared with the original input. The hard requirements remained identical atom
count/composition, element-aware periodic coordinate error below 1e-6 angstrom,
and lattice-matrix component error below 1e-8 angstrom.

| Passed case | Coordinate error (angstrom) | Lattice error (angstrom) |
| --- | ---: | ---: |
| mp-1205515 | 0.0 | 0.0 |
| mp-1207068 | 0.0 | 0.0 |
| mp-646926 | 0.0 | 0.0 |
| mp-2582 | 0.0 | 0.0 |

The primary agent independently reopened and rescored all four CIFs, checked
their frozen source hashes, and obtained the same errors. Three of these four
cases have nonidentity frames; none has precision corrections. This pilot
therefore supplies no positive real-LLM residual-correction example.

## Failure Interpretation

The main bottleneck was encoder compliance with the inline protocol, before
geometric reconstruction. Some responses repeated a literal value beside its
reference as though the reference were a citation. Other numeric failures were
public SG/Wyckoff notation, fixed-site relations, formulas, or orbit IDs; the
frozen gate rejects numeric characters outside parameter tags regardless of
their role. These failures are not all evidence of lost numerical precision or
incorrect crystal geometry.

All five inventory failures had missing references, including malformed
`[[field]]` markers lacking `param:` and omitted orbit/correction element refs.
The raw outputs were preserved. No response was repaired, no missing field was
filled from an oracle, and no failed case was regenerated within this pilot.

The independent read-only failure audit is at
`research/NatureCry_pilot30_failure_audit_20261004.md`. It includes final counts,
specific omissions, and representative final-content examples. It did not use
reasoning text as the encoder answer. Arbitrary qualitative prose correctness
has not received a comprehensive independent semantic audit.

## Usage And Separate Transport Trials

This completed pilot made **30 encoder + 4 decoder = 34 client requests**.
There were 34 distinct response IDs, all reporting model `tokenhub/kimi-k3` and
finish_reason `stop`; all 34 supplied usage. Reported total tokens: **130158**.
Underlying model weights/version were not independently verified beyond the
requested and returned gateway alias. Dollar cost is not estimated.

| Earlier, separate activity | Recorded request attempts |
| --- | ---: |
| Interrupted legacy-parameter first pass | 7 |
| Official-field request rejected by gateway with HTTP 400 | 30 |
| Separate HTTP-400 contract diagnostic | 1 |
| Nonstreaming passthrough run, connection failures and one interrupted call | 14 |
| Max-effort stream ending before final answer | 1 |
| Completed low-effort streaming pilot | 34 |
| Total recorded/reserved client attempts | 87 |

Earlier calls and their outcomes are excluded from the 4/30 score. Interrupted
requests are conservatively counted as possibly issued. Some earlier streams
or disconnections had no terminal usage receipt, so this pilot's reported token
total is not a complete historical billing total. All historical directories
and the interruption receipt remain intact. The cumulative count is below the
previously authorized 200-attempt ceiling.

## Evidence And Stop Rule

Canonical evidence in this directory:

- `summary.json` and `records.json`: fixed denominator, per-case outcome, stages,
  request reconciliation, configuration, and usage.
- `requests/`: header-free payloads, redacted stream chunks, and reconstructed
  completed responses. Only final content entered the codec.
- `decoded_plans/` and `decoded_cifs/`: the four successful blind-decoder outputs.
- `pilot_firstpass.py`: source snapshot of the controller used for this run.

A local scan of all 117 files in this directory found no occurrence of the
configured credential. The result supports neither universal recovery nor a
training benefit. **The 30/30 qualification threshold was not met; stop at the
pilot.** Any prompt/protocol revision and new experiment should be separately
identified rather than overwriting this first-pass result.
