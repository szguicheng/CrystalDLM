# Phase 1–2 implementation record

Authority: handoff document and approved `00_recon_report.md`. Gate 1 approved on 2026-10-25 PDT; user explicitly changed training to three GPUs and then confirmed global batch 12 (three ranks × one example × four accumulation steps). Annotation stays paused. Stop at gate 2 before any training.

Work occurs on `codex/crystal-tasks-v1-phase12`; preserve all pre-existing migration changes. Only new `experiments/crystal_tasks_v1/` files and the authorized remote project dependency checkout/environment may change. Canonical task status is in claude-mem work_state, list `Crystal tasks v1 handoff execution`.

| Work | Owned files | Check |
| --- | --- | --- |
| Environment and configuration | setup receipt, configs, framework integration patch | pinned versions, pip check, real template/collator labels, tiny-model adapter parameter check |
| Understanding and loss metrics | evaluation/understanding.py, losses.py, associated tests | gold answers, isolated errors, formula equivalence, AR shift/token weighting, denoising denominators |
| Design metrics and old-code binding | evaluation/design.py, legacy.py, associated tests | strict audit reuse, SG mismatch, per-request properties, duplicate/invalid denominators, novelty and SM coverage |
| Executable evaluation and provenance | evaluation CLI, reference loader, GRACE bridge, CPU acceptance | shared record schema, complete reference counts, persisted run_meta and per-row results |
| Integration review | metric specification and gate2 report | remote tests, full Understanding gold grading, actual dataset fixtures, fixed-data hashes preserved |

Interfaces: attempts contain attempt_id, seed, request{space_group,ehull_upper}, output, and optional property_receipt from the frozen GRACE scorer. Reference banks contain pymatgen Structures. Understanding rows contain response and reference{formula,space_group,cell_atom_count,independent_orbit_count}. No component edits frozen datasets or archived source.

Resolved conflict: global batch 8 is not divisible by three ranks. Owner chose global batch 12; keep both design branches identical. Design full JSONL/cutoff remains unavailable; configuration must record that limitation instead of truncating or silently using a subset.

Framework integration: v0.9.5 lacks a public trainable_token_indices setting and normally makes entire embedding/head modules trainable when resizing. Preserve the approved four-input-row LoRA contract by a small recorded patch to the project-owned pinned framework checkout; verify against a tiny Qwen model. Do not change the shared environment or silently enable full embedding training.
