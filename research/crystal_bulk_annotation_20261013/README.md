# Fixed one-fifth annotation subset

User-approved scope, 2026-10-13: first annotate one fifth of the existing eligible train and validation splits, check training usability, and run a preliminary small experiment. Full annotation and later full training follow only after reviewing that experiment. Test remains unchanged.

## Frozen selection

- Seed: `42`, independently initialized for each original split.
- Algorithm: sample `floor(N / 5)` from lexicographically sorted material IDs using Python's `random.Random(42)`.
- Train: 7,272 of 36,362; validation: 900 of 4,501; total: **8,172**.
- IDs: `subset_001/train-material-ids.txt` and `subset_001/val-material-ids.txt`. Use these same lists in the preliminary training experiment; do not draw another sample.
- Facts: `subset_001/{train,val}-facts.jsonl.gz`. Original lines, PLaID text, structure facts and continuous archived MP ehull labels are preserved verbatim.
- Identity and source hashes: `subset_001/manifest.json`.

Re-running `rtk proxy python3 research/crystal_bulk_annotation_20261013/sample.py` verifies and preserves the existing selection. A different seed or changed facts cannot overwrite this subset.

## Annotation settings

Use `research/crystal_sharing_20261012/annotation-flash-config.json` and `description-prompt.md`: DeepSeek V4 Flash, thinking enabled, reasoning effort max, English structure-only prompt. Keep the original ehull outside the language rewriting request and render its exact label separately.

Eight concurrent requests, no delay between successful requests. Retry transient failures with bounded backoff using the same model and settings. Do not silently downgrade the model or replace a failed description with fabricated text.

## Scope and result meaning

Annotation request success and durable storage do not establish scientific correctness of every generated sentence. Preserve raw structure descriptions and property labels for review. The pilot training must consume the same crystal/ehull/description triples and check sample parsing, token lengths and actual model behavior before expanding to full data.

This stage does not launch full-dataset annotation, change the original split, recompute ehull, access the source database, or run MLIP evaluation.

## Running and recovery

Started locally in tmux session `crystal-annotation-subset-20261013`, with `caffeinate -i` while the job runs. This does not use GPU resources. Execution is fixed to the subset and eight workers by `run_subset.py`.

```bash
# View progress while running.
rtk proxy python3 research/crystal_bulk_annotation_20261013/batch.py status
# Resume after an interruption; never start a second runner while the tmux job is active.
rtk proxy python3 research/crystal_bulk_annotation_20261013/run_subset.py
# Re-run only unresolved attempted materials after reviewing the failure queue.
rtk proxy python3 research/crystal_bulk_annotation_20261013/batch.py retry-failed
# Refresh exports after retrying.
rtk proxy python3 research/crystal_bulk_annotation_20261013/batch.py export
```

The source of truth is `run_001/annotation.sqlite3`, using SQLite WAL and FULL synchronous commits. `run_001/batch.log` contains nonsecret progress. Successful and unresolved receipts are exported as `successes.jsonl` and `failures.jsonl` when the wrapper exits. Input, config, prompt, helper and runner hashes are checked before resume. Keep these files unchanged during this run; their nonsecret snapshots and request identity are under `run_001/source_snapshot/` and `request-manifest.json`.

Validation: seven fake-network unit tests, a real local-HTTP subprocess SIGKILL/restart test, and six existing helper tests passed. The real API canary committed 16/16; repeating it left the attempt count at 16, proving those committed materials were skipped. Scientific review of all generated captions and training remain subsequent stages.

A killed process may repeat an in-flight API call whose response was not yet committed; committed successes are retained and skipped. Failures receive at most four attempts per retry pass, with backoff only for transient errors. Authentication/access or model mismatch stops further scheduling; active requests may finish and their successes are committed.
