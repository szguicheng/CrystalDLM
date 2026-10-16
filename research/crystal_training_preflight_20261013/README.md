# Verified training triples for the fixed subset

The frozen seed42 train/validation IDs are unchanged: 7,272 train and 900 validation. `data_001/{train,val}-triples.jsonl.gz` binds PLaID text, English structure description, original continuous archived MP ehull and source fingerprints.

Every source row's canonical SHA256 matches its annotation receipt. IDs and splits are unique and match the fixed lists; all property labels/provenance are unchanged, units are eV/atom, and all stored PLaID consistency checks passed. No records were excluded.

The original 8,172 successful receipts remain unchanged. Two Chinese/Chinese-note captions (`mp-1113520`, `mp-1184824`) were reannotated with the same Flash/max prompt and stored separately in `caption-repairs.jsonl`. Their source hashes and labels match the originals. The merged triples preserve the original raw captions alongside repaired raw captions. Deterministic cleanup removed own material-ID mentions from 376 target descriptions; no structural facts or labels were replaced.

Final language-format flags: zero. This check covers empty paragraphs, line breaks, CJK script and punctuation; it is not a semantic scientific review of all generated sentences. Source identities and file hashes are in `data_001/manifest.json`.

The existing tokenizer audit found crystal body plus END p50/p95/max=158/384/531 tokens. The old K256 cannot hold 1,303 records. The actual 1.5B tokenizer and complete sequence capacities are checked again by the allocated A100 technical preflight in `experiments/ehull_interleaved_smoke_20261013/`.

Binding and cleanup tests: `rtk proxy python3 -m unittest discover -s research/crystal_training_preflight_20261013 -p test_assemble.py -v` (4 tests passed). Generated data was assembled using the existing `agent-safety` environment's tokenizers library; no new dependencies were installed.

Do not rebuild `data_001` while a training run uses its hashes. Original facts/receipts and the new source manifest provide the binding evidence; model runtime checks and scientific outcomes are recorded separately.
