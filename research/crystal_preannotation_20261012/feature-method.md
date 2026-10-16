# Frozen crystal preannotation method

Method ID: `crystal-preannotation-20261012-v3`. The exact parameters and software versions are in [feature-method.json](feature-method.json); each output row carries that file's SHA-256.

## Structure and label sources

For each source JSONL record, the script reconstructs its pymatgen `Structure`, writes it to CIF text, calls the existing `get_crystal_string_wyckoff_pyx` encoder with `tol=0.01` and `translate=False`, and decodes the returned text with `parse_fn_wyckoff(..., to_cif=False)`. The encoder's built-in PyXtal fallback from `0.01` to `0.0001` remains active and is recorded per row when used. `work/PLaID/utils.py` is unmodified; its pinned SHA-256 and the runtime package versions are checked before processing.

Core structural facts and every local descriptor are calculated from the **decoded** structure. The original `energy_above_hull` value, unit, and source provenance stay in a separate `property_labels` object. The pipeline never recalculates this label from the decoded structure.

The consistency flags compare the encoded `Sites` header and summed Wyckoff multiplicities with the decoded site count, compare source and decoded reduced composition and element sets, check that the decoded cell has positive volume and no periodic duplicate sites within `1e-5 Å`, compare the encoded and decoded international group numbers using PyXtal's `Group` parser (the same alias parser used by PLaID decode), canonicalize both symbols with Pymatgen `SpaceGroup.from_int_number`, compare independent-orbit counts, and compare the source and decoded structures with the frozen `StructureMatcher` settings. This accepts valid spellings such as `P21/m`, `P21`, and `P21/c` while retaining the group-number mismatch check. Any failed check creates a `status="error"` row that retains the encoded text, decoded structure, source fields, and flags. The annotation client should select only `status="ok"` rows.

Source atom count and decoded cell atom count are both recorded. A larger conventional decoded cell is retained when the consistency checks pass; the source `nsites` limit is not reapplied to the decoded cell.

## Local coordination and connectivity

The periodic neighbor list comes from pymatgen `CrystalNN` in geometry-only mode: `weighted_cn=False`, `cation_anion=False`, `distance_cutoffs=None`, `x_diff_weight=0`, `porous_adjustment=False`, and `search_cutoff=7 Å`. These settings remove oxidation-state and electronegativity preferences. No oxidation states are added or inferred. Neighbor element counts, coordination numbers, and distance ranges are geometric descriptors; the `StructureGraph` edges are not asserted to be chemical bonds.

Polyhedral shape scores are retained as metadata. `CrystalNN.get_local_order_parameters` can return `None` for coordination numbers it does not cover; in that case the score map is empty and no shape label is emitted. Individual undefined Pymatgen scores remain JSON `null`; any threshold rule that needs one of those scores fails closed, while another fully supported motif rule may still label the site. Categorical labels are emitted only when Pymatgen's documented `LocalStructOrderParams` criteria cross their fixed coordination-specific thresholds: tetrahedral `qtet>0.5` at CN 4; square-pyramidal `qsqpyr>0.8` or trigonal-bipyramidal `qtribipyr>0.8` at CN 5; octahedral `qoct>0.5` at CN 6; bcc `qbcc>0.5` plus `qtet<0.5` at CN 8; and cp `q6>0.4` plus `qtet`, `qoct`, and `qbcc<0.4` at CN 12. The order-parameter values are scores, not probabilities. An unrecognized or multiply assigned site has no categorical motif label. The motif calculation uses the same CrystalNN neighbors as the reported CN.

Connectivity is the maximum periodic dimension from Pymatgen's Larsen algorithm on a `StructureGraph` built with that same CrystalNN instance. Per-component dimensions and site counts are retained. This describes the selected geometric neighbor network.

Corner-, edge-, and face-sharing polyhedra are **not computed** by this method. No such relationship is supplied to the annotation model; it must not infer sharing from the dimension or raw order-parameter scores.

The pymatgen APIs and algorithms are described in its official [CrystalNN and StructureGraph documentation](https://pymatgen.org/pymatgen.core.html), [local order parameter documentation](https://pymatgen.org/pymatgen.core.html), and [dimensionality documentation](https://pymatgen.org/pymatgen.analysis.html). The available categorical labels and thresholds are frozen in `feature-method.json`.

## Output and running

The CLI reads either plain or gzip-compressed JSONL and writes one UTF-8 JSONL row for each selected source row. It streams rows, assigns source row indices from zero, and assigns rows to shards using `row_index % shards == shard_index`. `--limit` counts selected rows after sharding. Output is written to a temporary file in the target directory and atomically linked to the requested path when the shard finishes; an existing output path is rejected. Failed records are isolated as error rows and later rows continue.

Run the requested input with the installed PLaID environment. A bounded local check is:

```bash
research/plaid_20260930/.venv/bin/python research/crystal_preannotation_20261012/preprocess.py \
  --input research/mp_ehull030_20261011/collection_001/data_001/records.jsonl.gz \
  --output /path/to/new/shard-000.jsonl \
  --shard-index 0 --shards 1 --limit 3
```

The full 45,629-row archive is not computed locally by this script task. Full processing should use allocated compute and distinct output paths for each shard. No LLM calls or remote jobs are part of preprocessing.

## Official API references

- [pymatgen `CrystalNN`, local order parameters, and `StructureGraph`](https://pymatgen.org/pymatgen.core.html)
- [pymatgen structure dimensionality APIs](https://pymatgen.org/pymatgen.analysis.html)
- [Larsen et al., “Definition of a scoring parameter to identify low-dimensional materials components,” Phys. Rev. Materials 3, 034003 (2019)](https://doi.org/10.1103/PhysRevMaterials.3.034003)
