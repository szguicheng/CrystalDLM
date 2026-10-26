"""Measure capacity from fixed complete structures; never publish training data."""
import json
import os
from pathlib import Path
import prepare_data as prep


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Use an allocated CPU node"
    assert prep.sha256(Path(prep.__file__).read_bytes()) == "92c65d07289257391cc400d3260759d99398c06ff68f34175eef10d103fe988b"
    selection_path = prep.PREPARE_DIR / "selection.json"
    selection_bytes = selection_path.read_bytes()
    selection_hash = prep.sha256(selection_bytes)
    assert selection_hash == "82707935dd3d7c71c70cefc2fb7ca6eadc73079d83ed9e0a55cf34147af25988"
    selection = json.loads(selection_bytes)
    helper = prep._load_file_module("capacity_scaled32_helper", prep.HELPER_PATH)
    assert prep.sha256(prep.HELPER_PATH.read_bytes()) == prep.EXPECTED_HELPER_SHA256
    helper.verify_frozen_sources()
    _, _, adapter, score = prep._load_adapters(helper)
    before = prep._runtime_source_state()
    assert prep._runtime_state_matches_pin(before)
    assert prep._raw_cif_dependencies_match(prep._raw_cif_dependency_state())
    # These generous bounds enable measurement only, not a training configuration.
    helper.K = adapter.K = 4096
    helper.MAXLEN = adapter.MAXLEN = 32768
    attempt = prep.PREPARE_DIR / "prepare_attempt_003"
    paths = prep._source_paths(selection["cases"], attempt)
    adapter._source_paths = paths.__getitem__
    new_rows, failures = [], []
    for case in selection["cases"]:
        case_id = case["id"]
        source = paths[case_id]
        hashes = {"selection_sha256": selection_hash,
                  "source_cif_sha256": case["input_cif_sha256"],
                  **{f"source_{key}_sha256": prep.sha256(source[key].read_bytes())
                     for key in ("plan", "facts", "text")}}
        try:
            new_rows.append(helper.prepare_row(case_id, selection_bytes,
                                              {case_id: hashes}, adapter, score))
        except Exception as error:
            failures.append({"id": case_id, "error": f"{type(error).__name__}: {error}"})
        if (len(new_rows) + len(failures)) % 20 == 0:
            print(json.dumps({"processed": len(new_rows) + len(failures),
                              "passed": len(new_rows), "failed": len(failures)}), flush=True)
    result = {"scope": "fixed224 complete native source/plus/minus validation; no data/model writes",
              "selection_sha256": selection_hash, "diagnostic_upper_bounds": {"K": 4096, "MAXLEN": 32768},
              "selected": 224, "passed": len(new_rows), "failures": failures}
    if not failures:
        assert len(new_rows) == 224
        rows = prep._old_cohort()["rows"] + new_rows
        maximum_body = max(row["token_lengths"][key] for row in rows
                           for key in ("c1", "plus", "minus"))
        minimum_K = ((maximum_body + 31) // 32) * 32
        helper.K = adapter.K = minimum_K
        packing = helper.audit_packing(rows, adapter)
        full_path = prep.PROJECT / "experiments/interleaved_relational_scaled32_20261006/train_full_history.py"
        assert prep.sha256(full_path.read_bytes()) == "e2954cf3eea94639782987596799a772b475d6a3b62eada787ad8a4820633112"
        full = prep._load_file_module("capacity_full_history", full_path)
        assert Path(full.runner.prepare.__file__).resolve() == Path(adapter.__file__).resolve()
        encode = lambda text: len(adapter.TOK.encode(text, add_special_tokens=False))
        reader_lengths = []
        for split in ("train", "heldout"):
            cohort = [row for row in rows if row["split"] == split]
            for index, row in enumerate(cohort):
                partner = cohort[(index + 1) % len(cohort)]
                for direction in ("minus", "plus"):
                    tasks = adapter.tasks(row, split, direction, 0)
                    for task in (tasks[1], tasks[3]):
                        for context in ("neutral", "conflict"):
                            prefix, _ = full.BASE._reader_prefix(
                                task, row, partner, direction, "grounded", context)
                            reader_lengths.append(encode(prefix) + encode(task["target"]))
        maximum_packed = max(packing["max_actual_dual_pack_prefix_streams_plus_2K"],
                             packing["max_dual_stream_prefix_plus_2K"], max(reader_lengths),
                             *(value["max"] for arm in packing["arms"].values()
                               for split in arm.values() for value in split.values()))
        result.update(maximum_body_tokens_including_END=maximum_body,
                      minimum_K_aligned32=minimum_K, packing=packing,
                      full_history_reader_checks=len(reader_lengths),
                      maximum_full_history_reader_tokens=max(reader_lengths),
                      maximum_required_context_tokens=maximum_packed,
                      minimum_MAXLEN_aligned256=((maximum_packed + 255) // 256) * 256,
                      original_MAXLEN8192_sufficient=maximum_packed <= 8192,
                      long_case=next(row for row in new_rows if row["id"] == "mp-1276200"))
    after = prep._runtime_source_state()
    assert before == after and prep._runtime_state_matches_pin(after)
    result["frozen_sources_unchanged"] = True
    output = prep.ROOT / "capacity_diagnostic_001.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("packing", "long_case")}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
