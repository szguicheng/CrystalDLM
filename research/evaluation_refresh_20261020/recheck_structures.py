#!/usr/bin/env python3
"""Read-only replay of saved crystal bodies through the frozen strict auditor."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "experiments/interleaved_ar_diffusion_20261002/score.py"
OUTPUT = ROOT / "research/evaluation_refresh_20261020/structural-recheck.json"
SOURCES = (
    ("ehull_interleaved_smoke_20261013", "experiments/ehull_interleaved_smoke_20261013/run_001/evaluation.jsonl", "jsonl"),
    ("ehull_pure_ar_baseline_20261013", "experiments/ehull_pure_ar_baseline_20261013/run_001/evaluation.jsonl", "jsonl"),
    ("ehull_pure_ar_qwen3_8b_20261015", "experiments/ehull_pure_ar_qwen3_8b_20261015/evaluation_001/evaluation.jsonl", "jsonl"),
    ("structure_request_eval_20261017/run_001", "experiments/structure_request_eval_20261017/run_001", "attempts"),
    ("structure_request_eval_20261017/preflight_002", "experiments/structure_request_eval_20261017/preflight_002", "attempts"),
    ("structure_conditioned_ar8b_20261017/evaluation_001", "experiments/structure_conditioned_ar8b_20261017/evaluation_001", "attempts"),
    ("structure_conditioned_ar8b_20261017/evaluation_preflight_001", "experiments/structure_conditioned_ar8b_20261017/evaluation_preflight_001", "attempts"),
    ("structure_conditioned_hybrid8b_20261018/evaluation_001", "experiments/structure_conditioned_hybrid8b_20261018/evaluation_001", "attempts"),
    ("structure_conditioned_hybrid8b_20261018/evaluation_preflight_001", "experiments/structure_conditioned_hybrid8b_20261018/evaluation_preflight_001", "attempts"),
    ("structure_conditioned_hybrid8b_20261018/validation_001/evaluation", "experiments/structure_conditioned_hybrid8b_20261018/validation_001/evaluation", "case_tree"),
    ("flare_crystal_qwen3_8b_20261019/evaluation", "experiments/flare_crystal_qwen3_8b_20261019/evaluation", "case_tree"),
    ("flare_crystal_full_20261019/ar_evaluation", "experiments/flare_crystal_full_20261019/ar_evaluation", "attempts"),
    ("flare_crystal_full_20261019/flare_evaluation", "experiments/flare_crystal_full_20261019/flare_evaluation", "attempts"),
    ("flare_crystal_full_20261019/ar_preflight", "experiments/flare_crystal_full_20261019/ar_preflight", "attempts"),
    ("flare_crystal_full_20261019/flare_preflight", "experiments/flare_crystal_full_20261019/flare_preflight", "attempts"),
)
AUDIT_FIELDS = (
    "valid", "reasons", "max_fractional_constraint_error", "decoded_atoms",
    "detected_sg", "decoder_max_representative_shift_A", "minimum_overlap_distance_A",
)
FACT_FIELDS = ("formula", "counts", "sg", "lengths", "angles", "atoms", "declared_atoms", "orbits", "sites")
_SCORE = None


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def init_auditor(path):
    global _SCORE
    spec = importlib.util.spec_from_file_location("frozen_strict_structure_score", path)
    _SCORE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_SCORE)


def audit_body(body):
    return _SCORE.audit(body)


def old_audit(record):
    for value in (record.get("audit"), record.get("crystal_audit")):
        if isinstance(value, dict) and "valid" in value:
            return value
    crystal = record.get("crystal")
    if isinstance(crystal, dict):
        value = crystal.get("audit")
        if isinstance(value, dict) and "valid" in value:
            return value
    return {}


def body_and_boundary(record):
    body = None
    body_field = None
    for field in ("crystal_text", "body", "crystal_body"):
        value = record.get(field)
        if isinstance(value, str):
            body, body_field = value, field
            break
    if body is None:
        crystal = record.get("crystal")
        if isinstance(crystal, dict) and crystal.get("ok") is True and isinstance(crystal.get("raw"), str):
            body, body_field = crystal["raw"], "crystal.raw"
    if body is None and record.get("boundary_ok") is True and isinstance(record.get("raw_text"), str):
        assert "<CRYSTAL_END>" in record["raw_text"]
        body, body_field = record["raw_text"].split("<CRYSTAL_END>", 1)[0], "raw_text_before_end"

    if "boundary_ok" in record:
        boundary = record["boundary_ok"] is True
        boundary_source = "boundary_ok"
    elif "ended" in record:
        boundary = record["ended"] is True
        boundary_source = "ended"
    elif isinstance(record.get("crystal_status"), str):
        boundary = record["crystal_status"] == "ok"
        boundary_source = "crystal_status"
    elif isinstance(record.get("crystal"), dict) and isinstance(record["crystal"].get("ok"), bool):
        boundary = record["crystal"]["ok"]
        boundary_source = "crystal.ok"
    elif body is not None:
        # These body fields are assigned only after the saved evaluator accepts the END boundary.
        boundary = True
        boundary_source = "saved_body_field"
    else:
        boundary = None
        boundary_source = "not_saved"

    return body, body_field, boundary, boundary_source


def extract_case(record, source, path, jsonl_index=None, jsonl_line=None):
    old = old_audit(record)
    old_valid = old.get("valid")
    if type(old_valid) is not bool:
        for candidate in (record.get("structural_valid"), record.get("valid")):
            if type(candidate) is bool:
                old_valid = candidate
                break
    crystal = record.get("crystal")
    if type(old_valid) is not bool and isinstance(crystal, dict) and type(crystal.get("ok")) is bool:
        old_valid = crystal["ok"]
    lead = record.get("lead")
    if type(old_valid) is not bool and isinstance(lead, dict) and lead.get("ok") is False:
        old_valid = False
    if type(old_valid) is not bool:
        old_valid = None
    body, body_field, boundary, boundary_source = body_and_boundary(record)
    case_id = record.get("request_id", record.get("material_id"))
    if case_id is None:
        case_id = "/".join(str(record[k]) for k in ("arm", "mode", "split", "attempt_id") if k in record)
    return {
        "source": source,
        "path": path,
        "jsonl_index": jsonl_index,
        "jsonl_line": jsonl_line,
        "case_id": case_id,
        "body": body,
        "body_saved": body is not None,
        "body_field": body_field,
        "body_sha256": sha256_text(body) if body is not None else None,
        "boundary": "passed" if boundary is True else "failed" if boundary is False else "not_saved",
        "boundary_source": boundary_source,
        "saved_failure_reason": next((value for value in (
            record.get("crystal_error"), record.get("decode_error"), record.get("error"),
            record.get("crystal", {}).get("error") if isinstance(record.get("crystal"), dict) else None,
            record.get("lead_error"), record.get("lead", {}).get("error") if isinstance(record.get("lead"), dict) else None,
        ) if isinstance(value, str) and value), None),
        "old_valid": old_valid,
        "old_audit": old,
        "new_full": None,
    }


def source_files(root, source_path, kind):
    base = root / source_path
    if kind == "jsonl":
        return [base] if base.is_file() else []
    if not base.is_dir():
        return []
    if kind == "attempts":
        return sorted(p for p in base.glob("shard-*/attempts/**/*.json")
                      if "grace_score" not in p.parts)
    return sorted(p for p in base.rglob("*.json") if p.name != "summary.json")


def load_cases(root, stdin_path):
    cases = []
    inputs = []
    for name, source_path, kind in SOURCES:
        paths = source_files(root, source_path, kind)
        via_stdin = kind == "jsonl" and source_path == stdin_path
        if not paths and not via_stdin:
            inputs.append({"source": name, "path": source_path, "status": "missing"})
            continue
        inputs.append({"source": name, "path": source_path, "status": "stdin" if via_stdin else "present"})
        if kind == "jsonl":
            path = root / source_path
            stream = sys.stdin if via_stdin else path.open(encoding="utf-8")
            try:
                for line_index, line in enumerate(stream):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cases.append(extract_case(row, name, source_path, line_index, line_index + 1))
            finally:
                if not via_stdin:
                    stream.close()
            continue
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if kind == "attempts" and "attempts" not in path.parts:
                raise AssertionError(f"unexpected file outside attempt receipts: {relative}")
            if "grace_score" in path.parts:
                raise AssertionError(f"GRACE score output entered structural audit: {relative}")
            cases.append(extract_case(json.loads(path.read_text(encoding="utf-8")), name, relative))
    if stdin_path and not any(item["path"] == stdin_path and item["status"] == "stdin" for item in inputs):
        raise ValueError(f"--stdin-jsonl does not match a configured missing JSONL source: {stdin_path}")
    return cases, inputs


def compact_audit(audit):
    return {key: audit[key] for key in AUDIT_FIELDS if key in audit}


def comparable_differences(old, new):
    different = []
    for key in AUDIT_FIELDS:
        if key in old and key in new and old[key] != new[key]:
            different.append(key)
    old_facts, new_facts = old.get("facts"), new.get("facts")
    if isinstance(old_facts, dict) and isinstance(new_facts, dict):
        different.extend(f"facts.{key}" for key in FACT_FIELDS
                         if key in old_facts and key in new_facts and old_facts[key] != new_facts[key])
    return different


def summary_counts(cases):
    counts = {key: 0 for key in (
        "cases", "body_saved", "boundary_passed", "boundary_failed", "boundary_not_saved",
        "old_valid_true", "old_valid_false", "old_valid_unknown", "new_valid_true", "new_valid_false",
        "changed_validity", "audit_discrepancies",
    )}
    counts["cases"] = len(cases)
    for case in cases:
        counts["body_saved"] += case["body_saved"]
        counts["boundary_passed"] += case["boundary"] == "passed"
        counts["boundary_failed"] += case["boundary"] == "failed"
        counts["boundary_not_saved"] += case["boundary"] == "not_saved"
        counts["old_valid_true"] += case["old_valid"] is True
        counts["old_valid_false"] += case["old_valid"] is False
        counts["old_valid_unknown"] += case["old_valid"] is None
        counts["new_valid_true"] += case["new_valid"] is True
        counts["new_valid_false"] += case["new_valid"] is False
        counts["changed_validity"] += case["changed_validity"]
        counts["audit_discrepancies"] += case["audit_discrepancy"]
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin-jsonl", help="configured JSONL path whose bytes are streamed on stdin")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 4 <= args.workers <= 8:
        parser.error("--workers must be between 4 and 8")
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        parser.error("CUDA_VISIBLE_DEVICES must be empty for this CPU-only audit")
    if OUTPUT.resolve() != (ROOT / "research/evaluation_refresh_20261020/structural-recheck.json").resolve():
        raise AssertionError("output path escaped the owned refresh directory")

    cases, inputs = load_cases(ROOT, args.stdin_jsonl)
    scoreable = [case for case in cases if case["boundary"] == "passed" and case["body"] is not None]
    for case in cases:
        if case["boundary"] == "failed":
            reason = case["saved_failure_reason"] or "saved boundary/status failed; body not repaired"
            case["new_full"] = {"valid": False, "reasons": [reason]}
        elif case["body"] is None:
            reason = case["saved_failure_reason"] or "no saved body to audit"
            case["new_full"] = {"valid": False, "reasons": [reason]}

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_auditor, initargs=(str(AUDITOR),)) as pool:
        futures = {pool.submit(audit_body, case["body"]): case for case in scoreable}
        for done, future in enumerate(as_completed(futures), 1):
            futures[future]["new_full"] = future.result()
            if done % 100 == 0:
                print(f"strict audits {done}/{len(scoreable)}", flush=True)

    discrepancies = []
    for case in cases:
        new = case["new_full"]
        case["new_valid"] = new.get("valid") is True
        old = case.pop("old_audit")
        changed = case["old_valid"] is not None and case["old_valid"] != case["new_valid"]
        diff_fields = comparable_differences(old, new)
        discrepancy = bool(changed or diff_fields)
        case["changed_validity"] = changed
        case["audit_discrepancy"] = discrepancy
        case["new_audit"] = compact_audit(new)
        case.pop("body", None)
        case.pop("new_full", None)
        case.pop("saved_failure_reason", None)
        if discrepancy:
            discrepancies.append({
                "source": case["source"], "path": case["path"],
                "jsonl_index": case["jsonl_index"], "jsonl_line": case["jsonl_line"],
                "case_id": case["case_id"], "body_sha256": case["body_sha256"],
                "old_valid": case["old_valid"], "new_valid": case["new_valid"],
                "changed_validity": changed, "audit_diff_fields": diff_fields,
                "old_audit": old, "new_audit": new,
            })

    by_source = {name: summary_counts([case for case in cases if case["source"] == name])
                 for name, _, _ in SOURCES}
    result = {
        "schema": 1,
        "scope": "Saved structure bodies only; no generation, training, repair, GRACE scoring, or input mutation.",
        "strict_auditor": {
            "path": AUDITOR.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(AUDITOR.read_bytes()).hexdigest(),
            "function": "score.audit(body)",
        },
        "execution": {
            "host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "cpu_workers": args.workers, "worker_mode": "spawned processes",
        },
        "inputs": inputs,
        "counts_by_source": by_source,
        "counts_total": summary_counts(cases),
        "validity_discrepancy_count": len(discrepancies),
        "discrepancies": discrepancies,
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    assert sum(item["cases"] for item in by_source.values()) == result["counts_total"]["cases"]
    assert result["counts_total"]["new_valid_true"] + result["counts_total"]["new_valid_false"] == len(cases)
    print(json.dumps({"output": str(OUTPUT), "counts_total": result["counts_total"],
                      "discrepancies": len(discrepancies)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
