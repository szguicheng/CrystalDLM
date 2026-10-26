"""Replay the frozen relational scorer over successful B4 heldout attempts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time


PROJECT = Path("/scratch/chenggui2/CrystalDLM")
B4_ROOT = PROJECT / "experiments/interleaved_relational_scaled32_20261006"
BLOCK_ROOT = PROJECT / "experiments/interleaved_relational_blocks_20261005"
DATA_ROOT = B4_ROOT / "data_001"
EVAL_ROOT = B4_ROOT / "eval_full_history_b4_001"
INPUT = EVAL_ROOT / "heldout16/evaluation.jsonl"
INPUT_SUMMARY = EVAL_ROOT / "heldout16/evaluation_summary.json"
OUTPUT = B4_ROOT / "b4_heldout_native_score_replay.json"

EXPECTED_SCORE_SHA256 = "3c1760481872726d798dd18ec4fb3557d6940e669e4c4d818a07da276d9c477a"
EXPECTED_RECORDS_SHA256 = "231f336234d15a216dc9cb800d8dc1c3f30cf90985184ccc0d3e5c26333e95c6"
EXPECTED_MANIFEST_SHA256 = "25f150360f173970895dafd31e390f94b4b01ca174e6ad09b9e288c94b1e48cc"
EXPECTED_EVALUATION_SHA256 = "0bca5670e0ea9e70e2ceb2c89b13ade589ba5ca17eddba04bbcbe1d0d1aaad70"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    return sha256(Path(path).read_bytes())


def canonical_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return sha256(payload)


sys.path.insert(0, str(BLOCK_ROOT))
import prepare
import score


def main():
    started = time.perf_counter()
    records_path = DATA_ROOT / "records.jsonl"
    manifest_path = DATA_ROOT / "manifest.json"
    score_path = BLOCK_ROOT / "score.py"
    hashes = {
        "score_sha256": file_sha256(score_path),
        "records_sha256": file_sha256(records_path),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_jsonl_sha256": file_sha256(INPUT),
        "evaluation_summary_sha256": file_sha256(INPUT_SUMMARY),
    }
    assert hashes["score_sha256"] == EXPECTED_SCORE_SHA256
    assert hashes["records_sha256"] == EXPECTED_RECORDS_SHA256
    assert hashes["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert hashes["evaluation_jsonl_sha256"] == EXPECTED_EVALUATION_SHA256

    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    heldout = [row for row in records if row["split"] == "heldout"]
    by_id = {row["id"]: row for row in heldout}
    results = [json.loads(line) for line in INPUT.read_text().splitlines() if line]
    assert len(heldout) == 16 and len(results) == 16
    assert [row["id"] for row in heldout] == [row["id"] for row in results]

    source_checks = {row["id"]: score.audit(row["c1"]) for row in heldout}
    assert all(item["valid"] for item in source_checks.values())
    checks = []
    skipped = []

    def add_check(record, stage, attempt, grade_key, rescored, valid):
        stored = record[grade_key]
        matched = stored == rescored
        differing_keys = (
            sorted(key for key in set(stored) | set(rescored)
                   if stored.get(key) != rescored.get(key))
            if isinstance(stored, dict) and isinstance(rescored, dict)
            else ([] if matched else ["$"])
        )
        checks.append({
            "id": record["id"], "stage": stage, "matched": matched,
            "valid": bool(valid), "recorded_valid": bool(
                stored.get("valid", stored.get("success", False))),
            "recorded_sha256": canonical_digest(stored),
            "rescored_sha256": canonical_digest(rescored),
            "differing_top_level_keys": differing_keys,
        })

    def skip_failed(record, stage, attempt):
        if attempt.get("ok"):
            return False
        failure = next((item for item in record.get("failure_classes", [])
                        if item.get("stage") == stage), {})
        error = attempt.get("error", "attempt failed")
        skipped.append({
            "id": record["id"], "stage": stage,
            "reason": failure.get("kind", "attempt_failed"),
            "error_summary": str(error).splitlines()[0][:240],
            "raw_token_slots": len(attempt.get("raw_token_ids", [])),
            "raw_attempt_retained_in": str(INPUT),
        })
        return True

    for record in results:
        source = by_id[record["id"]]

        attempt = record["generation"]
        if not skip_failed(record, "G", attempt):
            audit = score.audit(attempt["value"]["crystal"], requested=source)
            audit = {key: value for key, value in audit.items() if key != "structure"}
            add_check(record, "G", attempt, "generation_audit", audit, audit["valid"])

        attempt = record["reference_read"]
        if not skip_failed(record, "reference_read", attempt):
            grade = score.grade_answer(attempt["value"], source_checks[record["id"]])
            add_check(record, "reference_read", attempt, "reference_read_grade",
                      grade, grade["success"])

        attempt = record["reference_edit"]
        if not skip_failed(record, "E_reference", attempt):
            grade = score.grade_edit(attempt["value"]["crystal"], source["c1"],
                                     prepare.edit_factor(record["direction"]))
            add_check(record, "E_reference", attempt, "reference_edit_grade",
                      grade, grade["success"])

        partner = by_id[record["swap_id"]]
        attempt = record["swap_read"]
        if not skip_failed(record, "Q_swap", attempt):
            grade = score.grade_answer(attempt["value"], source_checks[partner["id"]])
            add_check(record, "swap_read", attempt, "swap_read_grade",
                      grade, grade["success"])

        attempt = record["no_crystal_read"]
        if not skip_failed(record, "Q_no_crystal", attempt):
            grade = score.grade_answer(attempt["value"], source_checks[partner["id"]])
            add_check(record, "no_crystal_read", attempt, "no_crystal_fact_match",
                      grade, grade["success"])

    counts = Counter(check["stage"] for check in checks)
    assert counts == Counter({
        "G": 15, "E_reference": 16, "reference_read": 16,
        "swap_read": 16, "no_crystal_read": 16,
    })
    assert len(skipped) == 1 and skipped[0]["id"] == "mp-1173947"
    mismatches = [check for check in checks if not check["matched"]]
    output = {
        "status": "passed" if not mismatches else "mismatch",
        "scope": "native frozen score.py replay for every successful B4 heldout G, reference E, Q_reference, Q_swap, and no-crystal attempt",
        "checks_count": len(checks),
        "all_checks_matched": not mismatches,
        "checks_by_stage": dict(counts),
        "mismatches": mismatches,
        "checks": checks,
        "skipped_failed_attempts": skipped,
        "input_hashes": hashes,
        "training_or_inference_performed": False,
        "no_goal_completion_claim": True,
        "runtime": {
            "host": socket.gethostname(),
            "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
            "slurm_partition": __import__("os").environ.get("SLURM_JOB_PARTITION"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": output["status"], "checks_count": len(checks),
                      "checks_by_stage": dict(counts), "skipped_failed_attempts": skipped,
                      "output": str(OUTPUT)}, ensure_ascii=False), flush=True)
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
