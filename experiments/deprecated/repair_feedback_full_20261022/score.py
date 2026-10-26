"""Export one selected loop result per group to frozen GRACE and summarize both stages."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
LOOP_SUMMARIZER = ROOT / "experiments/structure_feedback_loop_20261020/summarize.py"
ARMS = ("independent", "feedback")
MAX_ATTEMPTS = 3
PROPERTY_THRESHOLD = 0.08


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_once(path, value):
    path = Path(path)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    _write_bytes_once(path, payload)


def _write_bytes_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing output differs; refusing to overwrite {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"existing output differs; refusing to overwrite {path}")
    finally:
        Path(temporary).unlink(missing_ok=True)


def select_attempt(attempts):
    if not 1 <= len(attempts) <= MAX_ATTEMPTS:
        raise ValueError("an arm must have one to three persisted attempts")
    first_success = None
    for round_index, attempt in enumerate(attempts):
        if attempt.get("round") != round_index:
            raise ValueError("attempt rounds must be contiguous from zero")
        strict_success = (attempt.get("structural_valid") is True and
                          attempt.get("requested_sg_match") is True)
        if attempt.get("success") is not strict_success:
            raise ValueError("attempt success does not match strict validity and requested space group")
        if strict_success:
            if first_success is not None or round_index != len(attempts) - 1:
                raise ValueError("loop continued after a successful candidate")
            first_success = attempt
    return first_success if first_success is not None else attempts[-1]


def _load_run(run_dir):
    run_dir = Path(run_dir).expanduser().resolve(strict=True)
    if not run_dir.is_relative_to(ROOT.resolve()):
        raise ValueError("run-dir must be under the project root")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    plan = manifest["group_plan"]
    if manifest.get("group_plan_sha256") != _json_sha256(plan):
        raise ValueError("run manifest group-plan checksum does not match its frozen plan")
    planned = {}
    for item in plan:
        request_id = item["request_id"]
        attempt_id = item["attempt_id"]
        if not re.fullmatch(r"sg-\d{3}", request_id):
            raise ValueError("request_id cannot contain path components")
        if not isinstance(attempt_id, int) or attempt_id < 0:
            raise ValueError("group attempt_id must be a nonnegative integer")
        key = (request_id, attempt_id)
        if key in planned:
            raise ValueError(f"duplicate planned group {key}")
        planned[key] = item
    if not plan:
        raise ValueError("run manifest has no planned groups")

    groups = {}
    for path in sorted(run_dir.glob("shard-*/groups/sg-*/group-*.json")):
        group = _read_json(path)
        key = (group["request_id"], group["group_index"])
        if key not in planned:
            raise ValueError(f"unexpected group receipt {key}: {path}")
        if key in groups:
            raise ValueError(f"duplicate group receipt {key}")
        item = planned[key]
        if group.get("status") != "complete" or group.get("plan_item") != item:
            raise ValueError(f"group receipt is incomplete or differs from its frozen plan: {path}")
        if (group.get("run_identity_sha256") != manifest.get("run_identity_sha256") or
                group.get("checkpoint_fingerprint_sha256") != manifest.get("checkpoint_fingerprint_sha256")):
            raise ValueError(f"group receipt belongs to a different run or checkpoint: {path}")
        arms = group["arms"]
        for arm in ARMS:
            record = arms[arm]
            attempts = record["attempts"]
            selected = select_attempt(attempts)
            succeeded = selected.get("success") is True
            if record.get("attempts_used") != len(attempts) or record.get("success") is not succeeded:
                raise ValueError(f"arm terminal status differs from its saved attempts: {path} {arm}")
            if not succeeded and len(attempts) != MAX_ATTEMPTS:
                raise ValueError(f"failed arm ended before exhausting its attempt budget: {path} {arm}")
            if any(attempt.get("checkpoint_fingerprint_sha256") !=
                   manifest.get("checkpoint_fingerprint_sha256") for attempt in attempts):
                raise ValueError(f"attempt belongs to a different checkpoint: {path} {arm}")
            if any(attempt.get("request_id") != item["request_id"] or
                   attempt.get("plan_index") != item["plan_index"] for attempt in attempts):
                raise ValueError(f"attempt belongs to a different planned group: {path} {arm}")
        if arms["independent"]["attempts"][0] != arms["feedback"]["attempts"][0]:
            raise ValueError(f"paired initial candidates differ: {path}")
        groups[key] = {"group": group, "path": path}

    missing = planned.keys() - groups.keys()
    if missing:
        raise ValueError(f"run is missing {len(missing)} completed group receipts")
    loop_summary = _load_loop_summarizer().summarize_groups(
        [groups[(item["request_id"], item["attempt_id"])]["group"] for item in plan],
        scheduled=len(plan))
    return run_dir, manifest_path, manifest, plan, groups, loop_summary


def _cif_for_attempt(run_dir, attempt):
    if attempt.get("structural_valid") is not True:
        return None
    cif = attempt.get("cif")
    if not cif or not cif.get("path") or not cif.get("sha256"):
        raise ValueError("strict-valid selected attempt has no checksummed CIF")
    path = Path(cif["path"])
    if not path.is_absolute():
        path = run_dir / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("selected CIF must be under the project root")
    digest = _sha256_file(path)
    if digest != cif["sha256"] or cif.get("bytes") != path.stat().st_size:
        raise ValueError(f"selected CIF checksum or size does not match its receipt: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _export_receipt(run_dir, item, group, group_path, group_sha256, arm):
    attempts = group["arms"][arm]["attempts"]
    selected = select_attempt(attempts)
    successful = selected["success"] is True
    return {
        "request_id": item["request_id"],
        "attempt_id": item["attempt_id"],
        "seed": item["seed"],
        "space_group_number": item["space_group_number"],
        "structural_valid": selected.get("structural_valid") is True,
        "requested_sg_match": selected.get("requested_sg_match") is True,
        "actual_space_group_number": selected.get("actual_space_group_number"),
        "cif": _cif_for_attempt(run_dir, selected),
        "loop_selection": {
            "arm": arm,
            "source_group": str(group_path.relative_to(run_dir)),
            "source_group_sha256": group_sha256,
            "selected_round": selected["round"],
            "selected_seed": selected.get("seed"),
            "selection": "first_strict_valid_requested_sg" if successful else "terminal_failure",
            "checkpoint_fingerprint_sha256": group["checkpoint_fingerprint_sha256"],
        },
    }


def export_run(run_dir):
    run_dir, manifest_path, manifest, plan, groups, _ = _load_run(run_dir)
    if not manifest.get("checkpoint_fingerprint_sha256"):
        raise ValueError("run manifest has no checkpoint fingerprint")
    run_manifest_sha256 = _sha256_file(manifest_path)
    result = {"status": "complete", "raw_attempt_denominator": len(plan), "arms": {}}
    for arm in ARMS:
        attempt_plan = [dict(item) for item in plan]
        selected_identity = []
        receipts = []
        for item in plan:
            key = (item["request_id"], item["attempt_id"])
            group, path = groups[key]["group"], groups[key]["path"]
            group_sha256 = _sha256_file(path)
            receipt = _export_receipt(run_dir, item, group, path, group_sha256, arm)
            receipts.append((item, receipt))
            selected_identity.append({
                "request_id": item["request_id"], "group_index": item["attempt_id"],
                "source_group": receipt["loop_selection"]["source_group"],
                "source_group_sha256": group_sha256,
                "selected_round": receipt["loop_selection"]["selected_round"],
                "selected_seed": receipt["loop_selection"]["selected_seed"],
            })
        export_manifest = {
            "schema": 1,
            "experiment": "repair_feedback_full_20261022",
            "arm": arm,
            "source_run": str(run_dir.relative_to(ROOT.resolve())),
            "source_run_manifest_sha256": run_manifest_sha256,
            "source_run_identity_sha256": manifest.get("run_identity_sha256"),
            "checkpoint_fingerprint_sha256": manifest["checkpoint_fingerprint_sha256"],
            "selection_policy": "first strict-valid requested-space-group match; otherwise terminal attempt",
            "raw_attempt_denominator": len(plan),
            "attempt_plan": attempt_plan,
            "attempt_plan_sha256": _json_sha256(attempt_plan),
            "selected_identity_sha256": _json_sha256(selected_identity),
            "selected_identity": selected_identity,
        }
        export_dir = run_dir / "selected" / arm
        _write_json_once(export_dir / "run_manifest.json", export_manifest)
        for item, receipt in receipts:
            path = (export_dir / "shard-00-of-01" / "attempts" / item["request_id"] /
                    f"attempt-{item['attempt_id']:03d}.json")
            _write_json_once(path, receipt)
        result["arms"][arm] = {"selected_outputs": len(receipts), "path": str(export_dir)}
    return result


def _load_loop_summarizer():
    spec = importlib.util.spec_from_file_location("repair_feedback_loop_summarizer", LOOP_SUMMARIZER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_property_results(run_dir, arm, plan, groups, source_manifest_sha256, manifest):
    export_dir = run_dir / "selected" / arm
    export_manifest_path = export_dir / "run_manifest.json"
    export_manifest = _read_json(export_manifest_path)
    if (export_manifest.get("arm") != arm or
            export_manifest.get("raw_attempt_denominator") != len(plan) or
            export_manifest.get("attempt_plan") != plan or
            export_manifest.get("attempt_plan_sha256") != _json_sha256(plan) or
            export_manifest.get("source_run_manifest_sha256") != source_manifest_sha256 or
            export_manifest.get("source_run_identity_sha256") != manifest.get("run_identity_sha256") or
            export_manifest.get("checkpoint_fingerprint_sha256") !=
            manifest.get("checkpoint_fingerprint_sha256")):
        raise ValueError(f"selected export does not match the run plan: {export_manifest_path}")
    scoring_dir = export_dir / "grace_score"
    scoring_manifest_path = scoring_dir / "scoring_manifest.json"
    scoring_manifest = _read_json(scoring_manifest_path)
    if (scoring_manifest.get("run_manifest_sha256") != _sha256_file(export_manifest_path) or
            scoring_manifest.get("generation_attempt_plan_sha256") != export_manifest["attempt_plan_sha256"]):
        raise ValueError(f"GRACE scoring manifest does not match the selected export: {arm}")
    summary = _read_json(scoring_dir / "summary.json")
    if (summary.get("status") != "complete" or summary.get("completed_attempts") != len(plan) or
            summary.get("raw_attempt_denominator") != len(plan)):
        raise ValueError(f"GRACE results are incomplete or have the wrong denominator: {arm}")
    rows = {}
    for item in plan:
        request_id, attempt_id = item["request_id"], item["attempt_id"]
        key = (request_id, attempt_id)
        result_path = (scoring_dir / "attempts" / request_id / f"attempt-{attempt_id:03d}.json")
        row = _read_json(result_path)
        if (row.get("request_id"), row.get("attempt_id")) != key:
            raise ValueError(f"GRACE result does not match selected group {key}")
        exported_receipt = (export_dir / "shard-00-of-01" / "attempts" / request_id /
                            f"attempt-{attempt_id:03d}.json")
        _validate_generation_receipt(row, exported_receipt)
        selected = select_attempt(groups[key]["group"]["arms"][arm]["attempts"])
        if (row.get("structural_valid") is not (selected.get("structural_valid") is True) or
                row.get("requested_sg_match") is not (selected.get("requested_sg_match") is True) or
                row.get("actual_space_group_number_pre_relax") !=
                selected.get("actual_space_group_number") or
                row.get("space_group_number_requested") != item["space_group_number"]):
            raise ValueError(f"GRACE result changed generated structural facts for {key} {arm}")
        rows[key] = row
    return summary, rows


def _validate_generation_receipt(row, expected_path):
    expected_path = Path(expected_path).resolve(strict=True)
    receipt = row.get("generation_receipt") or {}
    try:
        actual_path = Path(receipt["path"]).resolve(strict=True)
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("GRACE result has no readable source generation receipt") from error
    if (actual_path != expected_path or not actual_path.is_relative_to(ROOT.resolve()) or
            receipt.get("sha256") != _sha256_file(expected_path)):
        raise ValueError(f"GRACE result references a different selected receipt: {expected_path}")


def _load_frozen_grace_scorer():
    path = ROOT / "experiments/structure_request_eval_20261017/grace_score.py"
    spec = importlib.util.spec_from_file_location("repair_feedback_frozen_grace_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gated_unique_novel_counts(rows, generated_success_keys, matcher):
    grace = _load_frozen_grace_scorer()
    eligible = []
    for row in rows:
        key = (row["request_id"], row["attempt_id"])
        if key not in generated_success_keys:
            continue
        candidate = dict(row)
        candidate["_structure"] = grace._structure_from_result(row)
        eligible.append(candidate)
    grace._mark_unique_novel(eligible, matcher)
    result = {}
    for threshold in grace.THRESHOLDS:
        key = format(float(threshold), "g")
        per_request = {}
        within_request = global_unique = 0
        for row in eligible:
            request_id = row["request_id"]
            counts = per_request.setdefault(request_id, {"within_request": 0, "global": 0})
            is_within = threshold in row.get("unique_novel_thresholds_eV_per_atom", [])
            is_global = threshold in row.get("global_unique_novel_thresholds_eV_per_atom", [])
            within_request += is_within
            global_unique += is_global
            counts["within_request"] += is_within
            counts["global"] += is_global
        result[key] = {"within_request": within_request, "global": global_unique,
                       "per_request": per_request}
    return result


def _arm_gated_property_counts(plan, groups, rows, arm, matcher):
    generated_successes = final_successes = 0
    per_request = {}
    generated_success_keys = set()
    for item in plan:
        key = (item["request_id"], item["attempt_id"])
        selected = select_attempt(groups[key]["group"]["arms"][arm]["attempts"])
        generated_success = (selected.get("structural_valid") is True and
                             selected.get("requested_sg_match") is True)
        generated_successes += generated_success
        request_counts = per_request.setdefault(item["request_id"], {
            "raw_group_denominator": 0, "generated_strict_successes": 0,
            "final_post_relax_successes_leq_0_08": 0})
        request_counts["raw_group_denominator"] += 1
        request_counts["generated_strict_successes"] += generated_success
        if generated_success:
            generated_success_keys.add(key)
        row = rows[key]
        ehull = row.get("ehull_proxy_corrected_eV_per_atom")
        post_relax_success = (
            generated_success and row.get("score_status") == "scored" and
            row.get("converged") is True and row.get("post_relax_valid") is True and
            row.get("post_relax_sg_match") is True and ehull is not None and
            ehull <= PROPERTY_THRESHOLD)
        final_successes += post_relax_success
        request_counts["final_post_relax_successes_leq_0_08"] += post_relax_success
    unique_novel = gated_unique_novel_counts(list(rows.values()), generated_success_keys, matcher)
    main_unique_novel = unique_novel[format(PROPERTY_THRESHOLD, "g")]
    for request_id, counts in per_request.items():
        unique_counts = main_unique_novel["per_request"].get(request_id, {})
        counts["unique_novel_final_successes_leq_0_08"] = unique_counts.get("within_request", 0)
        counts["global_unique_novel_final_successes_leq_0_08"] = unique_counts.get("global", 0)
    return {
        "generated_strict_successes": generated_successes,
        "final_post_relax_successes_leq_0_08": final_successes,
        "unique_novel_final_successes_leq_0_08": main_unique_novel["within_request"],
        "global_unique_novel_final_successes_leq_0_08": main_unique_novel["global"],
        "unique_novel_successes_by_threshold": unique_novel,
        "per_request": per_request,
        "raw_group_denominator": len(plan),
    }


def summarize_run(run_dir):
    run_dir, manifest_path, manifest, plan, groups, loop_summary = _load_run(run_dir)
    source_manifest_sha256 = _sha256_file(manifest_path)
    property_reports = {}
    gated_reports = {}
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True,
                               scale=True, attempt_supercell=False)
    for arm in ARMS:
        property_reports[arm], rows = _load_property_results(
            run_dir, arm, plan, groups, source_manifest_sha256, manifest)
        gated_reports[arm] = _arm_gated_property_counts(plan, groups, rows, arm, matcher)
    result = {
        "status": "complete",
        "experiment": "repair_feedback_full_20261022",
        "checkpoint_fingerprint_sha256": manifest["checkpoint_fingerprint_sha256"],
        "scheduled_groups_per_arm": len(plan),
        "loop_summary": loop_summary,
        "property_evaluation": {
            "status": "complete",
            "scorer": "GRACE-3L-OAM-L with fixed MP 2023 hull and MP2020 corrections",
            "property_scores_used_in_feedback_or_selection": False,
            "main_success_definition": (
                "loop-selected strict-valid requested-SG output AND converged GRACE result "
                "that is post-relax valid, preserves requested SG, and has corrected proxy ehull <=0.08 eV/atom"),
            "score_label": property_reports[ARMS[0]].get("score_label"),
            "gated_main_metrics": gated_reports,
            "frozen_grace_reports_supplementary": property_reports,
        },
    }
    json_path = run_dir / "scoring-summary.json"
    markdown_path = run_dir / "scoring-RESULTS.md"
    _write_json_once(json_path, result)
    _write_bytes_once(markdown_path, _summary_markdown(result).encode())
    return result


def _summary_markdown(result):
    loop = result["loop_summary"]
    props = result["property_evaluation"]
    lines = [
        "# Repair feedback full-loop evaluation",
        "",
        f"Status: {result['status']}; {result['scheduled_groups_per_arm']} scheduled groups per policy.",
        "",
        "Generation success requires strict structural validity and the originally requested space group.",
        "GRACE relaxed-space-group and proxy-ehull results are reported separately; the primary final count also requires generation success.",
        "",
        "| Policy | Generated strict successes | GRACE post-relax valid | GRACE post-relax requested-SG matches | Final valid, matching, converged and ≤0.08 eV/atom | Unique and novel among final successes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        raw = props["frozen_grace_reports_supplementary"][arm]
        gated = props["gated_main_metrics"][arm]
        lines.append(
            f"| {arm} | {gated['generated_strict_successes']}/{gated['raw_group_denominator']} "
            f"| {raw['post_relax_valid_attempts']}/{gated['raw_group_denominator']} "
            f"| {raw['post_relax_condition_matches']}/{gated['raw_group_denominator']} "
            f"| {gated['final_post_relax_successes_leq_0_08']}/{gated['raw_group_denominator']} "
            f"| {gated['unique_novel_final_successes_leq_0_08']}/{gated['raw_group_denominator']} |"
        )
    lines += [
        "",
        f"Paired loop outcomes: both {loop['paired']['both_success']}; independent only {loop['paired']['independent_only']}; feedback only {loop['paired']['feedback_only']}; neither {loop['paired']['neither']}.",
        "",
        "The final main count requires the loop-selected output to have passed strict structure and requested-space-group checks before GRACE relaxation, then remain valid, preserve the requested space group, converge, and score ≤0.08 eV/atom.",
        "",
        "Full per-arm GRACE reports, including supplementary thresholds, uniqueness, novelty, and failures, are in `scoring-summary.json`. GRACE proxy scores were post-hoc only and were not returned to the feedback loop.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "summarize"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = export_run(args.run_dir) if args.mode == "export" else summarize_run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
