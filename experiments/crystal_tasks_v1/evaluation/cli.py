"""Shared command-line evaluators for the frozen Crystal Tasks v1 splits."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

from .references import (
    _manifest,
    load_reference_banks,
    load_split_rows,
    read_jsonl,
)
from .understanding import summarize_understanding


ROOT = Path(__file__).resolve().parents[3]


def _row_id(row, key, label):
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} row needs a nonempty {key}")
    return value


def _unique_by(rows, key, label):
    indexed = {}
    for index, row in enumerate(rows):
        value = _row_id(row, key, f"{label} row {index}")
        if value in indexed:
            raise ValueError(f"duplicate {key} {value!r} in {label}")
        indexed[value] = row
    return indexed


def _require_same_ids(expected, observed, label):
    missing = expected.keys() - observed.keys()
    extra = observed.keys() - expected.keys()
    if missing or extra or len(expected) != len(observed):
        detail = []
        if missing:
            detail.append(f"missing={sorted(missing)[:5]}")
        if extra:
            detail.append(f"extra={sorted(extra)[:5]}")
        raise ValueError(f"{label} identity/count mismatch: {'; '.join(detail)}")


def _attempt_hashes(attempt):
    request = _canonical_json(attempt["request"]).encode("utf-8")
    return {
        "output_sha256": hashlib.sha256(attempt["output"].encode("utf-8")).hexdigest(),
        "request_sha256": hashlib.sha256(request).hexdigest(),
    }


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _join_predictions(dataset_rows, predictions):
    expected = _unique_by([row["metadata"] for row in dataset_rows], "material_id", "frozen metadata")
    observed = _unique_by(predictions, "material_id", "predictions")
    _require_same_ids(expected, observed, "prediction")
    joined = []
    for row in dataset_rows:
        material_id = row["metadata"]["material_id"]
        prediction = observed[material_id]
        response = prediction.get("response")
        if not isinstance(response, str):
            raise ValueError(f"prediction for {material_id} needs a string response")
        facts = row["metadata"]["core_facts"]
        joined.append({
            "material_id": material_id,
            "response": response,
            "reference": {
                "formula": facts["cell_formula"],
                "space_group": facts["space_group_number"],
                "cell_atom_count": facts["cell_atom_count"],
                "independent_orbit_count": facts["independent_orbit_count"],
            },
        })
    return joined


def _join_property_receipts(attempts, receipt_rows):
    expected = _unique_by(attempts, "attempt_id", "attempts")
    observed = _unique_by(receipt_rows, "attempt_id", "property receipts")
    _require_same_ids(expected, observed, "property receipt")
    joined = []
    for attempt in attempts:
        attempt_id = attempt["attempt_id"]
        if "property_receipt" in attempt:
            raise ValueError(f"attempt {attempt_id} already contains a property_receipt")
        receipt_row = observed[attempt_id]
        if receipt_row.get("seed") != attempt["seed"]:
            raise ValueError(f"seed mismatch for property receipt {attempt_id}")
        for key, expected_hash in _attempt_hashes(attempt).items():
            if receipt_row.get(key) != expected_hash:
                raise ValueError(f"{key} mismatch for property receipt {attempt_id}")
        receipt = receipt_row.get("property_receipt")
        if not isinstance(receipt, dict):
            raise ValueError(f"property receipt for {attempt_id} needs an object property_receipt")
        joined.append({**attempt, "property_receipt": receipt})
    return joined


def _verify_inline_property_receipts(attempts):
    for attempt in attempts:
        if "property_receipt" not in attempt:
            continue
        for key, expected_hash in _attempt_hashes(attempt).items():
            if attempt.get(key) != expected_hash:
                raise ValueError(f"{key} mismatch for inline property receipt {attempt['attempt_id']}")


def _join_attempt_plan(attempts, plan_rows):
    if not plan_rows:
        raise ValueError("attempt plan is empty")
    by_attempt = _unique_by(attempts, "attempt_id", "attempts")
    by_plan = _unique_by(plan_rows, "attempt_id", "attempt plan")
    _require_same_ids(by_plan, by_attempt, "attempt plan")
    ordered = []
    for index, planned in enumerate(plan_rows):
        attempt = by_attempt[planned["attempt_id"]]
        if type(planned.get("seed")) is not int or type(attempt.get("seed")) is not int:
            raise ValueError(f"attempt plan row {index} and attempt need integer seeds")
        if planned["seed"] != attempt["seed"]:
            raise ValueError(f"seed mismatch for attempt {planned['attempt_id']}")
        planned_request = planned.get("request")
        attempt_request = attempt.get("request")
        for label, request in (("plan", planned_request), ("attempt", attempt_request)):
            if (
                not isinstance(request, dict)
                or set(request) != {"space_group", "ehull_upper"}
                or type(request.get("space_group")) is not int
                or type(request.get("ehull_upper")) not in (int, float)
                or not math.isfinite(request["ehull_upper"])
            ):
                raise ValueError(f"{label} row for {planned['attempt_id']} has an invalid request")
        if _canonical_json(planned_request) != _canonical_json(attempt_request):
            raise ValueError(f"request mismatch for attempt {planned['attempt_id']}")
        ordered.append(attempt)
    return ordered


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_metadata():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            capture_output=True,
        ).stdout.strip()

    try:
        return {
            "branch": git("branch", "--show-current"),
            "git_commit": git("rev-parse", "HEAD"),
            "working_tree": "unknown",
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        source_manifest = ROOT / "experiments/crystal_tasks_v1/source_manifest.json"
        if not source_manifest.is_file():
            raise RuntimeError(f"no Git checkout or source manifest for provenance under {ROOT}")
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
        commit, branch = manifest.get("git_commit"), manifest.get("branch")
        if not isinstance(commit, str) or not commit or not isinstance(branch, str) or not branch:
            raise ValueError(f"source manifest lacks git_commit or branch: {source_manifest}")
        dirty = manifest.get("dirty")
        return {
            "branch": branch,
            "git_commit": commit,
            "working_tree": "dirty" if dirty is True else "clean" if dirty is False else "unknown",
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": _sha256(source_manifest),
            "source_manifest_files_sha256": manifest.get("sha256"),
        }


def _environment():
    packages = ("pymatgen", "numpy", "torch", "transformers", "tensorflow", "tensorpotential", "ase")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": versions,
    }


def _run_meta(args, *, started_at, seeds, inputs, code_files, dataset_dir, extras=None):
    _, _, manifest_sha256 = _manifest(dataset_dir)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "schema": 1,
        "command": sys.argv,
        "config": config,
        "git": _git_metadata(),
        "base_model": args.model,
        "checkpoint": args.checkpoint,
        "parameter_count": args.parameter_count,
        "dataset_manifest_sha256": manifest_sha256,
        "input_sha256": {str(path): _sha256(path) for path in inputs},
        "seeds": sorted(set(seeds)),
        "environment": _environment(),
        "code_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in code_files},
        "started_at": started_at,
        "ended_at": _timestamp(),
        **(extras or {}),
    }


def _json_default(value):
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_outputs(output_dir, *, summary, receipts, run_meta):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "receipts.jsonl").open("x", encoding="utf-8") as stream:
        for receipt in receipts:
            stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False, default=_json_default) + "\n")
    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _check_new_output(output_dir):
    if Path(output_dir).exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")


def _read_attempts(path):
    attempts = read_jsonl(path)
    if not attempts:
        raise ValueError(f"attempts JSONL is empty: {path}")
    _unique_by(attempts, "attempt_id", "attempts")
    for index, attempt in enumerate(attempts):
        if type(attempt.get("seed")) is not int:
            raise ValueError(f"attempt row {index} needs an integer seed")
        request = attempt.get("request")
        if (
            not isinstance(request, dict)
            or set(request) != {"space_group", "ehull_upper"}
            or type(request.get("space_group")) is not int
            or type(request.get("ehull_upper")) not in (int, float)
            or not math.isfinite(request["ehull_upper"])
        ):
            raise ValueError(f"attempt row {index} needs integer request.space_group and finite numeric ehull_upper")
        if attempt.get("output") is None:
            attempt["output"] = ""
        if not isinstance(attempt.get("output"), str):
            raise ValueError(f"attempt row {index} needs a string output")
    return attempts


def _code_files(*names):
    return [Path(__file__).with_name(name + ".py") for name in names]


def _understanding(args, started_at):
    dataset_rows = load_split_rows(args.dataset_dir, args.split)
    predictions = read_jsonl(args.predictions)
    joined = _join_predictions(dataset_rows, predictions)
    result = summarize_understanding(joined)
    grades = result.pop("grades")
    if len(grades) != len(joined):
        raise ValueError(f"Understanding evaluator returned {len(grades)} grades for {len(joined)} rows")
    receipts = [
        {"material_id": row["material_id"], "response": row["response"], "reference": row["reference"], "grade": grade}
        for row, grade in zip(joined, grades)
    ]
    seeds = [row["seed"] for row in predictions if type(row.get("seed")) is int]
    run_meta = _run_meta(
        args, started_at=started_at, seeds=seeds, inputs=[args.predictions],
        code_files=_code_files("cli", "references", "understanding"),
        dataset_dir=args.dataset_dir,
        extras={"split": args.split, "evaluated_rows": len(joined)},
    )
    _write_outputs(args.output, summary=result, receipts=receipts, run_meta=run_meta)


def _design(args, started_at):
    attempts = _read_attempts(args.attempts)
    plan_rows = read_jsonl(args.attempt_plan)
    attempts = _join_attempt_plan(attempts, plan_rows)
    inputs = [args.attempts, args.attempt_plan]
    if args.property_receipts is not None:
        attempts = _join_property_receipts(attempts, read_jsonl(args.property_receipts))
        inputs.append(args.property_receipts)
    else:
        _verify_inline_property_receipts(attempts)
    training_refs, test_refs = load_reference_banks(
        args.dataset_dir, extra_training_references=args.extra_training_references,
    )
    if args.extra_training_references is not None:
        inputs.append(args.extra_training_references)
    from .design import evaluate_design

    result = evaluate_design(
        attempts, training_references=training_refs, test_references=test_refs,
    )
    receipts = result["receipts"]
    expected_ids = [attempt["attempt_id"] for attempt in attempts]
    if [receipt.get("attempt_id") for receipt in receipts] != expected_ids:
        raise ValueError("design evaluator changed or lost raw attempt identity/order")
    summary = {
        "summary": result["summary"],
        "per_seed": result["per_seed"],
        "multiseed": result["multiseed"],
    }
    seeds = [attempt["seed"] for attempt in attempts]
    run_meta = _run_meta(
        args, started_at=started_at, seeds=seeds, inputs=inputs,
        code_files=_code_files("cli", "references", "legacy", "design"),
        dataset_dir=args.dataset_dir,
        extras={
            "training_reference_count": len(training_refs),
            "test_reference_count": len(test_refs),
            "attempt_count": len(attempts),
        },
    )
    _write_outputs(args.output, summary=summary, receipts=receipts, run_meta=run_meta)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for name in ("understanding", "design"):
        command = subparsers.add_parser(name)
        command.add_argument("--dataset-dir", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--model", required=True, help="base model or 'oracle' for a diagnostic")
        command.add_argument("--checkpoint", required=True, help="evaluated checkpoint identity/path")
        command.add_argument("--parameter-count", type=int)
        if name == "understanding":
            command.add_argument("--predictions", required=True, type=Path)
            command.add_argument("--split", choices=("training", "validation", "test"), default="test")
        else:
            command.add_argument("--attempt-plan", required=True, type=Path)
            command.add_argument("--attempts", required=True, type=Path)
            command.add_argument("--property-receipts", type=Path)
            command.add_argument("--extra-training-references", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    _check_new_output(args.output)
    started_at = _timestamp()
    if args.task == "understanding":
        _understanding(args, started_at)
    else:
        _design(args, started_at)


if __name__ == "__main__":
    main()

