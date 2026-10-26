"""Write one pinned GRACE property receipt for every raw design attempt."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from . import legacy
from .cli import (
    ROOT,
    _attempt_hashes,
    _check_new_output,
    _environment,
    _git_metadata,
    _json_default,
    _read_attempts,
    _sha256,
)
from .references import _manifest


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _selected_gpu():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not value or value == "-1" or "," in value:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one selected GPU before GRACE scoring")
    return value


def _receipt(attempt, parser, grace, calculator, hull):
    hashes = _attempt_hashes(attempt)
    base = {
        "attempt_id": attempt["attempt_id"],
        "seed": attempt["seed"],
        **hashes,
    }
    try:
        body = legacy.extract_crystal(attempt["output"])
        structure = parser.parse_fn_wyckoff(body, to_cif=False)
        if structure is None:
            raise ValueError("PLaID parser returned no structure")
    except Exception as error:
        return {
            **base,
            "property_receipt": {
                "score_status": "decode_error",
                "error": f"{type(error).__name__}: {error}",
            },
        }

    try:
        audit = parser.audit(body)
    except Exception as error:
        return {
            **base,
            "property_receipt": {
                "score_status": "audit_error",
                "error": f"{type(error).__name__}: {error}",
            },
        }
    if audit.get("valid") is not True:
        return {
            **base,
            "property_receipt": {
                "score_status": "invalid_structure",
                "strict_valid": False,
                "strict_audit": audit,
            },
        }
    try:
        score = grace._score_structure(structure, calculator, hull)
        return {
            **base,
            "property_receipt": {**score, "strict_valid": True, "strict_audit": audit},
        }
    except Exception as error:
        return {
            **base,
            "property_receipt": {
                "score_status": "error",
                "strict_valid": True,
                "strict_audit": audit,
                "error": f"{type(error).__name__}: {error}",
            },
        }


def _write_run_meta(path, run_meta):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(run_meta, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_receipt(stream, row):
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False, default=_json_default) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _error_receipt(attempt, status, error):
    return {
        "attempt_id": attempt["attempt_id"],
        "seed": attempt["seed"],
        **_attempt_hashes(attempt),
        "property_receipt": {
            "score_status": status,
            "error": f"{type(error).__name__}: {error}",
        },
    }


def _finish_run_meta(run_meta, statuses, setup_error=None):
    run_meta.update({
        "status": "failed" if setup_error else "complete",
        "property_evaluation_available": setup_error is None,
        "completed_receipts": sum(statuses.values()),
        "receipt_status_counts": dict(statuses),
        "ended_at": _timestamp(),
    })
    if setup_error:
        run_meta["setup_error"] = f"{type(setup_error).__name__}: {setup_error}"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, help="generation model identity")
    parser.add_argument("--checkpoint", required=True, help="generation checkpoint identity/path")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    selected_gpu = _selected_gpu()
    _check_new_output(args.output)
    started_at = _timestamp()
    attempts = _read_attempts(args.attempts)
    scorer = legacy.load_grace_module()
    parser = legacy.load_legacy_audit()
    model_dir = scorer.ASSETS / "grace-model" / scorer.MODEL_NAME
    hull_path = scorer.HULL_PATH
    manifest_path, _, dataset_manifest_sha256 = _manifest(args.dataset_dir)
    model_manifest = json.loads(scorer.MODEL_MANIFEST.read_text(encoding="utf-8"))

    source_paths = {
        "evaluation_grace": Path(__file__).resolve(),
        "evaluation_cli": Path(__file__).with_name("cli.py").resolve(),
        "evaluation_legacy": Path(legacy.__file__).resolve(),
        "legacy_grace_scorer": Path(scorer.__file__).resolve(),
        "legacy_audit": Path(legacy.AUDIT_PATH).resolve(),
        "plaid_parser": (ROOT / "work/PLaID/utils.py").resolve(),
    }
    run_meta = {
        "schema": 1,
        "status": "running",
        "property_evaluation_available": False,
        "command": sys.argv,
        "config": {
            "attempts": str(args.attempts), "output": str(args.output),
            "dataset_dir": str(args.dataset_dir), "model": args.model,
            "checkpoint": args.checkpoint,
        },
        "git": _git_metadata(),
        "generation_model": args.model,
        "generation_checkpoint": args.checkpoint,
        "dataset_dir": str(args.dataset_dir),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "property_scorer": scorer.MODEL_NAME,
        "cuda_visible_devices": selected_gpu,
        "environment": _environment(),
        "input_sha256": {str(args.attempts): _sha256(args.attempts)},
        "attempt_count": len(attempts),
        "receipt_status_counts": {},
        "assets": {
            "model_dir": str(model_dir),
            "model_manifest": str(scorer.MODEL_MANIFEST),
            "model_manifest_sha256": _sha256(scorer.MODEL_MANIFEST),
            "model_manifest_content": model_manifest,
            "hull_path": str(hull_path),
            "hull_sha256": _sha256(hull_path),
        },
        "scoring_parameters": {
            "fmax_eV_per_A": scorer.FMAX,
            "max_steps": scorer.MAX_STEPS,
            "minimum_distance_A": scorer.MIN_DISTANCE,
            "symprec_A": scorer.SYMPREC,
            "angle_tolerance_deg": scorer.ANGLE_TOLERANCE,
        },
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "seeds": sorted({attempt["seed"] for attempt in attempts}),
        "started_at": started_at,
        "planned_receipts": len(attempts),
        "completed_receipts": 0,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    run_meta_path = args.output / "run_meta.json"
    _write_run_meta(run_meta_path, run_meta)

    statuses = Counter()
    setup_error = None
    with (args.output / "receipts.jsonl").open("x", encoding="utf-8") as stream:
        try:
            calculator = scorer._model_calculator(model_dir)
            hull = scorer._load_hull(hull_path)
        except Exception as error:
            setup_error = error
            for attempt in attempts:
                row = _error_receipt(attempt, "setup_error", error)
                _append_receipt(stream, row)
                statuses["setup_error"] += 1
                run_meta["completed_receipts"] = sum(statuses.values())
                run_meta["receipt_status_counts"] = dict(statuses)
        else:
            run_meta["property_evaluation_available"] = True
            _write_run_meta(run_meta_path, run_meta)
            for attempt in attempts:
                row = _receipt(attempt, parser, scorer, calculator, hull)
                _append_receipt(stream, row)
                statuses[row["property_receipt"]["score_status"]] += 1
                run_meta["completed_receipts"] = sum(statuses.values())
                run_meta["receipt_status_counts"] = dict(statuses)

    _finish_run_meta(run_meta, statuses, setup_error)
    _write_run_meta(run_meta_path, run_meta)
    if setup_error:
        raise RuntimeError("GRACE setup failed; per-attempt error receipts were persisted") from setup_error


if __name__ == "__main__":
    main()

