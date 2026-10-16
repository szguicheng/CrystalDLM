"""Fresh scaled32 run with conflicting readers grounded in parent history."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SCALED_ROOT = ROOT
JOINT_ROOT = PROJECT / "experiments/interleaved_relational_joint_20261006"
FROZEN_TRAIN = SCALED_ROOT / "train.py"
FROZEN_TRAIN_SHA256 = "1bb619fb809d71c68813eb07aa162b2b2bfa1b8f2de7cef1a67b60054d4ae816"
BASE_ALIAS = "_frozen_scaled32_train_20261006"

for import_root in (str(SCALED_ROOT), str(JOINT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

frozen_hash = hashlib.sha256(FROZEN_TRAIN.read_bytes()).hexdigest()
assert frozen_hash == FROZEN_TRAIN_SHA256, f"frozen scaled32 trainer changed: {frozen_hash}"
spec = importlib.util.spec_from_file_location(BASE_ALIAS, FROZEN_TRAIN)
BASE = importlib.util.module_from_spec(spec)
sys.modules[BASE_ALIAS] = BASE
spec.loader.exec_module(BASE)

runner = BASE.runner
UPDATES = BASE.UPDATES
READER_RECIPE = (
    "Fresh grounded run: each train case receives 16 neutral and 16 conflicting contexts per reader task; "
    "for Q2, each reader direction receives 8 neutral and 8 full-parent-history contexts. "
    "Q1 conflicts replace the parent design conditions with a cyclic training partner's conditions "
    "while preserving the current case's C1 and target; Q2 retains its same-case G-C1-Q1-E-C2 history."
)
READER_DIRECTION_NOTE = (
    "Parent Q1/Q2 task and direction schedule is unchanged. Per train case Q1 has 16 neutral and 16 conflict; "
    "each Q2 direction has 8 neutral and 8 full-history conflict examples."
)

_ORIGINAL_READER_PREFIX = BASE._reader_prefix


def _reader_prefix(base, row, partner, direction, mode, context_mode):
    if mode == "plain" or context_mode != "conflict":
        return _ORIGINAL_READER_PREFIX(base, row, partner, direction, mode, context_mode)

    latest = row["c1"] if base["task"] == "Q1" else row[direction]
    if base["task"] == "Q2":
        return base["prefix"], latest

    suffix = runner._read_question_suffix(base["prefix"], row["c1"])
    partner_generation = runner.prepare.tasks(partner, "train", direction, 0)[0]
    question = (
        "Read the current replacement crystal, ignoring the earlier design conditions. "
        + runner._read_text_question(suffix)
    )
    prefix = (
        partner_generation["prefix"] + " I generated a crystal.\n" + runner.START
        + row["c1"] + runner.END + "\nUser: " + question + "\nAssistant:"
    )
    return prefix, latest


BASE._reader_prefix = _reader_prefix
BASE.READER_RECIPE = READER_RECIPE
BASE.READER_DIRECTION_NOTE = READER_DIRECTION_NOTE

_original_source_hashes = runner.runner_source_hashes


def _pinned_source_hashes():
    hashes = _original_source_hashes()
    for path in (Path(__file__), ROOT / "train_full_history.slurm", ROOT / "test_train_full_history.py"):
        key = Path(os.path.relpath(path, JOINT_ROOT)).as_posix()
        hashes[key] = runner.sha(path.read_bytes())
    return dict(sorted(hashes.items()))


runner.runner_source_hashes = _pinned_source_hashes


def run_training(out=ROOT / "training_full_history_001"):
    return BASE.run_training("grounded", out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training_full_history_001")
    args = parser.parse_args(argv)
    if not runner.torch.cuda.is_available():
        raise SystemExit("training requires the requested Slurm A100 allocation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("refusing to train outside Slurm")
    summary = run_training(args.out)
    print(json.dumps({"status": "training_only_checkpoint_verified", "summary": str(summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
