"""Fresh full-history run with the joint denoiser block size set to 4."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
JOINT_ROOT = ROOT.parents[1] / "experiments/interleaved_relational_joint_20261006"
FULL_HISTORY_PATH = ROOT / "train_full_history.py"
FROZEN_FULL_HISTORY_SHA256 = "e2954cf3eea94639782987596799a772b475d6a3b62eada787ad8a4820633112"
assert hashlib.sha256(FULL_HISTORY_PATH.read_bytes()).hexdigest() == FROZEN_FULL_HISTORY_SHA256
spec = importlib.util.spec_from_file_location("_frozen_scaled32_full_history_b4", FULL_HISTORY_PATH)
FULL_HISTORY = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FULL_HISTORY
spec.loader.exec_module(FULL_HISTORY)

BASE = FULL_HISTORY.BASE
runner = FULL_HISTORY.runner
B4_ARM = "joint_ar_diffusion_b4"
runner.ARMS = (B4_ARM,)
runner.MODE_OFFSETS = {B4_ARM: 0}
runner.CORE_MODES = {B4_ARM: "hybrid"}
runner.BLOCK_SIZES = {B4_ARM: 4}

_ORIGINAL_SOURCE_HASHES = runner.runner_source_hashes


def _source_hashes():
    hashes = _ORIGINAL_SOURCE_HASHES()
    for path in (Path(__file__), ROOT / "train_full_history_b4.slurm", ROOT / "test_train_full_history_b4.py"):
        key = Path(os.path.relpath(path, JOINT_ROOT)).as_posix()
        hashes[key] = runner.sha(path.read_bytes())
    return dict(sorted(hashes.items()))


runner.runner_source_hashes = _source_hashes


def run_training(out=ROOT / "training_full_history_b4_001"):
    return BASE.run_training("grounded", out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "training_full_history_b4_001")
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
