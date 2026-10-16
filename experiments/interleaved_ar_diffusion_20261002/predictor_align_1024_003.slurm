#!/bin/bash
#SBATCH --job-name=crystal-align-1024-003
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/predictor_align_1024_003-%j.out
#SBATCH --error=logs/predictor_align_1024_003-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1

env/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

expected = {
    "model.py": "036ac4d4fc08437e2325c31265fceb08a573d8c703a1ca551c4f921541345406",
    "adapt.py": "6ebd2b152dcd636391b73df8f612f7f13aab39360b9f47e4e6995fdec683cd58",
    "align.py": "b55808efddeba49be24f1b13674fd7dc2ca6d0fd55f4eb31466c0103ace56ed5",
    "causal_diagnose.py": "3bf4be91dbe72849a26f60406b4571f5a7cb73be11fe4d07a6b86d876bf32602",
    "cohort.py": "ead8877f3edfc0079d9b6797c6829289dea632cfb1089148271bf80da6094593",
    "diagnose.py": "674ec4ec5a6c2cdd78c58e51a288a066a7bd6d52ac5e81d7b6bd3a9f4b2a7772",
    "test_checks.py": "b6195f633283a0df8ceff33992645776e10d32a9894803a5d43592baa80a819f",
    "predictor_align.py": "8567c5724d115ad1d999e50886b2fc65f919920708acde5fc4e531b710a8a781",
    "prepare.py": "864927e1c27d3e3baa8107eba2db09d320d6ce3ee17b6c801370d85fd63ba688",
    "rebalance.py": "8f689794ae1f8e512c0a705a6fcedd58060633dfa16cb9e3277193efb9e15ae6",
    "recover.py": "3f216daace0a156754b982d6fc0319d42a11d44898861f873113ad8533bc4db1",
    "rescore.py": "254192f4e2482f331fb42d95591f98bccdcc1c65f2bd8c6951faca194560e2ad",
    "run.py": "889e8f2a6083bcc4d04c5d0b646212d131f247897f71df7e522d1fd838abc143",
    "score.py": "c1dfe1b87715b5b650b7152e02fed567bd5f7f0fd9109295ec87208f23165c6b",
}
for name, digest in expected.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name + " changed after review"
data_sha = hashlib.sha256(Path("data_002/manifest.json").read_bytes()).hexdigest()
v8_manifest = json.loads(Path("rebalance_001/manifest.json").read_text())
assert data_sha == "e3a5a86e58eaae871fa2718d3c9f646551cb297047ef724bf841b24bc48c0004"
assert v8_manifest["source_manifest_sha256"] == data_sha
status = json.loads(Path("rebalance_001/ar/status.json").read_text())
assert status["status"] == "validated" and status["completed_updates"] == 4096
assert status["final_fingerprint"] == "7567d5dd878a1e02a51b05a9ec4ecd67e68cd1a79da88235c975e1eeb4b21c85"
print("Fixed 002 continuation sources and V8 AR checkpoint verified")
PY

exec env/bin/python -u predictor_align.py --data data_002 --initial rebalance_001/ar --out predictor_align_1024_003
