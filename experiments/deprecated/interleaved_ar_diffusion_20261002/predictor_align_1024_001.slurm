#!/bin/bash
#SBATCH --job-name=crystal-align-1024
#SBATCH --partition=stingy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/predictor_align_1024_001-%j.out
#SBATCH --error=logs/predictor_align_1024_001-%j.err
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
    "adapt.py": "43d4ffccc86f4f8ce20922c4bb5592064990769f37ab7a7e935e1d4efe2c8caf",
    "align.py": "b55808efddeba49be24f1b13674fd7dc2ca6d0fd55f4eb31466c0103ace56ed5",
    "causal_diagnose.py": "3bf4be91dbe72849a26f60406b4571f5a7cb73be11fe4d07a6b86d876bf32602",
    "cohort.py": "ead8877f3edfc0079d9b6797c6829289dea632cfb1089148271bf80da6094593",
    "diagnose.py": "674ec4ec5a6c2cdd78c58e51a288a066a7bd6d52ac5e81d7b6bd3a9f4b2a7772",
    "test_checks.py": "13228a70ebc2cd00ca0fac4111a0715cc27334da3941fe6b373b9447498c68c8",
    "predictor_align.py": "fcdea0b94afbe509f556b76a2eab77e973827fa680ca63bffa95e51cc6144ebf",
    "prepare.py": "864927e1c27d3e3baa8107eba2db09d320d6ce3ee17b6c801370d85fd63ba688",
    "rebalance.py": "8f689794ae1f8e512c0a705a6fcedd58060633dfa16cb9e3277193efb9e15ae6",
    "recover.py": "3f216daace0a156754b982d6fc0319d42a11d44898861f873113ad8533bc4db1",
    "rescore.py": "254192f4e2482f331fb42d95591f98bccdcc1c65f2bd8c6951faca194560e2ad",
    "run.py": "889e8f2a6083bcc4d04c5d0b646212d131f247897f71df7e522d1fd838abc143",
    "score.py": "23b657ca2128119e9b19538a2ce73b0cc122c130623141bc334e6e416781d161",
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
print("V9 reviewed source and fixed AR initialization verified")
PY

exec env/bin/python -u predictor_align.py --data data_002 --initial rebalance_001/ar --out predictor_align_1024_001
