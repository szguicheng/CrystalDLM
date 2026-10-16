"""Run one deterministic structure shard per allocated CPU and record progress."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Full structure processing requires a Slurm allocation"
    stage = Path(__file__).resolve().parent
    root = stage.parents[1]
    source = root / "research/mp_ehull030_20261011/collection_001/data_001/records.jsonl.gz"
    shards = int(os.environ["SLURM_CPUS_PER_TASK"])
    output = stage / os.environ.get("CRYSTAL_FACTS_RUN", "full_001")
    output.mkdir(exist_ok=True)
    started = time.monotonic()

    def run(index):
        target = output / f"shard-{index:03d}.jsonl"
        if target.exists():
            return {"shard": index, "status": "existing"}
        command = [sys.executable, str(stage / "preprocess.py"), "--input", str(source),
                   "--output", str(target), "--shard-index", str(index), "--shards", str(shards)]
        with (output / f"shard-{index:03d}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        return {"shard": index, "exit_code": result.returncode}

    results = []
    with ThreadPoolExecutor(max_workers=shards) as pool:
        for future in as_completed([pool.submit(run, i) for i in range(shards)]):
            result = future.result()
            results.append(result)
            print(json.dumps({**result, "completed_shards": len(results), "total_shards": shards,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    summary = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "node": os.environ.get("SLURMD_NODENAME"),
               "shards": shards, "elapsed_seconds": round(time.monotonic() - started, 1),
               "results": sorted(results, key=lambda r: r["shard"])}
    (output / "execution.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(r.get("exit_code", 0) == 0 for r in results), "One or more preprocessing shards failed"


if __name__ == "__main__":
    main()
