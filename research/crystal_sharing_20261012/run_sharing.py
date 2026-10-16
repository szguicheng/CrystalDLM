"""Augment frozen fact shards using one process per allocated CPU."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    assert os.environ.get("SLURM_JOB_ID"), "Full sharing extraction requires a Slurm CPU allocation"
    stage = Path(__file__).resolve().parent
    base = stage.parent / "crystal_preannotation_20261012/full_002"
    output = stage / "full_001"
    output.mkdir(exist_ok=True)
    inputs = sorted(base.glob("shard-*.jsonl"))
    assert len(inputs) == 32
    started = time.monotonic()

    def run(source):
        target = output / source.name
        assert not target.exists(), "Use a new output directory for a new method run"
        with (output / source.with_suffix(".log").name).open("w") as log:
            result = subprocess.run([sys.executable, str(stage / "augment.py"),
                "--input", str(source), "--output", str(target)], stdout=log, stderr=subprocess.STDOUT)
        return {"shard": source.name, "exit_code": result.returncode}

    results = []
    with ThreadPoolExecutor(max_workers=int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
        for future in as_completed([pool.submit(run, source) for source in inputs]):
            result = future.result()
            results.append(result)
            print(json.dumps({**result, "completed_shards": len(results),
                "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    (output / "execution.json").write_text(json.dumps({"job_id": os.environ["SLURM_JOB_ID"],
        "node": os.environ.get("SLURMD_NODENAME"), "elapsed_seconds": time.monotonic() - started,
        "results": results}, indent=2) + "\n")
    assert all(r["exit_code"] == 0 for r in results)


if __name__ == "__main__":
    main()
