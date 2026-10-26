"""Resume missing Design captions with the existing approved annotation client."""
import argparse
import json
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[1]
sys.path.insert(0, str(ROOT / "research/crystal_bulk_annotation_20261013"))
from batch import BatchRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    runner = BatchRunner(inputs=[STAGE / "annotation_inputs" / f"{s}-facts.jsonl.gz"
                                 for s in ("train", "val", "test")],
                         run_dir=STAGE / "annotation_run", workers=8)
    try:
        status = runner.run(limit=args.limit)
        print(json.dumps(status), flush=True)
    finally:
        runner.export()
    if args.limit is None:
        assert status["successes"] == 37293 and status["failed_materials"] == 0, status
