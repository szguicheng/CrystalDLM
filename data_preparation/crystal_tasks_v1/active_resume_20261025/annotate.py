"""Resume the frozen Design caption queue with the existing annotation client."""
import json
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
sys.path.insert(0, str(ROOT / "research/crystal_bulk_annotation_20261013"))
from batch import BatchRunner


if __name__ == "__main__":
    runner = BatchRunner(
        inputs=[STAGE / "annotation_inputs" / f"{split}-facts.jsonl.gz"
                for split in ("train", "val", "test")],
        run_dir=STAGE / "annotation_run",
        config=ROOT / "research/crystal_sharing_20261012/annotation-flash-config.json",
        prompt=ROOT / "research/crystal_sharing_20261012/description-prompt.md",
        env=ROOT / ".env",
        workers=8,
        max_attempts=4,
    )
    try:
        status = runner.run()
    finally:
        runner.export()
    print(json.dumps(status), flush=True)
    assert status["successes"] == 37293 and status["failed_materials"] == 0, status
