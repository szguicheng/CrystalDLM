"""One bounded retry round for the nine unfinished Design annotation requests."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

OUT = Path(__file__).resolve().parent
STAGE = OUT.parent
ROOT = STAGE.parents[2]
sys.path.insert(0, str(ROOT / 'research/crystal_bulk_annotation_20261013'))
from batch import BatchRunner


runner = BatchRunner(
    inputs=[STAGE / 'annotation_inputs' / f'{split}-facts.jsonl.gz' for split in ('train','val','test')],
    run_dir=STAGE / 'annotation_run',
    config=ROOT / 'research/crystal_sharing_20261012/annotation-flash-config.json',
    prompt=ROOT / 'research/crystal_sharing_20261012/description-prompt.md',
    env=ROOT / '.env', workers=8, max_attempts=4,
)
try:
    status = runner.run(retry_failed=True)
finally:
    runner.export()
con = sqlite3.connect(f'file:{STAGE}/annotation_run/annotation.sqlite3?mode=ro', uri=True)
before = set((OUT/'success_ids_before.txt').read_text().splitlines())
after = {row[0] for row in con.execute('select material_id from successes')}
expected = {row[0] for row in json.loads((OUT/'manifest.json').read_text())['failed_rows']}
assert before <= after and after-before <= expected
status.update(finished_at=datetime.now(timezone.utc).isoformat(), added_ids=sorted(after-before),
              previous_successes_preserved=len(before), complete=len(after)==37293)
(OUT/'result.json').write_text(json.dumps(status,indent=2)+'\n')
con.close()
print(json.dumps(status), flush=True)
assert status['complete'] and status['failed_materials']==0, status
