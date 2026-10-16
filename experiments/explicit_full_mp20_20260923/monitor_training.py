"""Lightweight JSON status reader; contains no model or dataset computation."""
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parent
job = sys.argv[1]
seen = {}
while True:
    complete = 0
    for rep in ['explicit', 'wyckoff']:
        path = root / f'runs/{rep}_17/status.json'
        if not path.exists(): continue
        try: state = json.loads(path.read_text())
        except json.JSONDecodeError: continue  # A status write can briefly overlap this read.
        token = (state.get('state'), state.get('epoch', state.get('epochs')))
        if seen.get(rep) != token:
            out = {'representation': rep, **{k: state.get(k) for k in ['state', 'epoch', 'epochs', 'target_epochs', 'step', 'steps', 'elapsed_seconds', 'training_loss', 'best_epoch']}}
            if 'validation' in state:
                out['validation_loss'] = state['validation']['loss']
                out['content_accuracy'] = state['validation']['masked_content_accuracy']
            print(json.dumps(out), flush=True); seen[rep] = token
        complete += state.get('state') == 'complete'
    if complete == 2: break
    receipt = root / f'compute_allocations/{job}/finished/03_train/receipt.json'
    if receipt.exists() and json.loads(receipt.read_text())['exit_code'] != 0:
        print(receipt.read_text(), flush=True)
        raise SystemExit(1)
    time.sleep(10)
