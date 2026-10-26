"""Next 512 hash-sorted official test IDs, disjoint from every earlier viewed case."""
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

from prepare import ROOT, INPUT, HASHES, process, wp


def main():
    out = ROOT / 'followup'; out.mkdir(exist_ok=True)
    path = INPUT / 'test.csv'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES['test']
    with path.open() as file: rows = list(csv.DictReader(file))
    rows.sort(key=lambda r: hashlib.sha256(r['material_id'].encode()).hexdigest())
    selected = rows[512:1024]
    previous = json.loads((ROOT / 'data/selection.json').read_text())['ids']
    previous = set(sum(previous.values(), []))
    assert len(selected) == 512 and not ({r['material_id'] for r in selected} & previous)
    (out / 'fresh_selection.json').write_text(json.dumps({'source_sha256': HASHES['test'],
        'selection': 'SHA256(material_id), official test rows 512:1024, selected before generation',
        'ids': [r['material_id'] for r in selected]}, indent=2))
    wp.reference_table()
    with mp.get_context('fork').Pool(4) as pool:
        records = list(pool.imap(process, [('fresh', r) for r in selected], chunksize=4))
    vocab = set(json.loads((ROOT / 'data/vocab.json').read_text()))
    unknown = []
    for r in records:
        if r['status'] == 'ok':
            for name, p in r['packed'].items():
                for token in set(p['tokens']) - vocab: unknown.append([r['id'], name, token])
    (out / 'fresh_records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    summary = {'selected': 512, 'coverage': {s: sum(r['status'] == s for r in records) for s in ['ok', 'failed', 'overlength']},
               'unknown_tokens': unknown, 'disjoint_from_previous': True}
    (out / 'fresh_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
