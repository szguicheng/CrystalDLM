"""Pinned split, paired canonical structures, unchanged SP1/Wyckoff codecs."""
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import warnings
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent / 'codec'))
import numpy as np
import spglib
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifWriter
import program_codec as sp
import wyckoff_baseline as wp
from test_program import parse, compare

ROOT = Path(__file__).resolve().parent
INPUT = Path('/scratch/chenggui2/CrystalDLM/tokenizer_search_20260908/input')
HASHES = {'train': '133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d',
          'val': '7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29',
          'test': 'dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a'}
LIMIT = 256
NUMERIC = ['<LENGTH>', '<ANGLE>', '<COORD>', '<OFFSET>', '<DELTA>']
original_select = sp.select_generators


@lru_cache(None)
def cached_select(keys, denominator):
    return original_select(keys, denominator)


sp.select_generators = lambda keys, denominator: cached_select(tuple(sorted(keys)), denominator)


def composition(structure):
    counts = np.bincount(structure.atomic_numbers, minlength=119)[1:119]
    divisor = math.gcd(*map(int, counts[counts > 0]))
    return (counts // divisor).tolist()


def normalize(kind, value):
    if kind == '<LENGTH>': return math.log(value)
    if kind == '<ANGLE>': return math.log(value / (180 - value))
    if kind == '<COORD>': return value - 0.5
    if kind == '<DELTA>': return value * 1e4
    return value


def pack(text, representation):
    tokens, numbers = [], []
    family_members = family_params = residuals = 0
    for line in text.strip().splitlines():
        parts = line.split()
        kinds = {}
        if parts[0] == 'CELL':
            kinds = {i: '<LENGTH>' if i <= 3 else '<ANGLE>' for i in range(1, 7)}
        elif parts[0] == 'ORIGIN': kinds = {i: '<OFFSET>' for i in range(1, 4)}
        elif parts[0] == 'RESIDUALS': residuals = int(parts[1])
        elif residuals:
            kinds = {i: '<DELTA>' for i in range(1, 4)}
            residuals -= 1
        elif representation == 'wyckoff' and parts[0] in ELEMENTS:
            kinds = {i: '<COORD>' for i in range(2, len(parts))}
        elif parts[0] == 'FAMILY':
            family_members, family_params = int(parts[1]), parts.count('PARAM')
        elif family_members:
            assert len(parts) == family_params + 1
            kinds = {i: '<COORD>' for i in range(1, len(parts))}
            family_members -= 1
        else:
            kinds = {i + 1: '<COORD>' for i, token in enumerate(parts) if token == 'FREE'}
        for i, token in enumerate(parts):
            kind = kinds.get(i)
            tokens.append(kind or token)
            numbers.append(normalize(kind, float(token)) if kind else 0.0)
    assert not family_members and not residuals and tokens[-1] == 'END'
    return tokens, numbers


ELEMENTS = {Element.from_Z(z).symbol for z in range(1, 119)}


def process(item):
    split, row = item
    record = {'id': row['material_id'], 'split': split, 'formula': row.get('pretty_formula')}
    try:
        with warnings.catch_warnings(), tempfile.TemporaryDirectory() as directory:
            warnings.simplefilter('ignore')
            path = Path(directory) / 'input.cif'
            path.write_text(row['cif'])
            raw = parse(path)
            cell = (raw.lattice.matrix, raw.frac_coords, raw.atomic_numbers)
            detected = spglib.get_symmetry_dataset(cell, symprec=1e-3)
            table = wp.reference_table()[int(detected.number)]
            ds = spglib.get_symmetry_dataset(cell, hall_number=table['hall'], symprec=1e-3)
            conventional = Structure(ds.std_lattice, ds.std_types, np.round(ds.std_positions, 6) % 1)
            CifWriter(conventional, symprec=None, significant_figures=14).write_file(path)
            _, baseline, audit = wp.encode_cif(path)
            # Both targets describe precisely the same idealized standard structure.
            canonical = wp.decode(baseline)
            CifWriter(canonical, symprec=None, significant_figures=14).write_file(path)
            reference = parse(path)
            explicit, meta = sp.encode_cif(path)
            err_sp = compare(reference, sp.decode(explicit))
            err_wp = compare(reference, wp.decode(baseline))
            record.update(sg=meta['space_group'], raw_sg=int(detected.number),
                          composition=composition(reference), atoms=len(reference),
                          reference={'lattice': reference.lattice.matrix.tolist(),
                                     'species': list(map(int, reference.atomic_numbers)),
                                     'positions': reference.frac_coords.tolist()},
                          oracle_errors_A={'explicit': err_sp, 'wyckoff': err_wp},
                          residual_atoms=meta['residual_atoms'], programs={}, packed={})
            for name, text in [('explicit', explicit), ('wyckoff', baseline)]:
                tokens, values = pack(text, name)
                record['programs'][name] = text
                record['packed'][name] = {'tokens': tokens, 'values': values}
            record['lengths'] = {name: len(data['tokens']) for name, data in record['packed'].items()}
            record['status'] = 'ok' if max(record['lengths'].values()) <= LIMIT else 'overlength'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    return record


def main():
    began = time.time()
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    items, ids = [], {}
    for split, count in [('train', 6000), ('val', 512), ('test', 512)]:
        path = INPUT / f'{split}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[split]
        with path.open() as file: rows = list(csv.DictReader(file))
        rows.sort(key=lambda r: hashlib.sha256(r['material_id'].encode()).hexdigest())
        selected = rows[:count]
        ids[split] = [row['material_id'] for row in selected]
        items.extend((split, row) for row in selected)
    assert len(set(sum(ids.values(), []))) == sum(map(len, ids.values()))
    (out / 'selection.json').write_text(json.dumps({'source_sha256': HASHES, 'ids': ids}, indent=2))
    wp.reference_table()  # Read-only table inherited by fork workers.
    records = []
    with mp.get_context('fork').Pool(4) as pool, (out / 'records.jsonl').open('w') as file:
        for i, record in enumerate(pool.imap(process, items, chunksize=4)):
            records.append(record)
            file.write(json.dumps(record) + '\n'); file.flush()
            if (i + 1) % 100 == 0:
                print(json.dumps({'prepared': i + 1, 'seconds': round(time.time() - began),
                                  'ok': sum(r['status'] == 'ok' for r in records)}), flush=True)
    training = [r for r in records if r['split'] == 'train' and r['status'] == 'ok']
    symbols = set(ELEMENTS) | set(NUMERIC) | {str(i) for i in range(-4, 769)}
    symbols |= {label for table in wp.reference_table().values() for label in table['definitions']}
    symbols |= {'SP1', 'WYCKOFF_STD1', 'CELL', 'SG', 'HALL', 'SITES', 'ROUND', 'GENERATORS',
                'G', 'ORIGIN', 'FAMILY', 'RESIDUALS', 'FREE', 'PARAM', 'ADD', 'NEG', 'MUL', 'x', 'y', 'z', 'END'}
    for record in training:
        for data in record['packed'].values(): symbols.update(data['tokens'])
    vocab = ['<PAD>', '<MASK>', '<UNK>'] + sorted(symbols)
    (out / 'vocab.json').write_text(json.dumps(vocab, indent=2))
    unknown = []
    for r in records:
        if r['status'] == 'ok':
            for name, data in r['packed'].items():
                missing = sorted(set(data['tokens']) - set(vocab))
                if missing: unknown.append({'id': r['id'], 'representation': name, 'tokens': missing})
    summary = {'seconds': time.time() - began, 'vocab_size': len(vocab), 'unknown_tokens': unknown,
               'coverage': {s: {status: sum(r['split'] == s and r['status'] == status for r in records)
                               for status in ['ok', 'failed', 'overlength']} for s in ids},
               'mean_lengths': {name: float(np.mean([r['lengths'][name] for r in training]))
                                for name in ['explicit', 'wyckoff']},
               'failures': [{k: v for k, v in r.items() if k in ['id', 'split', 'error', 'lengths', 'status']}
                            for r in records if r['status'] != 'ok']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
