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


def process(item, encoder_symprec=1e-3):
    split, row = item
    record = {'id': row['material_id'], 'split': split, 'formula': row.get('pretty_formula'),
              'encoder_symprec': encoder_symprec}
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
            _, baseline, audit = wp.encode_cif(path, symprec=encoder_symprec)
            # Both targets describe precisely the same idealized standard structure.
            canonical = wp.decode(baseline)
            CifWriter(canonical, symprec=None, significant_figures=14).write_file(path)
            reference = parse(path)
            explicit, meta = sp.encode_cif(path, symprec=encoder_symprec)
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
        if encoder_symprec == 1e-3 and str(error).startswith('No standard Wyckoff template matches'):
            # Verified Ho5Pt3 boundary case: standardization triggers approximate
            # SG 63 -> 193 promotion. A tighter common encoder tolerance retains
            # SG 63 and passes both exact reconstruction checks, without dropping it.
            retried = process(item, encoder_symprec=5e-4)
            retried['symmetry_retry_reason'] = str(error)
            return retried
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    return record


def main():
    began = time.time()
    out = ROOT / 'data'; out.mkdir(exist_ok=True)
    items, ids = [], {}
    expected = {'train': 27136, 'val': 9047, 'test': 9046}
    for split in ['train', 'val', 'test']:
        path = INPUT / f'{split}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[split]
        with path.open() as file: rows = list(csv.DictReader(file))
        rows.sort(key=lambda r: hashlib.sha256(r['material_id'].encode()).hexdigest())
        assert len(rows) == expected[split], (split, len(rows))
        selected = rows if split != 'test' else rows[1024:1536]
        ids[split] = [row['material_id'] for row in selected]
        items.extend((split, row) for row in selected)
    assert len(set(sum(ids.values(), []))) == sum(map(len, ids.values()))
    (out / 'selection.json').write_text(json.dumps({'source_sha256': HASHES, 'ids': ids}, indent=2))
    pilot = Path('/scratch/chenggui2/CrystalDLM/explicit_model_pilot_20260922')
    original_cache = pilot / 'data/records.jsonl'
    assert hashlib.sha256(original_cache.read_bytes()).hexdigest() == '18f153289955f12b1dd4460b46fee1cfc7121189e5645589d252bfa55d1a9d5b'
    cache_path = out / 'records.jsonl' if (out / 'records.jsonl').exists() else original_cache
    for name in ['program_codec.py', 'wyckoff_baseline.py']:
        assert (ROOT / 'codec' / name).read_bytes() == (pilot / 'codec' / name).read_bytes()
    with cache_path.open() as file:
        cache = {r['id']: r for r in map(json.loads, file) if r['status'] in ['ok', 'overlength']}
    known = []
    pending = []
    for split, row in items:
        if row['material_id'] in cache:
            record = cache[row['material_id']]
            assert record['split'] == split
            known.append(record)
        else: pending.append((split, row))
    wp.reference_table()  # Read-only table inherited by fork workers.
    records = list(known)
    print(json.dumps({'cached': len(known), 'to_prepare': len(pending)}), flush=True)
    with mp.get_context('fork').Pool(4) as pool, (out / 'records.jsonl').open('w') as file:
        for record in known: file.write(json.dumps(record) + '\n')
        for i, record in enumerate(pool.imap(process, pending, chunksize=4)):
            records.append(record)
            file.write(json.dumps(record) + '\n'); file.flush()
            if (i + 1) % 100 == 0:
                print(json.dumps({'prepared': i + 1, 'seconds': round(time.time() - began),
                                  'ok': sum(r['status'] == 'ok' for r in records)}), flush=True)
    failures = [r for r in records if r['status'] == 'failed']
    (out / 'encoding_failures.json').write_text(json.dumps(failures, indent=2))
    assert not failures, f'{len(failures)} encoding failures; refusing to call a filtered subset full MP-20'
    training = [r for r in records if r['split'] == 'train']
    maximum = max(max(r['lengths'].values()) for r in training)
    canvas = max(256, 32 * math.ceil(maximum / 32))
    for r in records:
        assert max(r['lengths'].values()) <= canvas, (r['id'], r['lengths'], canvas)
        r['status'] = 'ok'
    records.sort(key=lambda r: (['train', 'val', 'test'].index(r['split']), hashlib.sha256(r['id'].encode()).hexdigest()))
    (out / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
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
               'max_length': canvas, 'maximum_training_length': maximum, 'cached_records': len(known),
               'encoder_tolerance_exceptions': [{'id': r['id'], 'encoder_symprec': r['encoder_symprec'],
                    'reason': r['symmetry_retry_reason']} for r in records if 'symmetry_retry_reason' in r],
               'coverage': {s: {status: sum(r['split'] == s and r['status'] == status for r in records)
                               for status in ['ok', 'failed', 'overlength']} for s in ids},
               'mean_lengths': {name: float(np.mean([r['lengths'][name] for r in training]))
                                for name in ['explicit', 'wyckoff']},
               'failures': [{k: v for k, v in r.items() if k in ['id', 'split', 'error', 'lengths', 'status']}
                            for r in records if r['status'] != 'ok']}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
