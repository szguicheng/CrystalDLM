"""Freeze audited PLaID materials, edits and prefix-target examples."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import warnings

from pymatgen.analysis.structure_matcher import StructureMatcher
from transformers import AutoTokenizer
from score import PROJECT, answer, audit, edit, parse, parse_fn_wyckoff
from utils import get_crystal_string_wyckoff_pyx

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
START, END, MASK, PAD = ('<CRYSTAL_START>', '<CRYSTAL_END>', '<CRYSTAL_MASK>', '<CRYSTAL_PAD>')
K, STEPS, MAXLEN = 256, 32, 1536
HASHES = dict(train='133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d',
              val='7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29',
              test='dd605c7543cf95a13e150aed630d0915e74edc8dea17e85301ee6fbfeb7f500a')
EOS = '<|im_end|>'
TEMPLATES = {
    'train': [('Generate a crystal with cell composition {formula} and space group {sg}.',
               'Read the current crystal. Report its composition, space group number, atom count, independent orbit count, and lattice lengths a, b, c.',
               'Scale every lattice length by {scale}, keeping the angles, elements, Wyckoff sites and fractional coordinates unchanged.'),
              ('Create a crystal whose unit cell composition is {formula}, using space group {sg}.',
               'Describe the actual crystal: give its formula, space group number, cell atoms, independent orbits, and a, b, c.',
               'Multiply a, b and c by {scale}. Preserve all other crystal fields.')],
    'val': [('Produce a crystal in space group {sg}, with cell composition {formula}.',
             'For the supplied crystal, state the formula, space group number, total atoms, independent orbits, and the three lattice lengths.',
             'Apply a factor of {scale} to all lattice lengths; leave all other fields fixed.')],
    'test': [('Design a unit cell containing {formula} in space group {sg}.',
              'Inspect the current structure and report its composition, space group number, atoms per cell, independent orbits, and a, b, c.',
              'Change each of the three lattice lengths to {scale} times its current value, without changing any other field.')],
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.add_special_tokens({'additional_special_tokens': [START, END, MASK, PAD]})
    assert tok.eos_token == EOS
    return tok


def block(text):
    return START + text + END


def edit_factor(direction):
    return {'plus': '1.05', 'minus': '0.95'}[direction]


def tasks(row, split, direction, variant=0):
    gen, read, modify = TEMPLATES[split][variant % len(TEMPLATES[split])]
    condition = row['condition']
    scale = edit_factor(direction)
    prefix = 'User: ' + gen.format(**condition) + '\nAssistant:'
    trigger = ' I will generate the requested crystal.\n' + START
    history = prefix + trigger + row['c1'] + END + '\nUser: ' + read + '\nAssistant:'
    response1 = ' ' + answer(row['c1']) + EOS
    editing = history + response1 + '\nUser: ' + modify.format(scale=scale) + '\nAssistant:'
    edit_trigger = ' I will apply the requested change.\n' + START
    reading = editing + edit_trigger + row[direction] + END + '\nUser: ' + read + '\nAssistant:'
    return [dict(task='G', prefix=prefix, lead=trigger, crystal=row['c1']),
            dict(task='Q1', prefix=history, target=response1),
            dict(task='E', prefix=editing, lead=edit_trigger, crystal=row[direction]),
            dict(task='Q2', prefix=reading, target=' ' + answer(row[direction]) + EOS)]


def encoded_length(example, tok):
    if 'crystal' in example:
        return len(tok.encode(example['prefix'] + example['lead'], add_special_tokens=False)) + K
    return len(tok.encode(example['prefix'] + example['target'], add_special_tokens=False))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(out, sizes):
    out.mkdir(parents=True, exist_ok=False)
    tok = tokenizer()
    tok.save_pretrained(out / 'tokenizer')
    start = time.monotonic()
    matcher = StructureMatcher(scale=False)
    seen_ids, seen_text, seen_structures = set(), set(), defaultdict(list)
    accepted = {}
    manifest = dict(status='preparing', model=MODEL, revision=REVISION,
                    capacity=K, steps=STEPS, max_length=MAXLEN, source_hashes={}, sizes=sizes,
                    templates=TEMPLATES, packages={name: importlib.metadata.version(name)
                    for name in ['pyxtal', 'pymatgen', 'spglib', 'numpy', 'transformers', 'torch']},
                    code_hashes={p.name: sha(p.read_bytes()) for p in ROOT.glob('*.py')})
    for split, size in sizes.items():
        source = PROJECT / f'work/PLaID/data/basic/{split}.csv'
        source_hash = sha(source.read_bytes())
        assert source_hash == HASHES[split], f'{split} source changed'
        manifest['source_hashes'][split] = source_hash
        with source.open() as f:
            candidates = sorted(csv.DictReader(f), key=lambda r: sha(r['material_id'].encode()))
        records, reasons = [], Counter()
        with (out / f'{split}_scan.jsonl').open('w') as log:
            for index, row in enumerate(candidates):
                if len(records) == size:
                    break
                entry = dict(id=row['material_id'], source_row_rank_by_hash=index, split=split)
                try:
                    if row['material_id'] in seen_ids:
                        raise ValueError('duplicate_material_id')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        text = get_crystal_string_wyckoff_pyx(row['cif']).strip()
                        plus, minus = edit(text, '1.05'), edit(text, '.95')
                        checked = {key: audit(value) for key, value in [('c1', text), ('plus', plus), ('minus', minus)]}
                    for key, result in checked.items():
                        if not result['valid']:
                            raise ValueError(key + ':' + '|'.join(result['reasons']))
                    lengths = {key: len(tok.encode(value, add_special_tokens=False)) + 1
                               for key, value in [('c1', text), ('plus', plus), ('minus', minus)]}
                    if max(lengths.values()) > K:
                        raise ValueError('crystal_token_capacity')
                    p = parse(text)
                    record = dict(id=row['material_id'], split=split, c1=text, plus=plus, minus=minus,
                                  condition=dict(formula=p['formula'], sg=p['sg']),
                                  token_lengths=lengths, audits=checked, source_cif_sha256=sha(row['cif'].encode()))
                    for direction in ['plus', 'minus']:
                        for variant in range(len(TEMPLATES[split])):
                            if max(encoded_length(e, tok) for e in tasks(record, split, direction, variant)) > MAXLEN:
                                raise ValueError('trajectory_token_capacity')
                    if text in seen_text:
                        raise ValueError('duplicate_canonical_text')
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        structure = parse_fn_wyckoff(text, False)
                    key = structure.composition.reduced_formula
                    if any(matcher.fit(structure, old) for old in seen_structures[key]):
                        raise ValueError('duplicate_structure')
                    seen_structures[key].append(structure)
                    seen_ids.add(row['material_id']); seen_text.add(text)
                    records.append(record)
                    entry.update(status='accepted', token_lengths=lengths)
                except Exception as error:
                    reason = f'{type(error).__name__}:{error}'
                    reasons[reason] += 1
                    entry.update(status='excluded', reason=reason)
                log.write(json.dumps(entry, allow_nan=False) + '\n'); log.flush()
                if (index + 1) % 50 == 0 or len(records) == size:
                    progress = dict(split=split, scanned=index+1, accepted=len(records), target=size,
                                    elapsed_seconds=round(time.monotonic()-start, 2))
                    print(json.dumps(progress), flush=True)
                    write_json(out / 'progress.json', progress)
        with (out / f'{split}.jsonl').open('w') as f:
            for record in records:
                f.write(json.dumps(record, allow_nan=False) + '\n')
        assert len(records) == size, f'{split}: only {len(records)}/{size} eligible; see scan log'
        accepted[split] = records
        manifest.setdefault('selection', {})[split] = dict(selected=len(records), scanned=len(records)+sum(reasons.values()),
                exclusions=dict(reasons), ids=[r['id'] for r in records],
                sg_counts=dict(Counter(str(r['condition']['sg']) for r in records)))
        write_json(out / 'manifest.json', manifest)
    if sizes.get('test', 0):
        records = accepted['test']
        rng = random.Random(20261002)
        indices = list(range(len(records)))
        for _ in range(10000):
            rng.shuffle(indices)
            if all(sum(parse(records[i]['c1'])[field] != parse(records[j]['c1'])[field]
                       for field in ['counts', 'sg', 'atoms', 'orbits', 'lengths']) >= 2
                   for i, j in enumerate(indices)):
                break
        else:
            raise ValueError('cannot construct a valid crystal replacement permutation')
        write_json(out / 'swap.json', {records[i]['id']: records[j]['id'] for i, j in enumerate(indices)})
    manifest.update(status='frozen', elapsed_seconds=time.monotonic()-start,
                    dataset_hashes={s: sha((out/f'{s}.jsonl').read_bytes()) for s in sizes})
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'status':'frozen', 'sizes':sizes, 'out':str(out)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.out.resolve(), dict(train=8) if args.smoke else dict(train=1024, val=128, test=256))
