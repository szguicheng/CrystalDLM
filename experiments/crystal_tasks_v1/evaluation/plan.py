"""Freeze attempt identities from the published design test requests."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from ..configuration import protocol


def make_plan(rows, seeds, attempts_per_condition):
    conditions=set()
    for row in rows:
        match=re.fullmatch(r'Generate a crystal with space group number (\d+) and energy above hull no greater than ([\d.]+) eV/atom\.',row['instruction'])
        if match is None or row['input']:
            raise ValueError('Design request differs from the frozen SG/ehull schema')
        conditions.add((int(match[1]),float(match[2])))
    return [{'attempt_id':f'c{index:03d}-s{seed}-a{attempt:03d}', 'condition_id':f'c{index:03d}',
        'seed':seed,'attempt_index':attempt,'request':{'space_group':sg,'ehull_upper':bound}}
        for index,(sg,bound) in enumerate(sorted(conditions)) for seed in seeds
        for attempt in range(attempts_per_condition)]


if __name__=='__main__':
    from .references import load_task_rows
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=protocol()['evaluation']
    rows=load_task_rows(args.dataset_dir,'design','test')
    plan=make_plan(rows,p['seeds'],p['attempts_per_condition_per_seed'])
    with args.output.open('x') as stream:
        for row in plan:
            stream.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps({'attempts':len(plan),'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
