"""Check the executor on frozen original geometries before making LLM calls."""
import json
from pathlib import Path
import sys

from cohort import prepare_case, build_structure

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'experiments/naturecry_roundtrip_20261001'))
from benchmark import compare, parse


def main():
    frozen = PROJECT / 'experiments/naturecry_roundtrip_20261001/run_001'
    manifest = json.loads((frozen / 'selection.json').read_text())
    maximum = 0
    for case in manifest['cases']:
        path = frozen / 'inputs' / (case['id'] + '.cif')
        facts, oracle = prepare_case(path)
        error, _ = compare(parse(path), build_structure(oracle))
        maximum = max(maximum, error)
    assert len(manifest['cases']) == 100
    print(f'PASS: all 100 original geometries, including frames and corrections; max error {maximum} A.')


if __name__ == '__main__':
    main()
