"""A small real-geometry check for the finite LLM decoder output format."""
from pathlib import Path
import sys

from pilot import build_structure

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/naturecry_roundtrip_20261001'))
from benchmark import compare, parse


def main():
    plan = dict(status='ok', space_group_number=62, hall_symbol='-P 2ac 2n',
                lattice=[7.11, 5.33, 9.75, 90, 90, 90], composition={'Tb': 8, 'Mg': 4},
                orbits=[dict(element='Tb', wyckoff='4c', representative=[0.818, 0.25, 0.084]),
                        dict(element='Tb', wyckoff='4c', representative=[0.467, 0.25, 0.728]),
                        dict(element='Mg', wyckoff='4c', representative=[0.247, 0.25, 0.124])])
    structure = build_structure(plan)
    compare(parse(ROOT / 'research/plaid_20260930/released_0_decoded.cif'), structure)
    plan['orbits'][0]['wyckoff'] = '4a'
    try:
        build_structure(plan)
    except ValueError:
        pass
    else:
        raise AssertionError('A wrong Wyckoff label was accepted')
    print('PASS: exact geometry reconstruction and conflicting orbit constraints rejected.')


if __name__ == '__main__':
    main()
