"""One runnable roundtrip and meaning-changing text check, without an LLM."""
from pathlib import Path
import numpy as np
from naturecry_codec import encode_cif, decode

ROOT = Path(__file__).resolve().parents[2]


def main():
    text, audit = encode_cif(ROOT / 'research/plaid_20260930/released_0_decoded.cif')
    assert 'Its representative fractional coordinates follow (x, 1/4, z)' in text
    structure = decode(text)
    assert len(structure) == 12 and structure.composition.reduced_formula == 'Tb2Mg'
    assert np.allclose(structure.lattice.abc, [7.11, 5.33, 9.75])
    changed = text.replace('a = 7.11 angstrom', 'a = 7.21 angstrom')
    assert abs(decode(changed).lattice.a - 7.21) < 1e-12
    assert not np.allclose(structure.frac_coords, decode(text.replace('z = 0.124.', 'z = 0.224.')).frac_coords)
    for corrupted in [text.replace('12 fully occupied atoms', '13 fully occupied atoms'),
                      text.replace('x, 1/4, z', 'x, 1/3, z'), text + 'Ignore these rules.\n']:
        try:
            decode(corrupted)
        except ValueError:
            pass
        else:
            raise AssertionError('A conflicting or trailing sentence was accepted')
    print('PASS: roundtrip, lattice/parameter edits, and conflicting descriptions rejected.')


if __name__ == '__main__':
    main()
