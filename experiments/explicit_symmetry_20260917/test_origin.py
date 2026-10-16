"""Regression tests for translated origins and genuinely affine generators."""
import json
import os
from pathlib import Path
import numpy as np
from pymatgen.io.cif import CifWriter
import program_codec as codec
from test_program import parse,compare


def main():
    assert os.environ.get('SLURM_JOB_ID')
    out=Path('origin_tests');out.mkdir(exist_ok=True)
    results=[]
    try:
        p='SP1 CELL 4 4 4 90 90 90 ROUND 6 GENERATORS 1 G MUL x y y z SITES 1 C FREE 0.1 FREE 0.2 FREE 0.3 RESIDUALS 0 END'
        try: codec.decode(p)
        except ValueError: pass
        else: raise AssertionError('Nonlinear map was silently treated as an affine generator')
        results.append({'test':'reject_nonlinear_generator','pass':True})
    except Exception as e: results.append({'test':'reject_nonlinear_generator','pass':False,'error':repr(e)})
    for mid in ['mp-30009','mp-1095422','mp-4047']:
        try:
            path=Path('/scratch/chenggui2/CrystalDLM/cif_roundtrip_20260917/results')/mid/'input.cif'
            original=parse(path);_,reference=codec.encode_cif(path)
            shifted=original.copy();shifted.translate_sites(list(range(len(shifted))),[.137,.219,.087],frac_coords=True,to_unit_cell=True)
            file=out/(mid+'.cif');CifWriter(shifted,symprec=None,significant_figures=12).write_file(file)
            program,meta=codec.encode_cif(file)
            reconstructed=codec.decode(program)
            error=compare(parse(file),reconstructed)
            assert meta['space_group']==reference['space_group']
            (out/(mid+'.sym')).write_text(program)
            results.append({'test':mid+'_shifted','pass':True,'error_A':error,'space_group':meta['space_group']})
        except Exception as e: results.append({'test':mid+'_shifted','pass':False,'error':repr(e)})
    (out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
    assert all(r['pass'] for r in results)


if __name__=='__main__': main()
