"""Fresh validation CIFs, with original, translated and sheared-cell frames."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

import program_codec as codec
from test_program import parse,compare


def main():
    assert os.environ.get('SLURM_JOB_ID')
    source=Path('/scratch/chenggui2/CrystalDLM/tokenizer_search_20260908/input/val.csv')
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29'
    with source.open() as stream: rows=sorted(csv.DictReader(stream),key=lambda r:hashlib.sha256(r['material_id'].encode()).hexdigest())[:32]
    trained=set(json.loads(Path('corpus_512/selection.json').read_text())['ids'])
    assert not (trained & {r['material_id'] for r in rows})
    out=Path('unseen_frames');out.mkdir(exist_ok=True);records=[];start=time.perf_counter()
    m=np.array([[1,1,0],[0,1,0],[0,0,1]])
    for row in rows:
        case=out/row['material_id'];case.mkdir(exist_ok=True)
        original_path=case/'input.cif';original_path.write_text(row['cif'])
        c=parse(original_path)
        shifted=c.copy();shifted.translate_sites(list(range(len(c))),[.137,.219,.087],frac_coords=True,to_unit_cell=True)
        sheared=Structure(m@c.lattice.matrix,c.species,c.frac_coords@np.linalg.inv(m),to_unit_cell=True)
        for variant,structure in [('original',c),('translated',shifted),('sheared_basis',sheared)]:
            try:
                path=case/(variant+'.cif')
                if variant=='original': path.write_text(row['cif'])
                else: CifWriter(structure,symprec=None,significant_figures=12).write_file(path)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    program,meta=codec.encode_cif(path)
                    decoded=codec.decode(program)
                    error=compare(parse(path),decoded)
                (case/(variant+'.sym')).write_text(program)
                records.append({'id':row['material_id'],'variant':variant,'pass':True,'error_A':error,'tokens':len(program.split()),'space_group':meta['space_group']})
            except Exception as e:
                records.append({'id':row['material_id'],'variant':variant,'pass':False,'error':repr(e)})
                (case/(variant+'.failure.txt')).write_text(traceback.format_exc())
    summary={'fresh_materials':32,'variants':len(records),'passed':sum(r['pass'] for r in records),'failures':[r for r in records if not r['pass']],
             'core_sha256':hashlib.sha256(Path('program_codec.py').read_bytes()).hexdigest(),'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,
             'scope':'Information retention under 32 unseen materials x 3 frames; not identical-token canonicalization across frames.'}
    (out/'records.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
