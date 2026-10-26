"""Frozen 512-case corpus audit; no sample replacement and no lookup in decoder."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import traceback
from unittest.mock import patch
import warnings

import numpy as np
from pymatgen.io.cif import CifWriter

import program_codec as codec
from test_program import compare,parse


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start=time.perf_counter()
    source=Path('/scratch/chenggui2/CrystalDLM/tokenizer_search_20260908/input/train.csv')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest=='133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d'
    with source.open() as stream: rows=sorted(csv.DictReader(stream),key=lambda r:hashlib.sha256(r['material_id'].encode()).hexdigest())[:512]
    out=Path('corpus_512');out.mkdir(exist_ok=True)
    (out/'selection.json').write_text(json.dumps({'source_sha256':digest,'rule':'first 512 IDs in SHA256 order; no replacement','ids':[r['material_id'] for r in rows]},indent=2))
    records=[]
    for i,row in enumerate(rows):
        case=out/row['material_id'];case.mkdir(exist_ok=True)
        path=case/'input.cif';path.write_text(row['cif'])
        record={'id':row['material_id'],'formula':row['pretty_formula'],'dataset_space_group':int(float(row['spacegroup.number']))}
        try:
            with warnings.catch_warnings(record=True) as notices:
                program,meta=codec.encode_cif(path)
                file=case/'encoded.sym';file.write_text(program)
                with patch.object(codec.spglib,'get_symmetry_dataset',side_effect=AssertionError('Decoder attempted symmetry recognition')),patch.object(codec.spglib,'get_symmetry_from_database',side_effect=AssertionError('Decoder attempted group table lookup')):
                    reconstructed=codec.decode(file.read_text())
                original=parse(path);error=compare(original,reconstructed)
                decoded=case/'decoded.cif';CifWriter(reconstructed,symprec=None,significant_figures=14).write_file(decoded)
                reread_error=compare(original,parse(decoded))
            lexer=shlex.shlex(row['cif'],posix=True);lexer.whitespace_split=True;lexer.commenters='#'
            nt=len(program.split());nc=len(list(lexer))
            record.update(status='pass',cif_tokens=nc,program_tokens=nt,ratio=nt/nc,error_A=error,reread_error_A=reread_error,
                plain_structure_fields=6+4*len(original),warnings=[str(w.message) for w in notices],**meta)
        except Exception as error:
            record.update(status='fail',error_type=type(error).__name__,error=str(error))
            (case/'failure.txt').write_text(traceback.format_exc())
        records.append(record)
        (out/'records.json').write_text(json.dumps(records,indent=2))
        if (i+1)%32==0: print(json.dumps({'processed':i+1,'passed':sum(r['status']=='pass' for r in records),'seconds':time.perf_counter()-start}),flush=True)
    passed=[r for r in records if r['status']=='pass'];failures=[r for r in records if r['status']=='fail']
    summary={'selected':len(rows),'passed':len(passed),'failures':failures,'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,
      'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')},
      'decoder_spglib_calls_disabled':True,'symprec_A':1e-3,'coordinate_tolerance_A':1e-7,'dictionary_required':False}
    if passed:
        ratios=[r['ratio'] for r in passed]
        summary.update(total_cif_tokens=sum(r['cif_tokens'] for r in passed),total_program_tokens=sum(r['program_tokens'] for r in passed),
          ratio_median=float(np.median(ratios)),ratio_p95=float(np.quantile(ratios,.95)),ratio_max=max(ratios),
          shorter_or_equal=sum(r['ratio']<=1 for r in passed),within_1_25=sum(r['ratio']<=1.25 for r in passed),
          space_groups=sorted({r['space_group'] for r in passed}),residual_free=sum(r['residual_atoms']==0 for r in passed),
          total_residual_atoms=sum(r['residual_atoms'] for r in passed),max_error_A=max(r['reread_error_A'] for r in passed))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
