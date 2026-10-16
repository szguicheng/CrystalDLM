"""Re-identify both actual CIF files under identical symmetry settings."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import warnings

import spglib
from test_program import parse

TOLERANCES=[1e-7,1e-5,1e-3,1e-2,1e-1]


def identify(crystal,tolerance):
    ds=spglib.get_symmetry_dataset((crystal.lattice.matrix,crystal.frac_coords,crystal.atomic_numbers),symprec=tolerance,angle_tolerance=-1)
    if ds is None: raise ValueError('Symmetry identification failed')
    return {'number':int(ds.number),'international':ds.international,'hall_number':int(ds.hall_number),'operations':len(ds.rotations)}


def declared(path):
    m=re.search(r'''(?m)^\s*_(?:symmetry_Int_Tables_number|space_group_IT_number)\s+['"]?(\d+)''',path.read_text())
    return int(m.group(1)) if m else None


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start=time.perf_counter();root=Path('corpus_512')
    records=json.loads((root/'records.json').read_text());results=[]
    for i,record in enumerate(records):
        case=root/record['id'];a=case/'input.cif';b=case/'decoded.cif'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');original=parse(a);decoded=parse(b)
        comparisons=[]
        for tolerance in TOLERANCES:
            source=identify(original,tolerance);target=identify(decoded,tolerance)
            comparisons.append({'symprec_A':tolerance,'source':source,'decoded':target,'same_number':source['number']==target['number'],
                                'same_hall':source['hall_number']==target['hall_number'],'same_operation_count':source['operations']==target['operations']})
        result={'id':record['id'],'formula':record['formula'],'dataset_space_group':record['dataset_space_group'],
                'source_declared_number':declared(a),'decoded_declared_number':declared(b),'angle_tolerance':-1,
                'source_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'decoded_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),
                'comparisons':comparisons}
        (case/'space_group_comparison.json').write_text(json.dumps(result,indent=2));results.append(result)
        if (i+1)%64==0: print(json.dumps({'processed':i+1,'seconds':time.perf_counter()-start}),flush=True)
    summary={'materials':len(results),'spglib_version':spglib.__version__,'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,
             'angle_tolerance':-1,'tolerances':[],'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for k,tol in enumerate(TOLERANCES):
        summary['tolerances'].append({'symprec_A':tol,'same_number':sum(r['comparisons'][k]['same_number'] for r in results),
            'same_hall':sum(r['comparisons'][k]['same_hall'] for r in results),
            'same_operation_count':sum(r['comparisons'][k]['same_operation_count'] for r in results),
            'mismatches':[{'id':r['id'],**r['comparisons'][k]} for r in results if not r['comparisons'][k]['same_number']]})
    out=Path('wyckoff_comparison_20260919');out.mkdir(exist_ok=True)
    (out/'space_group_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'space_group_records.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
