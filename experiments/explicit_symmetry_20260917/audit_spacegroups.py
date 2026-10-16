"""Generator/program closure for all 530 standard Hall settings, no decoder table."""
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice
import spglib

import program_codec as codec


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start=time.perf_counter();records=[]
    for hall in range(1,531):
        s=spglib.get_symmetry_from_database(hall)
        translations=[[codec.rational(v)%1 for v in t] for t in s['translations']]
        denominator=math.lcm(*[v.denominator for t in translations for v in t])
        keys=sorted(codec.op_key(r,[int(v*denominator) for v in t],denominator) for r,t in zip(s['rotations'],translations))
        gens=codec.select_generators(keys,denominator)
        assert codec.closure(gens,denominator)==keys
        r=np.array(s['rotations']);metric=np.mean(np.transpose(r,(0,2,1))@r,axis=0)
        lattice=Lattice(np.linalg.cholesky(metric)*5)
        q=np.array([.123457,.234569,.345679])
        expected=np.array([(np.array(k[:9]).reshape(3,3)@q+np.array(k[9:])/denominator)%1 for k in keys])
        expected=np.round(expected,8)%1
        lines=['SP1','CELL '+' '.join(map(str,lattice.parameters)),'ROUND 8',f'GENERATORS {len(gens)}']
        for key in gens: lines.append('G '+' '.join(t for expr in codec.operation_expressions(key,denominator) for t in expr))
        lines+=['SITES 1','C FREE .123457 FREE .234569 FREE .345679','RESIDUALS 0','END']
        program='\n'.join(lines)
        with patch.object(codec.spglib,'get_symmetry_dataset',side_effect=AssertionError('unexpected lookup')),patch.object(codec.spglib,'get_symmetry_from_database',side_effect=AssertionError('unexpected table')):
            decoded=codec.decode(program)
        assert len(decoded)==len(keys)
        delta=expected[:,None]-decoded.frac_coords[None,:];delta-=np.rint(delta)
        assert np.max(np.min(np.linalg.norm(delta,axis=-1),axis=1))<1e-10
        records.append({'hall_number':hall,'space_group':int(spglib.get_spacegroup_type(hall).number),'operations':len(keys),'generators':len(gens),'generator_tokens':sum(1+sum(len(x) for x in codec.operation_expressions(k,denominator)) for k in gens),'pass':True})
        if hall%50==0: print(json.dumps({'completed_hall_settings':hall,'seconds':time.perf_counter()-start}),flush=True)
    summary={'hall_settings':530,'space_groups':len({r['space_group'] for r in records}),'passed':len(records),'seconds':time.perf_counter()-start,
             'job_id':os.environ['SLURM_JOB_ID'],'scope':'Exact generator closure + prefix expression orbit decoding at a generic site; not all special Wyckoff families or arbitrary input settings.',
             'max_generators':max(r['generators'] for r in records),'max_generator_tokens':max(r['generator_tokens'] for r in records)}
    out=Path('hall_audit');out.mkdir(exist_ok=True)
    (out/'records.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
