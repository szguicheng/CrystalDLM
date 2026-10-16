"""Independent symbolic-family fixtures for all 1731 default-setting Wyckoff rows."""
import ast
import csv
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from unittest.mock import patch

import numpy as np
from pymatgen.core import Lattice
import sympy as sy

import program_codec as codec


@lru_cache(None)
def affine(text):
    assert re.fullmatch(r'[xyz0-9+*/().,\s-]+',text)
    expressions=[sy.sympify(re.sub(r'(?<=\d)(?=[xyz])','*',v),locals=dict(zip(['x','y','z'],codec.AXES))) for v in text.split(',')]
    a=sy.Matrix([[e.coeff(axis) for axis in codec.AXES] for e in expressions])
    b=sy.Matrix([e.subs(dict.fromkeys(codec.AXES,0)) for e in expressions])
    return a,b


def main():
    assert os.environ.get('SLURM_JOB_ID')
    source=Path('fixtures/wyckoff_list.csv');start=time.perf_counter()
    records=[];rng=np.random.default_rng(1709)
    with source.open() as stream:
        groups=[(int(i),ast.literal_eval(raw)) for i,raw in list(csv.reader(stream))[2:]]
    for sg,positions in groups:
        general=[affine(s) for s in positions[0]]
        denominator=math.lcm(*[Fraction(v).denominator for _,b in general for v in b])
        keys=sorted(codec.op_key(np.array(a,dtype=int),[int(v*denominator) for v in b],denominator) for a,b in general)
        gens=codec.select_generators(keys,denominator)
        rotations=np.array([np.array(a,dtype=int) for a,b in general])
        metric=np.mean(np.transpose(rotations,(0,2,1))@rotations,axis=0)
        lattice=Lattice(np.linalg.cholesky(metric)*5)
        header=['SP1','CELL '+' '.join(map(str,lattice.parameters)),'ROUND 10',f'GENERATORS {len(gens)}']
        header+=['G '+' '.join(t for e in codec.operation_expressions(g,denominator) for t in e) for g in gens]
        for row,position in enumerate(positions):
            a,b=affine(position[0]);null=a.T.nullspace()
            equations=sy.Matrix([[*v,(v.T*b)[0]] for v in null]) if null else sy.zeros(0,4)
            reduced,pivots=equations.rref();expressions=list(codec.AXES)
            for k,pivot in enumerate(pivots):
                expressions[pivot]=reduced[k,3]-sum(reduced[k,j]*codec.AXES[j] for j in range(3) if j!=pivot)
            assert 3-len(pivots)==a.rank()
            errors=[]
            for _ in range(3):
                parameters=rng.uniform(.11,.89,3)
                base=np.array(a,dtype=float)@parameters+np.array(b,dtype=float).reshape(3)
                fields=[]
                for j,e in enumerate(expressions):
                    if j not in pivots: fields+=['FREE',str(float(base[j]))]
                    else: fields+=codec.linear_tokens([Fraction(e.coeff(axis)) for axis in codec.AXES],Fraction(e.subs(dict.fromkeys(codec.AXES,0))))
                program='\n'.join([*header,'SITES 1','Si '+' '.join(fields),'RESIDUALS 0','END'])
                with patch.object(codec.spglib,'get_symmetry_dataset',side_effect=AssertionError('unexpected lookup')),patch.object(codec.spglib,'get_symmetry_from_database',side_effect=AssertionError('unexpected table')):
                    decoded=codec.decode(program)
                expected=np.array([(np.array(affine(s)[0],dtype=float)@parameters+np.array(affine(s)[1],dtype=float).reshape(3))%1 for s in position])
                expected=np.round(expected,10)%1
                assert len(decoded)==len(position),(sg,row,len(decoded),len(position))
                delta=expected[:,None]-decoded.frac_coords[None,:];delta-=np.rint(delta)
                error=float(np.max(np.min(np.linalg.norm(delta,axis=-1),axis=1)))
                assert error<2e-9,(sg,row,error)
                errors.append(error)
            records.append({'space_group':sg,'row':row,'multiplicity':len(position),'dof':int(a.rank()),'expressions':[str(e) for e in expressions],
                            'independent_parameter_draws':3,'max_fractional_error':max(errors),'program_tokens':len(program.split()),'pass':True})
        if sg%25==0: print(json.dumps({'groups':sg,'families':len(records),'seconds':time.perf_counter()-start}),flush=True)
    assert len(records)==1731
    out=Path('wyckoff_audit');out.mkdir(exist_ok=True)
    summary={'space_groups':230,'wyckoff_families':len(records),'instances':3*len(records),'passed':len(records),
             'max_fractional_error':max(r['max_fractional_error'] for r in records),'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,
             'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Official PyXtal default-setting affine families; explicit expressions to group orbit; fixture generation uses tables, decoder does not.'}
    (out/'records.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
