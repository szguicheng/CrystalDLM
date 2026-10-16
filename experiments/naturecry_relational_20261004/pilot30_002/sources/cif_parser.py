"""Real CIF roundtrips and semantic interventions for explicit symmetry programs."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from pymatgen.io.cif import CifParser
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent


def parse(path): return CifParser(str(path),frac_tolerance=0).parse_structures(primitive=False)[0]


def compare(source,target):
    assert len(source)==len(target),'atom count'
    assert source.composition==target.composition,'element counts'
    assert np.max(np.abs(source.lattice.matrix-target.lattice.matrix))<1e-8,'lattice'
    d=source.lattice.get_all_distances(source.frac_coords,target.frac_coords)
    same=np.array(source.atomic_numbers)[:,None]==np.array(target.atomic_numbers)[None,:]
    i,j=linear_sum_assignment(np.where(same,d,1e6))
    delta=source.frac_coords[i]-target.frac_coords[j];delta-=np.rint(delta)
    error=float(np.linalg.norm(delta@source.lattice.matrix,axis=1).max())
    assert error<1e-7,f'coordinates {error} A'
    return error


def main():
    assert os.environ.get('SLURM_JOB_ID')
    assert (ROOT/'program_codec.py').exists(),'Explicit program encoder/decoder not implemented'
    import program_codec as codec
    fixture=Path('/scratch/chenggui2/CrystalDLM/cif_roundtrip_20260917/results')
    out=ROOT/'pilot';out.mkdir(exist_ok=True)
    records=[]
    for path in sorted(fixture.glob('*/input.cif')):
        case=out/path.parent.name;case.mkdir(exist_ok=True)
        program,meta=codec.encode_cif(path)
        assert not any(t.startswith('RULE_') or t.startswith('WP_') for t in program.split())
        assert 'edge_tokens' not in program
        encoded=case/'encoded.sym';encoded.write_text(program)
        decoded=case/'decoded.cif'
        r=subprocess.run([sys.executable,str(ROOT/'program_codec.py'),'decode',str(encoded),str(decoded)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
        original=parse(path);reconstructed=parse(decoded)
        error=compare(original,reconstructed)
        repeat,meta2=codec.encode_cif(decoded)
        compare(original,codec.decode(repeat))
        lexer=shlex.shlex(path.read_text(),posix=True);lexer.whitespace_split=True;lexer.commenters='#'
        records.append({'id':path.parent.name,'formula':original.composition.reduced_formula,
                        'cif_tokens':len(list(lexer)),'program_tokens':len(program.split()),'error_A':error,**meta})
        (case/'audit.json').write_text(json.dumps(records[-1],indent=2))
        print(json.dumps(records[-1]),flush=True)
    (out/'records.json').write_text(json.dumps(records,indent=2))
    assert len(records)==10
    # Meaning-changing interventions, not merely syntactic parse checks.
    p='SP1 CELL 4 4 4 90 90 90 ROUND 6 GENERATORS 0 SITES 1 C 0 0 FREE 0.2 RESIDUALS 0 END'
    a=codec.decode(p);b=codec.decode(p.replace('FREE 0.2','FREE 0.3'))
    assert np.allclose(a.frac_coords[0],[0,0,.2])
    assert np.allclose(b.frac_coords[0],[0,0,.3])
    q='SP1 CELL 4 4 4 90 90 90 ROUND 6 GENERATORS 1 G NEG x NEG y NEG z SITES 1 C FREE 0.1 FREE 0.2 FREE 0.3 RESIDUALS 0 END'
    assert len(codec.decode(q))==2
    print('10 real-CIF roundtrips and semantic interventions passed',flush=True)


if __name__=='__main__': main()
