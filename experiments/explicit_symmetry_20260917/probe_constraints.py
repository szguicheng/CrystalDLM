"""Probe explicit affine Wyckoff constraints and source decimal precision."""
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import spglib
import sympy as sy
from pymatgen.io.cif import CifParser
from scipy.optimize import linear_sum_assignment


def frac(x): return sy.Rational(Fraction(float(x)).limit_denominator(48))


def main():
    assert os.environ.get('SLURM_JOB_ID')
    files=sorted(Path('/scratch/chenggui2/CrystalDLM/cif_roundtrip_20260917/results').glob('*/input.cif'))
    rows=[]
    for path in files:
        c=CifParser(str(path),frac_tolerance=0).parse_structures(primitive=False)[0]
        q=c.frac_coords%1
        ds=spglib.get_symmetry_dataset((c.lattice.matrix,q,c.atomic_numbers),symprec=1e-3)
        trans=[[frac(v) for v in t] for t in ds.translations]
        translation_error=float(np.max(np.abs(np.array(trans,dtype=float)-ds.translations)))
        digits=next(d for d in range(13) if np.max(np.abs(q-np.round(q,d)))<1e-12)
        nodes=[]; positions=[]; species=[]
        for rep in sorted(set(ds.equivalent_atoms)):
            equations=[]
            for rotation,t in zip(ds.rotations,trans):
                delta=rotation@q[rep]+np.array(t,dtype=float)-q[rep]
                image=np.rint(delta)
                if np.linalg.norm((delta-image)@c.lattice.matrix)<1e-3:
                    for a,b in zip(rotation-np.eye(3,dtype=int),[int(n)-v for n,v in zip(image,t)]):
                        equations.append([*map(int,a),b])
            reduced,pivots=sy.Matrix(equations).rref()
            assert 3 not in pivots
            axes=sy.symbols('x y z')
            expressions=list(axes)
            for i,pivot in enumerate(pivots):
                expressions[pivot]=reduced[i,3]-sum(reduced[i,j]*axes[j] for j in range(3) if j!=pivot)
            subs={axes[j]:q[rep,j] for j in range(3)}
            base=np.array([float(e.subs(subs)) for e in expressions])
            orbit=[]
            for r,t in zip(ds.rotations,trans):
                point=(r@base+np.array(t,dtype=float))%1
                if not any(np.max(np.abs(point-p-np.rint(point-p)))<1e-7 for p in orbit): orbit.append(point)
            rounded=np.round(np.array(orbit),digits)%1
            positions.extend(rounded);species.extend([c.atomic_numbers[rep]]*len(rounded))
            nodes.append({'element':str(c[rep].specie),'letter_for_audit_only':ds.wyckoffs[rep],
                'source_representative':q[rep].tolist(),'expressions':[str(e) for e in expressions],
                'free_axes':[str(axes[j]) for j in range(3) if j not in pivots],
                'ideal_representative':base.tolist(),'multiplicity':len(orbit)})
        positions=np.array(positions)
        d=c.lattice.get_all_distances(q,positions)
        compatible=np.array(c.atomic_numbers)[:,None]==np.array(species)[None,:]
        source_ids,target_ids=linear_sum_assignment(np.where(compatible,d,1e6))
        delta=q[source_ids]-positions[target_ids];delta-=np.rint(delta)
        error=float(np.linalg.norm(delta@c.lattice.matrix,axis=1).max())
        row={'id':path.parent.name,'formula':c.composition.reduced_formula,'space_group':int(ds.number),'source_atoms':len(c),
            'expanded_atoms':len(positions),'digits':digits,'translation_rationalization_error_fractional':translation_error,
            'maximum_rounded_coordinate_error_A':error,'nodes':nodes}
        rows.append(row);print(json.dumps(row),flush=True)
    Path('constraint_probe.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__': main()
