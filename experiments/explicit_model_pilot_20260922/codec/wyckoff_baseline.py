"""Conventional SG + Wyckoff label + free-parameter reference codec.

Uses the fixed, public PyXtal table. Offers a canonical idealized standard-cell
stream and an original-frame stream with explicit precision/residual fields.
"""
import argparse
import ast
import csv
from functools import lru_cache
import hashlib
import itertools
from pathlib import Path

import numpy as np
from pymatgen.core import Element,Lattice,Structure
from pymatgen.io.cif import CifWriter
from scipy.optimize import linear_sum_assignment
import spglib

from audit_wyckoff_families import affine
from test_program import parse

FIXTURE=Path(__file__).resolve().parent/'fixtures/wyckoff_list.csv'
IMAGES=np.array(list(itertools.product([-1,0,1],repeat=3)))


def operation_key(rotations,translations):
    return tuple(sorted(tuple(np.rint(r).astype(int).reshape(-1))+tuple((np.rint(t*48).astype(int)%48)) for r,t in zip(rotations,translations)))


@lru_cache(None)
def reference_table():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest()=='ca56602252b7896121253c910739ef2fccf0ba947f2940358feb941f30c5e13d'
    halls={}
    for h in range(1,531):
        s=spglib.get_symmetry_from_database(h)
        halls.setdefault(operation_key(s['rotations'],s['translations']),h)
    with FIXTURE.open() as stream: groups=[(int(i),ast.literal_eval(raw)) for i,raw in list(csv.reader(stream))[2:]]
    table={};letters='abcdefghijklmnopqrstuvwxyzA'
    for sg,positions in groups:
        general=[affine(s) for s in positions[0]]
        rotations=np.array([np.array(a,dtype=float) for a,b in general])
        translations=np.array([np.array(b,dtype=float).reshape(3) for a,b in general])
        hall=halls[operation_key(rotations,translations)]
        definitions={}
        for index,position in enumerate(positions):
            a,b=affine(position[0]);axes=list(a.rref()[1])
            label=str(len(position))+letters[len(positions)-1-index]
            definitions[label]={'label':label,'multiplicity':len(position),'axes':axes,'template':position[0],
                'a':np.array(a,dtype=float),'b':np.array(b,dtype=float).reshape(3),
                'operations':[(np.array(affine(s)[0],dtype=float),np.array(affine(s)[1],dtype=float).reshape(3)) for s in position]}
        table[sg]={'hall':hall,'rotations':rotations,'translations':translations,'definitions':definitions}
    return table


def periodic_error(a,b):
    delta=np.asarray(a)[:,None,:]-np.asarray(b)[None,:,:];delta-=np.rint(delta)
    return float(np.max(np.min(np.max(np.abs(delta),axis=-1),axis=1)))


def fit_parameters(points,definition):
    matrix=definition['a'][:,definition['axes']]
    targets=(points[:,None,:]+IMAGES[None,:,:]-definition['b']).reshape(-1,3)
    values=targets@np.linalg.pinv(matrix).T if matrix.shape[1] else np.zeros((len(targets),0))
    valid=np.max(np.abs(values@matrix.T-targets),axis=1)<1e-7
    if matrix.shape[1]: valid&=np.all((values>=-1e-8)&(values<=1+1e-8),axis=1)
    for parameters in sorted(values[valid].tolist()):
        parameters=np.array(parameters)%1
        full=np.zeros(3);full[definition['axes']]=parameters
        generated=np.array([(a@full+b)%1 for a,b in definition['operations']])
        if periodic_error(points,generated)<1e-7 and periodic_error(generated,points)<1e-7:
            return parameters.tolist()
    return None


def site_records(ds,table):
    positions=ds.std_positions%1;species=ds.std_types;remaining=set(range(len(positions)));nodes=[]
    while remaining:
        rep=min(remaining);z=int(species[rep])
        images=(np.einsum('nij,j->ni',table['rotations'],positions[rep])+table['translations'])%1
        delta=images[:,None,:]-positions[None,:,:];delta-=np.rint(delta)
        members=np.flatnonzero(np.any(np.max(np.abs(delta),axis=-1)<1e-7,axis=0)&(species==z))
        assert rep in members
        points=positions[members]
        matched=None
        for label,definition in table['definitions'].items():
            if definition['multiplicity']!=len(members):continue
            parameters=fit_parameters(points,definition)
            if parameters is not None:
                matched={'Z':z,'element':Element.from_Z(z).symbol,'label':label,'parameters':parameters,'free_axes':['xyz'[a] for a in definition['axes']],
                         'template':definition['template']};break
        if matched is None:raise ValueError(f'No standard Wyckoff template matches Z={z}, orbit={len(members)}')
        nodes.append(matched);remaining.difference_update(members.tolist())
    return nodes


def expand_standard(sg,nodes):
    definitions=reference_table()[sg]['definitions'];positions=[];species=[]
    for z,label,parameters in nodes:
        definition=definitions[label];full=np.zeros(3);full[definition['axes']]=parameters
        for a,b in definition['operations']:
            positions.append((a@full+b)%1);species.append(z)
    return np.array(positions),species


def to_input_frame(positions,species,p,shift):
    # Enumerate standard-cell images intersecting the input cell. This also
    # covers input supercells, not just primitive-to-conventional conversion.
    low=np.floor(np.minimum(p,0).sum(1)+shift-1).astype(int)
    high=np.ceil(np.maximum(p,0).sum(1)+shift+1).astype(int)
    images=np.array(list(itertools.product(*[range(a,b+1) for a,b in zip(low,high)])))
    inverse=np.linalg.inv(p);out=[];types=[]
    for q,z in zip(positions,species):
        candidates=(q+images-shift)@inverse.T
        candidates=candidates[np.all((candidates>=-1e-9)&(candidates<=1+1e-9),axis=1)]
        for point in candidates:
            point=point%1;point[np.abs(point-1)<1e-12]=0
            if any(z==zz and np.max(np.abs(point-qq-np.rint(point-qq)))<1e-8 for qq,zz in zip(out,types)):continue
            out.append(point);types.append(z)
    return np.array(out),types


def decode(text):
    tokens=text.split();standard=tokens[0]=='WYCKOFF_STD1'
    if tokens[0] not in ['WYCKOFF1','WYCKOFF_STD1']:raise ValueError('Unknown baseline format')
    cursor=1
    def take(value):
        nonlocal cursor
        if tokens[cursor]!=value:raise ValueError(f'Expected {value}')
        cursor+=1
    take('SG');sg=int(tokens[cursor]);cursor+=1
    take('HALL');hall=int(tokens[cursor]);cursor+=1
    table=reference_table()[sg]
    if hall!=table['hall']:raise ValueError('Wyckoff table setting mismatch')
    take('CELL');cell=list(map(float,tokens[cursor:cursor+6]));cursor+=6
    p=np.eye(3);shift=np.zeros(3)
    if tokens[cursor]=='FRAME':
        cursor+=1;p=np.array(list(map(float,tokens[cursor:cursor+9]))).reshape(3,3);cursor+=9
        shift=np.array(list(map(float,tokens[cursor:cursor+3])));cursor+=3
    digits=None
    if not standard:take('ROUND');digits=int(tokens[cursor]);cursor+=1
    take('SITES');n=int(tokens[cursor]);cursor+=1;nodes=[]
    for _ in range(n):
        z=Element(tokens[cursor]).Z;label=tokens[cursor+1];cursor+=2
        d=len(table['definitions'][label]['axes'])
        parameters=list(map(float,tokens[cursor:cursor+d]));cursor+=d
        nodes.append((z,label,parameters))
    positions,species=expand_standard(sg,nodes)
    if not standard:
        positions,species=to_input_frame(positions,species,p,shift)
        positions=np.round(positions,digits)%1
        take('RESIDUALS');nr=int(tokens[cursor]);cursor+=1
        for _ in range(nr):
            index=int(tokens[cursor]);delta=np.array(list(map(float,tokens[cursor+1:cursor+4])));cursor+=4
            positions[index]=(positions[index]+delta)%1
    take('END')
    if cursor!=len(tokens):raise ValueError('Trailing tokens')
    return Structure(Lattice.from_parameters(*cell),species,positions)


def encode_cif(path,symprec=1e-3):
    source=parse(path);cell=(source.lattice.matrix,source.frac_coords,source.atomic_numbers)
    detected=spglib.get_symmetry_dataset(cell,symprec=symprec,angle_tolerance=-1)
    sg=int(detected.number);table=reference_table()[sg];hall=table['hall']
    ds=spglib.get_symmetry_dataset(cell,hall_number=hall,symprec=symprec,angle_tolerance=-1)
    if ds is None:raise ValueError('Unable to transform into the reference Wyckoff setting')
    nodes=site_records(ds,table)
    node_lines=[node['element']+' '+node['label']+(' '+' '.join(map(str,node['parameters'])) if node['parameters'] else '') for node in nodes]
    standard_cell=Lattice(ds.std_lattice).parameters
    standard='\n'.join(['WYCKOFF_STD1',f'SG {sg}',f'HALL {hall}','CELL '+' '.join(map(str,standard_cell)),f'SITES {len(nodes)}',*node_lines,'END'])+'\n'
    expected=Structure(Lattice.from_parameters(*standard_cell),ds.std_types,ds.std_positions)
    from test_program import compare
    standard_error=compare(expected,decode(standard))
    p=np.array(ds.transformation_matrix);shift=np.array(ds.origin_shift)
    lines=['WYCKOFF1',f'SG {sg}',f'HALL {hall}','CELL '+' '.join(map(str,source.lattice.parameters))]
    frame=not(np.allclose(p,np.eye(3),atol=1e-12,rtol=0) and np.allclose(shift,0,atol=1e-12,rtol=0))
    if frame:lines.append('FRAME '+' '.join(map(str,[*p.reshape(-1),*shift])))
    q=source.frac_coords%1
    digits=next(d for d in range(13) if np.max(np.abs(q-np.round(q,d)))<1e-12)
    lines.extend([f'ROUND {digits}',f'SITES {len(nodes)}',*node_lines])
    uncorrected=decode('\n'.join([*lines,'RESIDUALS 0','END']))
    if len(uncorrected)!=len(source):raise ValueError('Original-cell atom count not recovered')
    distance=source.lattice.get_all_distances(q,uncorrected.frac_coords)
    same=np.array(source.atomic_numbers)[:,None]==np.array(uncorrected.atomic_numbers)[None,:]
    i,j=linear_sum_assignment(np.where(same,distance,1e6));assert np.all(same[i,j])
    residual=q[i]-uncorrected.frac_coords[j];residual-=np.rint(residual)
    errors=np.linalg.norm(residual@source.lattice.matrix,axis=1)
    corrections=sorted((int(target),delta) for target,delta,error in zip(j,residual,errors) if error>1e-10)
    lines.append(f'RESIDUALS {len(corrections)}')
    lines.extend(str(index)+' '+' '.join(map(str,delta.tolist())) for index,delta in corrections);lines.append('END')
    full='\n'.join(lines)+'\n'
    audit={'space_group':sg,'hall_number':hall,'standard_cell_parameters':list(standard_cell),'nodes':nodes,
           'standard_tokens':len(standard.split()),'roundtrip_tokens':len(full.split()),'frame_tokens':13 if frame else 0,
           'residual_atoms':len(corrections),'residual_tokens':4*len(corrections),'round_digits':digits,
           'standard_reconstruction_error_A':standard_error,'uncorrected_input_error_A':float(errors.max()),
           'table_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}
    return full,standard,audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['encode','decode']);p.add_argument('input');p.add_argument('output');args=p.parse_args()
    if args.command=='encode':
        full,standard,_=encode_cif(args.input);Path(args.output).write_text(full)
        Path(args.output).with_suffix('.standard.tokens').write_text(standard)
    else:CifWriter(decode(Path(args.input).read_text()),symprec=None,significant_figures=14).write_file(args.output)


if __name__=='__main__':main()
