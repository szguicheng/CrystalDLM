"""Explicit affine coordinate programs and space-group generators.

Decoder uses arithmetic group closure, not Wyckoff/space-group dictionaries.
Rounded source coordinates and explicit residuals are separate from exact rules.
"""
import argparse
from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice,Structure
from pymatgen.io.cif import CifParser,CifWriter
from scipy.optimize import linear_sum_assignment
import spglib
import sympy as sy

AXES=sy.symbols('x y z')


def rational(x): return Fraction(float(x)).limit_denominator(48)


def linear_tokens(coefficients,constant):
    terms=[]
    for axis,c in zip(['x','y','z'],coefficients):
        c=Fraction(c)
        if c==1: terms.append([axis])
        elif c==-1: terms.append(['NEG',axis])
        elif c: terms.append(['MUL',str(c),axis])
    if constant: terms.append([str(Fraction(constant))])
    if not terms: return ['0']
    result=terms[0]
    for term in terms[1:]: result=['ADD',*result,*term]
    return result


def read_expression(tokens,cursor):
    token=tokens[cursor];cursor+=1
    if token=='FREE': return ('FREE',float(tokens[cursor])),cursor+1
    if token=='NEG':
        x,cursor=read_expression(tokens,cursor);return ('NEG',x),cursor
    if token in ['ADD','MUL']:
        x,cursor=read_expression(tokens,cursor);y,cursor=read_expression(tokens,cursor)
        return (token,x,y),cursor
    if token in ['x','y','z']: return token,cursor
    return Fraction(token),cursor


def evaluate(expr,variables):
    if isinstance(expr,Fraction): return expr
    if isinstance(expr,str): return variables[expr]
    if expr[0]=='FREE': return expr[1]
    if expr[0]=='NEG': return -evaluate(expr[1],variables)
    a,b=evaluate(expr[1],variables),evaluate(expr[2],variables)
    return a+b if expr[0]=='ADD' else a*b


def affine_coefficients(expr):
    if isinstance(expr,Fraction): return [Fraction(0)]*3+[expr]
    if isinstance(expr,str):
        out=[Fraction(0)]*4;out[['x','y','z'].index(expr)]=Fraction(1);return out
    if expr[0]=='FREE': raise ValueError('FREE is only allowed as a whole site-coordinate expression')
    a=affine_coefficients(expr[1])
    if expr[0]=='NEG': return [-v for v in a]
    b=affine_coefficients(expr[2])
    if expr[0]=='ADD': return [x+y for x,y in zip(a,b)]
    if not any(a[:3]): return [a[3]*v for v in b]
    if not any(b[:3]): return [b[3]*v for v in a]
    raise ValueError('Nonlinear coordinate product is not an affine symmetry rule')


def op_key(r,t,denominator):
    return (*map(int,np.asarray(r).reshape(-1)),*(int(v)%denominator for v in t))


def compose(a,b,denominator):
    ar=np.array(a[:9],dtype=int).reshape(3,3);br=np.array(b[:9],dtype=int).reshape(3,3)
    return op_key(ar@br,ar@np.array(b[9:],dtype=int)+np.array(a[9:],dtype=int),denominator)


def closure(generators,denominator):
    identity=op_key(np.eye(3,dtype=int),[0,0,0],denominator)
    known={identity};pending=[identity]
    while pending:
        h=pending.pop()
        for g in generators:
            new=compose(g,h,denominator)
            if new not in known:
                known.add(new);pending.append(new)
                if len(known)>768: raise ValueError('Generator closure is not a supported finite crystallographic group')
    return sorted(known)


def operation_expressions(key,denominator):
    return [linear_tokens(key[i*3:i*3+3],Fraction(key[9+i],denominator)) for i in range(3)]


def select_generators(keys,denominator):
    """Greedy closure growth with expression length tie-break; remove redundancies."""
    keys=sorted(keys);lookup={k:i for i,k in enumerate(keys)};n=len(keys)
    rs=np.array([k[:9] for k in keys]).reshape(n,3,3);ts=np.array([k[9:] for k in keys])
    rotations=rs[:,None]@rs[None,:]
    translations=(np.einsum('aij,bj->abi',rs,ts)+ts[:,None,:])%denominator
    table=np.array([[lookup[op_key(rotations[i,j],translations[i,j],denominator)] for j in range(n)] for i in range(n)])
    identity=lookup[op_key(np.eye(3,dtype=int),[0,0,0],denominator)]
    def subgroup(gens):
        seen={identity};todo=[identity]
        while todo:
            h=todo.pop()
            for g in gens:
                new=int(table[g,h])
                if new not in seen: seen.add(new);todo.append(new)
        return seen
    selected=[];current={identity}
    costs=[sum(len(e) for e in operation_expressions(k,denominator)) for k in keys]
    while len(current)<n:
        candidates=[(len(subgroup(selected+[i])),-costs[i],-i,i) for i in range(n) if i not in current]
        selected.append(max(candidates)[3]);current=subgroup(selected)
    for i in list(selected):
        trial=[j for j in selected if j!=i]
        if len(subgroup(trial))==n: selected=trial
    generators=[keys[i] for i in selected]
    assert closure(generators,denominator)==keys
    return generators


def expand_sites(sites,keys,denominator,digits,origin=None):
    origin=np.zeros(3) if origin is None else np.asarray(origin)
    positions=[];species=[]
    for element,expressions in sites:
        variables={axis:expr[1] for axis,expr in zip(['x','y','z'],expressions) if isinstance(expr,tuple) and expr[0]=='FREE'}
        q=np.array([float(evaluate(expr,variables)) for expr in expressions])
        orbit=[]
        for key in keys:
            r=np.array(key[:9]).reshape(3,3);t=np.array(key[9:])/denominator
            point=(r@q+t)%1
            if not any(np.max(np.abs(point-p-np.rint(point-p)))<1e-8 for p in orbit): orbit.append(point)
        rounded=np.round(np.array(orbit)+origin,digits)%1
        positions.extend(rounded);species.extend([element]*len(rounded))
    return np.array(positions),species


def decode(program):
    tokens=program.split();cursor=0
    def take(expected):
        nonlocal cursor
        if tokens[cursor]!=expected: raise ValueError(f'Expected {expected}, got {tokens[cursor]}')
        cursor+=1
    take('SP1');take('CELL')
    cell=list(map(float,tokens[cursor:cursor+6]));cursor+=6
    origin=np.zeros(3)
    if tokens[cursor]=='ORIGIN':
        cursor+=1;origin=np.array(list(map(float,tokens[cursor:cursor+3])));cursor+=3
    take('ROUND');digits=int(tokens[cursor]);cursor+=1
    take('GENERATORS');ng=int(tokens[cursor]);cursor+=1
    generators=[]
    for _ in range(ng):
        take('G');exprs=[]
        for _ in range(3):
            e,cursor=read_expression(tokens,cursor);exprs.append(e)
        coefficients_all=[affine_coefficients(e) for e in exprs]
        t=[a[3] for a in coefficients_all];r=[]
        for a in coefficients_all:
            coefficients=a[:3]
            if any(c.denominator!=1 for c in coefficients): raise ValueError('Generator rotation must be integer in lattice coordinates')
            r.append(list(map(int,coefficients)))
        if abs(round(np.linalg.det(r)))!=1: raise ValueError('Generator must preserve the integer lattice invertibly')
        generators.append((r,t))
    denominator=math.lcm(*[v.denominator for _,t in generators for v in t]) if generators else 1
    generators=[op_key(r,[int(v*denominator) for v in t],denominator) for r,t in generators]
    keys=closure(generators,denominator)
    take('SITES');ns=int(tokens[cursor]);cursor+=1
    sites=[]
    for _ in range(ns):
        element=tokens[cursor];cursor+=1;exprs=[]
        for _ in range(3):
            e,cursor=read_expression(tokens,cursor);exprs.append(e)
            if not (isinstance(e,tuple) and e[0]=='FREE'): affine_coefficients(e)
        sites.append((element,exprs))
    positions,species=expand_sites(sites,keys,denominator,digits,origin)
    take('RESIDUALS');nr=int(tokens[cursor]);cursor+=1
    for _ in range(nr):
        i=int(tokens[cursor]);delta=np.array(list(map(float,tokens[cursor+1:cursor+4])));cursor+=4
        positions[i]=(positions[i]+delta)%1
    take('END')
    if cursor!=len(tokens): raise ValueError('Trailing tokens')
    return Structure(Lattice.from_parameters(*cell),species,positions)


def encode_cif(path,symprec=1e-3):
    crystal=CifParser(str(path),frac_tolerance=0).parse_structures(primitive=False)[0]
    if not crystal.is_ordered: raise ValueError('Ordered fully occupied structures only')
    source_q=crystal.frac_coords%1;lattice=crystal.lattice.matrix
    ds=spglib.get_symmetry_dataset((lattice,source_q,crystal.atomic_numbers),symprec=symprec)
    raw_translations=ds.translations
    translations=[[rational(v)%1 for v in t] for t in raw_translations]
    delta=np.array(translations,dtype=float)-raw_translations;delta-=np.rint(delta)
    snap_error=float(np.max(np.abs(delta)))
    origin=np.zeros(3)
    if snap_error>1e-8:
        # spglib convention: q_standard = P q_input + p.
        # Retain the original basis and use q_input = q_local + origin.
        origin=-np.linalg.solve(ds.transformation_matrix,ds.origin_shift)
        origin-=np.rint(origin)
        raw_translations=ds.translations+np.einsum('nij,j->ni',ds.rotations,origin)-origin
        translations=[[rational(v)%1 for v in t] for t in raw_translations]
        delta=np.array(translations,dtype=float)-raw_translations;delta-=np.rint(delta)
        if np.max(np.abs(delta))>1e-8: raise ValueError('Origin-normalized translations are not supported small rationals')
    q=(source_q-origin)%1
    denominator=math.lcm(*[v.denominator for t in translations for v in t])
    keys=[op_key(r,[int(v*denominator) for v in t],denominator) for r,t in zip(ds.rotations,translations)]
    generators=select_generators(keys,denominator)
    digits=next(d for d in range(13) if np.max(np.abs(source_q-np.round(source_q,d)))<1e-12)
    lines=['SP1','CELL '+' '.join(map(str,crystal.lattice.parameters))]
    if np.any(origin): lines.append('ORIGIN '+' '.join(map(str,origin.tolist())))
    lines.extend([f'ROUND {digits}',f'GENERATORS {len(generators)}'])
    for key in generators: lines.append('G '+' '.join(token for expression in operation_expressions(key,denominator) for token in expression))
    representatives=sorted(set(ds.equivalent_atoms));lines.append(f'SITES {len(representatives)}')
    node_audit=[]
    for rep in representatives:
        equations=[]
        for r,t in zip(ds.rotations,translations):
            delta=r@q[rep]+np.array(t,dtype=float)-q[rep];image=np.rint(delta)
            if np.linalg.norm((delta-image)@lattice)<symprec:
                equations.extend([[*map(int,a),sy.Rational(int(n)-b)] for a,n,b in zip(r-np.eye(3,dtype=int),image,t)])
        reduced,pivots=sy.Matrix(equations).rref()
        if 3 in pivots: raise ValueError('Inconsistent site constraints')
        expressions=list(AXES)
        for row,pivot in enumerate(pivots):
            expressions[pivot]=reduced[row,3]-sum(reduced[row,j]*AXES[j] for j in range(3) if j!=pivot)
        tokens=[]
        for j,expr in enumerate(expressions):
            if j not in pivots: tokens.extend(['FREE',str(float(q[rep,j]))])
            else:
                coefficients=[Fraction(expr.coeff(axis)) for axis in AXES]
                constant=Fraction(expr.subs(dict.fromkeys(AXES,0)))
                tokens.extend(linear_tokens(coefficients,constant))
        lines.append(str(crystal[rep].specie)+' '+' '.join(tokens))
        node_audit.append({'element':str(crystal[rep].specie),'coordinate_rules':[str(e) for e in expressions],'dof':3-len(pivots)})
    uncorrected='\n'.join([*lines,'RESIDUALS 0','END'])+'\n'
    rebuilt=decode(uncorrected)
    if len(rebuilt)!=len(crystal): raise ValueError('Symbolic orbit cardinality differs from original atom count')
    distances=crystal.lattice.get_all_distances(source_q,rebuilt.frac_coords)
    same=np.array(crystal.atomic_numbers)[:,None]==np.array(rebuilt.atomic_numbers)[None,:]
    i,j=linear_sum_assignment(np.where(same,distances,1e6))
    if not np.all(same[i,j]): raise ValueError('Element-preserving mapping failed')
    residual=source_q[i]-rebuilt.frac_coords[j];residual-=np.rint(residual)
    lengths=np.linalg.norm(residual@lattice,axis=1)
    corrections=sorted((int(target),vector) for target,vector,length in zip(j,residual,lengths) if length>1e-10)
    lines.append(f'RESIDUALS {len(corrections)}')
    lines.extend(str(index)+' '+' '.join(map(str,vector.tolist())) for index,vector in corrections)
    lines.append('END')
    program='\n'.join(lines)+'\n'
    meta={'space_group':int(ds.number),'source_atoms':len(crystal),'site_records':len(representatives),
          'symmetry_operations':len(keys),'generator_count':len(generators),'source_decimal_places':digits,
          'residual_atoms':len(corrections),'residual_tokens':4*len(corrections),
          'uncorrected_maximum_error_A':float(lengths.max()),'translation_rationalization_error_fractional':snap_error,
          'origin_shift_fractional':origin.tolist(),'origin_tokens':4 if np.any(origin) else 0,
          'node_rules':node_audit,'global_generator_tokens':sum(1+sum(len(x) for x in operation_expressions(k,denominator)) for k in generators)}
    return program,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['encode','decode']);parser.add_argument('input');parser.add_argument('output')
    args=parser.parse_args()
    if args.command=='encode':
        program,_=encode_cif(args.input);Path(args.output).write_text(program)
    else:
        structure=decode(Path(args.input).read_text())
        CifWriter(structure,symprec=None,significant_figures=14).write_file(args.output)


if __name__=='__main__': main()
