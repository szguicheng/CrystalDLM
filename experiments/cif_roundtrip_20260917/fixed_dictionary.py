"""Frozen crystallographic rule tables and a positional edge-orbit token stream.

Numeric scalars are structural tokens/continuous slots, not text BPE tokens.
No per-material geometry, chemistry or edge data is stored in the dictionary.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.io.cif import CifWriter

from cif_codec import encode_cif,decode_bundle
from discrete_graph_prototype import operations,chart


def canonical(value):
    return json.dumps(value,separators=(',',':'),sort_keys=True)


def group_rule(prototype):
    return {k:prototype[k] for k in ['space_group','hall_number','operations']}


def site_rule(node,group_id):
    return {'group':group_id,**{k:node[k] for k in ['wyckoff_letter','stabilizer','dof']}}


def fingerprint(dictionary):
    payload={k:v for k,v in dictionary.items() if k!='id'}
    return 'D'+hashlib.sha256(canonical(payload).encode()).hexdigest()


def validate(dictionary):
    if dictionary.get('id')!=fingerprint(dictionary):
        raise ValueError('Dictionary content/version fingerprint mismatch')


def build_dictionary(bundles):
    # Sorting makes IDs independent of material order. Updating a dictionary
    # creates a new version; encode/decode never insert rules automatically.
    groups=sorted({canonical(group_rule(b['prototype'])) for b in bundles})
    group_ids={key:i for i,key in enumerate(groups)}
    sites=sorted({canonical(site_rule(n,group_ids[canonical(group_rule(b['prototype']))]))
                  for b in bundles for n in b['prototype']['nodes']})
    dictionary={'format':'edge-orbit-rules-v1','groups':[json.loads(g) for g in groups],
                'sites':[json.loads(s) for s in sites]}
    dictionary['id']=fingerprint(dictionary)
    return dictionary


def free_chart(rule,group):
    """Choose independent components of u; recover all three using fixed rules."""
    r,t=operations(group)
    p,_=chart(rule,r,t)
    d=rule['dof']
    if d==0: return [],np.empty((3,0))
    columns=[]
    for k in range(3):
        if np.linalg.matrix_rank(p[:,columns+[k]],tol=1e-8)>len(columns): columns.append(k)
    c=p[:,columns]
    indices=[]
    for k in range(3):
        if np.linalg.matrix_rank(c[indices+[k],:],tol=1e-8)>len(indices): indices.append(k)
    assert len(indices)==d
    basis=c@np.linalg.inv(c[indices,:])
    basis[indices,:]=np.eye(d)
    return indices,basis


def pack(bundle,dictionary):
    validate(dictionary)
    prototype=bundle['prototype']; geometry=bundle['geometry']
    group_lookup={canonical(g):i for i,g in enumerate(dictionary['groups'])}
    site_lookup={canonical(s):i for i,s in enumerate(dictionary['sites'])}
    key=canonical(group_rule(prototype))
    if key not in group_lookup: raise ValueError('Global rule is not in the frozen dictionary')
    gid=group_lookup[key]
    tokens=['EO2',dictionary['id'],gid,len(prototype['nodes']),len(prototype['edge_tokens']),bundle['radius_A']]
    tokens.extend(np.asarray(geometry['lattice']).reshape(-1).tolist())
    for node,u in zip(prototype['nodes'],geometry['free_vectors']):
        rule=site_rule(node,gid); key=canonical(rule)
        if key not in site_lookup: raise ValueError('Site rule is not in the frozen dictionary')
        indices,basis=free_chart(rule,dictionary['groups'][gid])
        u=np.asarray(u); independent=u[indices]
        if not np.allclose(basis@independent,u,atol=1e-8,rtol=0):
            raise ValueError('Free vector is outside the dictionary site constraint')
        tokens.extend([node['Z'],site_lookup[key],*independent.tolist()])
    for edge in prototype['edge_tokens']: tokens.extend(edge)
    return list(map(str,tokens))


def unpack(tokens,dictionary):
    validate(dictionary)
    if len(tokens)<15 or tokens[0]!='EO2': raise ValueError('Invalid EO2 token stream')
    if tokens[1]!=dictionary['id']: raise ValueError('Token stream requires another dictionary version')
    gid,ns,ne=map(int,tokens[2:5]); radius=float(tokens[5])
    if not 0<=gid<len(dictionary['groups']) or ns<1 or ne<0: raise ValueError('Invalid stream counts or group ID')
    group=deepcopy(dictionary['groups'][gid]); nodes=[]; free=[]
    lattice=np.array(list(map(float,tokens[6:15]))).reshape(3,3).tolist()
    cursor=15
    for _ in range(ns):
        z,sid=map(int,tokens[cursor:cursor+2]); cursor+=2
        if not 0<=sid<len(dictionary['sites']): raise ValueError('Unknown site rule ID')
        rule=dictionary['sites'][sid]
        if rule['group']!=gid: raise ValueError('Site rule belongs to another global rule')
        d=rule['dof']; values=np.array(list(map(float,tokens[cursor:cursor+d]))); cursor+=d
        if len(values)!=d: raise ValueError('Missing continuous geometry')
        _,basis=free_chart(rule,group)
        free.append((basis@values).tolist())
        nodes.append({'Z':z,**{k:deepcopy(rule[k]) for k in ['wyckoff_letter','stabilizer','dof']}})
    if len(tokens)!=cursor+6*ne: raise ValueError('Token count does not match declared edge count')
    edges=[list(map(int,tokens[k:k+6])) for k in range(cursor,len(tokens),6)]
    if any(not (0<=e[0]<ns and 0<=e[1]<ns and 0<=e[2]<len(group['operations'])) for e in edges):
        raise ValueError('Edge references an unknown site or operation')
    return {'format':'edge-orbit-cif-v1','radius_A':radius,
            'prototype':{**group,'nodes':nodes,'edge_tokens':edges},'geometry':{'lattice':lattice,'free_vectors':free}}


def dumps(tokens): return ' '.join(tokens)+'\n'


def loads(text): return text.split()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build-dictionary')
    build.add_argument('bundles',nargs='+',help='Explicitly chosen existing encoded JSON files')
    build.add_argument('--output',required=True)
    for name in ['encode','decode']:
        p=sub.add_parser(name); p.add_argument('input'); p.add_argument('output')
        p.add_argument('--dictionary',required=True)
        if name=='encode':
            p.add_argument('--radius',type=float,default=4.5)
            p.add_argument('--symprec',type=float,default=1e-7)
    args=parser.parse_args()
    if args.command=='build-dictionary':
        bundles=[json.loads(Path(p).read_text()) for p in args.bundles]
        # Explicit build is separate from read-only encode/decode.
        Path(args.output).write_text(canonical(build_dictionary(bundles)))
        return
    dictionary=json.loads(Path(args.dictionary).read_text())
    if args.command=='encode':
        bundle=encode_cif(args.input,args.radius,args.symprec)
        Path(args.output).write_text(dumps(pack(bundle,dictionary)))
    else:
        bundle=unpack(loads(Path(args.input).read_text()),dictionary)
        crystal,_=decode_bundle(bundle)
        CifWriter(crystal,symprec=None,significant_figures=12).write_file(args.output)


if __name__=='__main__': main()
