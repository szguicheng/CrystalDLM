"""Integer-only graph prototype with affine site-stabilizer constraints.

Free coordinates and lattice are separate geometry. The periodic graph decoder
uses unwrapped, consistently selected space-group coset representatives.
"""
from fractions import Fraction
import numpy as np
from symmetry_graph_codec import action_tables, edge_key, match_sites, transform


def rational(x):
    f = Fraction(float(x)).limit_denominator(48)
    assert abs(float(f) - x) < 1e-8, 'operation translation is not a small rational'
    return [f.numerator, f.denominator]


def operations(prototype):
    r = np.asarray([o['rotation'] for o in prototype['operations']], dtype=int)
    t = np.asarray([[a/b for a,b in o['translation']] for o in prototype['operations']])
    return r,t


def chart(node, rotations, translations):
    ids = [s['operation'] for s in node['stabilizer']]
    images = np.asarray([s['image'] for s in node['stabilizer']],dtype=int)
    p = rotations[ids].mean(0)
    b = (translations[ids]-images).mean(0)
    assert np.allclose(p@p,p,atol=1e-10)
    assert np.allclose(p@b,0.,atol=1e-10)
    return p,b


def integer_tree(value):
    if isinstance(value,dict): return all(isinstance(k,str) and integer_tree(v) for k,v in value.items())
    if isinstance(value,list): return all(integer_tree(v) for v in value)
    return type(value) in (int,str)


def split(encoded):
    header=encoded['header']
    rotations=np.asarray(header['rotations'],dtype=int)
    translations=np.asarray(header['translations'])
    prototype={'space_group':header['space_group'],'hall_number':header['hall_number'],
               'operations':[{'rotation':r.tolist(),'translation':[rational(x) for x in t]}
                             for r,t in zip(rotations,translations)],
               'nodes':[],'edge_tokens':encoded['edge_tokens']}
    free=[]
    for source in header['nodes']:
        q=np.asarray(source['fractional'])
        delta=np.einsum('nij,j->ni',rotations,q)+translations-q
        images=np.rint(delta).astype(int)
        ids=np.flatnonzero(np.max(np.abs(delta-images),axis=1)<1e-7)
        node={'Z':source['Z'],'wyckoff_letter':source['wyckoff_letter'],
              'stabilizer':[{'operation':int(i),'image':images[i].tolist()} for i in ids]}
        p,b=chart(node,rotations,translations)
        assert np.allclose(p@q+b,q,atol=1e-8)
        u=q-b
        assert np.allclose(p@u,u,atol=1e-8)
        node['dof']=int(round(np.trace(p)))
        assert node['dof']==np.linalg.matrix_rank(p,tol=1e-8)
        prototype['nodes'].append(node)
        free.append(u.tolist())
    assert integer_tree(prototype), 'non-discrete value in prototype'
    return prototype,{'lattice':header['lattice'],'free_vectors':free}


def representatives(prototype,free_vectors):
    r,t=operations(prototype)
    return np.array([b+p@u for node,u in zip(prototype['nodes'],free_vectors)
                     for p,b in [chart(node,r,t)]])


def expand(prototype,free_vectors):
    r,t=operations(prototype)
    q=representatives(prototype,free_vectors)
    positions,species,cosets=[],[],[]
    for node,point in zip(prototype['nodes'],q):
        begin=len(positions)
        chosen=[]
        for g,(rotation,translation) in enumerate(zip(r,t)):
            x=rotation@point+translation
            if any(np.max(np.abs(x-p-np.rint(x-p)))<1e-6 for p in positions[begin:]):
                continue
            positions.append(x);species.append(node['Z']);chosen.append(g)
        cosets.append(chosen)
    positions=np.asarray(positions);species=np.asarray(species)
    permutations,offsets=action_tables(positions,species,r,t)
    asu_ids,asu_shifts=match_sites(q,[n['Z'] for n in prototype['nodes']],positions,species)
    edges=set()
    for i,j,g,*image in prototype['edge_tokens']:
        target=r[g]@q[j]+t[g]+image
        target_id,target_shift=match_sites([target],[prototype['nodes'][j]['Z']],positions,species)
        edge=edge_key(asu_ids[i],target_id[0],target_shift[0]-asu_shifts[i])
        edges.update(transform(edge,h,r,permutations,offsets) for h in range(len(r)))
    return positions,species,edges,cosets
