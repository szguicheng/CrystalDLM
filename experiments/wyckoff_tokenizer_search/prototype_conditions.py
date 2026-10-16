"""Compile graph-network conditions from integer prototype and specified lattice."""
import numpy as np
from discrete_graph_prototype import expand,operations,chart
from symmetry_graph_codec import action_tables,transform
from prepare_graph_backbone import pack
from benchmark_asu_control import add_asu
import torch


def compile_condition(prototype,lattice,seed):
    rotations,translations=operations(prototype)
    assert np.array_equal(rotations[0],np.eye(3)) and np.allclose(translations[0],0)
    rng=np.random.default_rng(seed)
    generic=rng.uniform(.05,.95,(len(prototype['nodes']),3))
    x,z,edges,cosets=expand(prototype,generic)
    ops=np.array([g for group in cosets for g in group])
    counts=np.array([len(group) for group in cosets])
    reps=np.cumsum(np.r_[0,counts[:-1]])
    assert all(group[0]==0 for group in cosets)
    oi=np.repeat(np.arange(len(cosets)),counts)
    a,b=rotations[ops],translations[ops]
    charts=[chart(node,rotations,translations) for node in prototype['nodes']]
    projection=np.array([p for p,bias in charts]);anchor=np.array([bias for p,bias in charts])
    perm,offsets=action_tables(x,z,rotations,translations)
    directed=[(i,j,*n) for i,j,*n in sorted(edges)]
    directed += [(j,i,*(-np.array(n))) for i,j,*n in sorted(edges)]
    coarse=[];remaining=set(edges)
    while remaining:
        e=min(remaining)
        orbit={transform(e,g,rotations,perm,offsets) for g in range(len(rotations))}
        assert orbit<=edges
        remaining.difference_update(orbit)
        arcs=[(i,j,*n) for i,j,*n in orbit]
        arcs += [(j,i,*(-np.array(n))) for i,j,*n in orbit]
        for source in sorted({int(oi[e[0]]),int(oi[e[1]])}):
            group=sorted(v for v in arcs if v[0]==reps[source])
            assert group
            targets={int(oi[v[1]]) for v in group};assert len(targets)==1
            _,j,*n=group[0]
            coarse.append({'src':source,'dst':targets.pop(),'A':a[j],'b':b[j]+n,
                           'sumA':sum(a[v[1]] for v in group),
                           'sumb':sum(b[v[1]]+v[2:] for v in group),'count':len(group)})
    global_projection=rotations.mean(0)
    dof=sum(node['dof'] for node in prototype['nodes'])-int(round(np.trace(global_projection)))
    return dict(lattice=np.asarray(lattice),q=anchor,projection=projection,
                Z=np.array([n['Z'] for n in prototype['nodes']]),orbit=oi,A=a,b=b,reps=reps,
                multiplicity=counts,edges=directed,coarse=coarse,global_projection=global_projection,
                internal_dof=dof)


def pack_condition(records):
    d=pack(records)
    for key in ['image','cb','sumb']:
        d[key]=d[key].reshape(-1,3)
    for key in ['cA','sumA']:
        d[key]=d[key].reshape(-1,3,3)
    d=add_asu(d)
    d['global_projection']=torch.as_tensor(np.array([r['global_projection'] for r in records]),dtype=torch.float64)
    return d
