"""Verify graph prototype without retaining source free coordinates or lattice."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from symmetry_graph_codec import decode,edge_key,match_sites
from discrete_graph_prototype import split,expand,representatives,integer_tree


def main():
    assert os.environ.get('SLURM_JOB_ID')
    start=time.perf_counter()
    source=Path('symmetry_graph_audit/train_512')
    records=json.loads((source/'records.json').read_text())
    output=Path('discrete_graph_prototypes');output.mkdir(exist_ok=True)
    results=[];failures=[]
    for index,record in enumerate(records):
        mid=record['material_id']
        try:
            encoded=json.loads((source/f'{mid}.json').read_text())
            prototype,geometry=split(encoded)
            prototype=json.loads(json.dumps(prototype))
            assert integer_tree(prototype)
            assert 'lattice' not in prototype and 'free_vectors' not in prototype
            x,z,edges,cosets=expand(prototype,geometry['free_vectors'])
            reference_x,reference_z,reference_edges=decode(encoded)
            assert len(x)==len(reference_x)
            mapping,images=match_sites(x,z,reference_x,reference_z)
            mapped={edge_key(mapping[i],mapping[j],images[j]+n-images[i]) for i,j,*n in edges}
            assert mapped==reference_edges,'original graph mismatch after gauge alignment'
            error=float(np.max(np.abs(x-reference_x[mapping]-images)))
            assert error<1e-6
            q0=representatives(prototype,geometry['free_vectors'])
            seed=int(hashlib.sha256(mid.encode()).hexdigest()[:8],16)
            rng=np.random.default_rng(seed)
            changes=[]
            for attempt in range(3):
                # Independent values: source free coordinates are not passed.
                u=rng.uniform(.05,.95,size=(len(prototype['nodes']),3))
                other_x,other_z,other_edges,other_cosets=expand(prototype,u)
                assert other_cosets==cosets,'generic embedding changes orbit cosets'
                assert np.array_equal(other_z,z)
                assert other_edges==edges,'prototype topology depends on source free coordinates'
                change=float(np.max(np.abs(representatives(prototype,u)-q0)))
                if any(n['dof'] for n in prototype['nodes']):assert change>1e-5
                changes.append(change)
            (output/f'{mid}.prototype.json').write_text(json.dumps(prototype,separators=(',',':')))
            (output/f'{mid}.geometry.json').write_text(json.dumps(geometry,separators=(',',':')))
            results.append({'id':mid,'space_group':prototype['space_group'],'nodes':len(prototype['nodes']),
                            'dof_sum':sum(n['dof'] for n in prototype['nodes']),
                            'expanded_atoms':len(x),'expanded_edges':len(edges),
                            'source_coordinate_error_fractional':error,'generic_coordinate_changes':changes})
        except Exception as error:
            failures.append({'id':mid,'type':type(error).__name__,'error':str(error)})
        if (index+1)%64==0:
            print(json.dumps({'processed':index+1,'passed':len(results),'failed':len(failures),
                              'seconds':time.perf_counter()-start}),flush=True)
    report={'selected_train_structures':len(records),'success':len(results),'failures':failures,
            'generic_embeddings_per_prototype':3,'integer_only_prototypes':True,
            'prototype_contains_lattice_or_free_coordinates':False,
            'structures_with_free_coordinates':sum(r['dof_sum']>0 for r in results),
            'coordinate_changes_do_not_preserve_original_radius_graph_or_physical_validity':True,
            'model_generation_not_tested':True,'job_id':os.environ['SLURM_JOB_ID'],
            'run_id':os.environ.get('CRYSTALDLM_RUN_ID'),'seconds':time.perf_counter()-start,
            'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
                             for f in ['discrete_graph_prototype.py','audit_discrete_prototype.py']}}
    (output/'records.json').write_text(json.dumps(results,indent=2))
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
