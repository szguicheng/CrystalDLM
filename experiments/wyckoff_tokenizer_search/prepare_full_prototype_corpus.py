"""Build the larger train / frozen-development corpus using the verified codec."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import numpy as np
import torch
from audit_symmetry_graph import process
from discrete_graph_prototype import split
from prototype_conditions import compile_condition


def initialize():
    torch.set_num_threads(1)


def work(payload):
    row,partition=payload;mid=row['material_id']
    try:
        stats,encoded=process(row,4.5)
        prototype,geometry=split(encoded)
        condition=compile_condition(prototype,geometry['lattice'],71)
        target=np.asarray(geometry['free_vectors'])
        recovered=condition['q']+np.einsum('nij,nj->ni',condition['projection'],target)
        expected=np.array([n['fractional'] for n in encoded['header']['nodes']])
        assert np.allclose(recovered,expected,atol=1e-6)
        assert int(condition['multiplicity'].sum())==stats['atoms']
        if condition['internal_dof']<=0:
            return {'id':mid,'state':'no_internal_dof','space_group':stats['space_group']}
        path=Path('prototype_full_corpus')/partition/f'{mid}.pt'
        torch.save({'id':mid,'condition':condition,'target':target,'prototype':prototype,
                    'geometry':geometry,'audit':stats},path)
        return {'id':mid,'state':'ready','space_group':stats['space_group'],
                'path':str(path),'internal_dof':condition['internal_dof']}
    except Exception as error:
        return {'id':mid,'state':'failed','error_type':type(error).__name__,'error':str(error)}


def main():
    assert os.environ.get('SLURM_JOB_ID')
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','val'],required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    torch.set_num_threads(1);start=time.perf_counter()
    source=Path('input')/f'{args.split}.csv'
    with source.open() as stream:rows=list(csv.DictReader(stream))
    selected_ids=None
    if args.split=='val':
        previous=torch.load('multiplicity_candidates/data.pt',map_location='cpu')
        selected=sorted(previous['splits']['val'],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:2048]
        selected_ids=[r['id'] for r in selected]
        wanted=set(selected_ids);rows=[r for r in rows if r['material_id'] in wanted]
        assert len(rows)==2048
    rows=sorted(rows,key=lambda r:hashlib.sha256(r['material_id'].encode()).hexdigest())
    output=Path('prototype_full_corpus');(output/args.split).mkdir(parents=True,exist_ok=True)
    # Freeze selection before any geometry filtering; no validation scores used.
    (output/f'{args.split}_selection.json').write_text(json.dumps([r['material_id'] for r in rows]))
    records=[]
    with (output/f'{args.split}_progress.jsonl').open('w') as log, ProcessPoolExecutor(
            max_workers=args.workers,initializer=initialize,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(work,((row,args.split) for row in rows),chunksize=8)):
            records.append(result);log.write(json.dumps(result)+'\n')
            if (i+1)%500==0:
                log.flush()
                print(json.dumps({'split':args.split,'processed':i+1,
                      'ready':sum(r['state']=='ready' for r in records),
                      'failed':sum(r['state']=='failed' for r in records),
                      'seconds':time.perf_counter()-start}),flush=True)
    manifest={'split':args.split,'selected':len(rows),'ready':sum(r['state']=='ready' for r in records),
              'excluded_no_internal_dof':sum(r['state']=='no_internal_dof' for r in records),
              'failures':[r for r in records if r['state']=='failed'],'records':records,
              'fixed_prior_development_selection':selected_ids is not None,
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'code_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                  ['prepare_full_prototype_corpus.py','audit_symmetry_graph.py','symmetry_graph_codec.py',
                   'discrete_graph_prototype.py','prototype_conditions.py']},
              'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start}
    (output/f'{args.split}_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['records','code_sha256','failures']}),flush=True)


if __name__=='__main__':main()
