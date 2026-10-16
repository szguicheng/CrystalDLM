"""Compile source-coordinate-independent conditions; freeze internal pilot split."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from prototype_conditions import compile_condition,pack_condition


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2)
    start=time.perf_counter();root=Path('discrete_graph_prototypes')
    rows=json.loads((root/'records.json').read_text())
    usable=[];excluded=[]
    for i,row in enumerate(rows):
        mid=row['id']
        prototype=json.loads((root/f'{mid}.prototype.json').read_text())
        geometry=json.loads((root/f'{mid}.geometry.json').read_text())
        first=compile_condition(prototype,geometry['lattice'],71)
        second=compile_condition(prototype,geometry['lattice'],903)
        x,y=pack_condition([first]),pack_condition([second])
        assert all(torch.allclose(x[k],y[k],rtol=0,atol=1e-10) for k in x),mid
        if first['internal_dof']<=0:
            excluded.append(mid)
        else:
            usable.append({'id':mid,'condition':first,'target':np.array(geometry['free_vectors'])})
        if (i+1)%64==0:
            print(json.dumps({'processed':i+1,'usable':len(usable),'seconds':time.perf_counter()-start}),flush=True)
    cut=int(len(usable)*.8)
    def batches(records):
        return [{'condition':pack_condition([r['condition'] for r in records[i:i+32]]),
                 'target':torch.as_tensor(np.concatenate([r['target'] for r in records[i:i+32]]),dtype=torch.float64),
                 'ids':[r['id'] for r in records[i:i+32]]} for i in range(0,len(records),32)]
    output=Path('prototype_denoising');output.mkdir(exist_ok=True)
    torch.save({'train':batches(usable[:cut]),'val':batches(usable[cut:])},output/'data.pt')
    report={'train_ids':[r['id'] for r in usable[:cut]],'validation_ids':[r['id'] for r in usable[cut:]],
            'excluded_no_internal_dof':excluded,'two_generic_condition_caches_equal':len(rows),
            'conditions':'Integer graph prototype plus explicitly specified lattice; no target free coordinates',
            'split_scope':'Internal pilot split of earlier original-training-set subset; not official test/replication',
            'steps':5000,'sigma_A':.1,'seed':17,'batch_size':32,'final_checkpoint_only':True,
            'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,
            'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                             ['prototype_conditions.py','prepare_prototype_denoising.py']}}
    (output/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':'COMPLETED','train':cut,'val':len(usable)-cut,
                      'excluded':len(excluded),'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':main()
