"""Frozen larger-corpus trial; same model/noise, final 20k checkpoint only."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from compared_graph_network import ComparedGraphNetwork
from prototype_conditions import pack_condition
from benchmark_graph_backbone import convert
from train_prototype_denoising import corrupt,remove_origin,per_graph,evaluate


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(17)
    corpus=Path('prototype_full_corpus');out=Path('prototype_full_model');out.mkdir(exist_ok=True)
    audit=json.loads((corpus/'failure_audit.json').read_text())
    assert all(d['reproduced_failure'] for p in ['train','val'] for d in audit[p]['details'])
    manifests={p:json.loads((corpus/f'{p}_manifest.json').read_text()) for p in ['train','val']}
    expected=json.loads(Path('input/manifest.json').read_text())
    ids={};data={};start=time.perf_counter()
    for partition,manifest in manifests.items():
        assert manifest['source_sha256']==expected['files'][partition]['sha256']
        assert manifest['selected']==manifest['ready']+manifest['excluded_no_internal_dof']+len(manifest['failures'])
        rows=[r for r in manifest['records'] if r['state']=='ready']
        ids[partition]=[r['id'] for r in rows];assert len(ids[partition])==len(set(ids[partition]))
        batches=[]
        for i in range(0,len(rows),32):
            objects=[torch.load(r['path'],map_location='cpu') for r in rows[i:i+32]]
            batches.append({'condition':convert(pack_condition([x['condition'] for x in objects]),torch.float32),
                            'target':torch.as_tensor(np.concatenate([x['target'] for x in objects]),dtype=torch.float32,device='cuda'),
                            'ids':[x['id'] for x in objects]})
            if i%4096==0:print(json.dumps({'loading':partition,'loaded':i+len(objects),'seconds':time.perf_counter()-start}),flush=True)
        data[partition]=batches
    assert not(set(ids['train'])&set(ids['val']))
    assert manifests['val']['selected']==2048 and manifests['val']['fixed_prior_development_selection']
    model=ComparedGraphNetwork().cuda()
    batch=data['train'][0];observed,_=corrupt(batch)
    with torch.no_grad():
        a=model(batch['condition'],observed,'orbit')[1]
        label=batch['target'];batch['target']=torch.randn_like(label)
        b=model(batch['condition'],observed,'orbit')[1];batch['target']=label
        assert torch.allclose(a,b,atol=3e-5,rtol=3e-4)
    torch.nn.init.zeros_(model.vector_weight[-1].weight);torch.nn.init.zeros_(model.vector_weight[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    rng=np.random.default_rng(17);prob=np.array([len(b['ids']) for b in data['train']],dtype=float);prob/=prob.sum()
    source_names=['train_full_prototype_denoising.py','train_prototype_denoising.py',
                  'prototype_conditions.py','compared_graph_network.py','orbit_graph_network.py',
                  'prepare_graph_backbone.py','benchmark_asu_control.py','benchmark_graph_backbone.py',
                  'FULL_PROTOTYPE_DENOISING_PROTOCOL.md']
    config={'steps':20000,'seed':17,'sigma_A':.1,'final_checkpoint_only':True,'ids':ids,
            'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_names},
            'corpus_manifest_sha256':{p:hashlib.sha256((corpus/f'{p}_manifest.json').read_bytes()).hexdigest() for p in manifests},
            'job_id':os.environ['SLURM_JOB_ID'],'loaded_seconds':time.perf_counter()-start,
            'parameter_count':sum(p.numel() for p in model.parameters())}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    began=time.perf_counter();history=[]
    for step in range(1,20001):
        model.train();batch=data['train'][int(rng.choice(len(data['train']),p=prob))];d=batch['condition']
        noisy,e=corrupt(batch);pred=remove_origin(model(d,noisy,'orbit')[1],d)
        loss=per_graph(pred+e,d).mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%2000==0:
            metrics,rows=evaluate(model,data['val'])
            history.append({'step':step,'training_batch_loss':float(loss.detach()),**metrics,'seconds':time.perf_counter()-began})
            (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(history[-1]),flush=True)
    torch.save(model.state_dict(),out/'final.pt')
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    delta=np.array([r['mse']-r['identity_mse'] for r in rows]);bootstrap=np.random.default_rng(2026)
    ci=np.quantile([delta[bootstrap.integers(len(delta),size=len(delta))].mean() for _ in range(2000)],[.025,.975]).tolist()
    result={'status':'COMPLETED','metrics':metrics,'relative_mse_reduction':1-metrics['mean_mse']/metrics['mean_identity_mse'],
            'paired_structure_difference_95ci':ci,'train_structures':len(ids['train']),'validation_structures':len(ids['val']),
            'training_seconds':time.perf_counter()-began,'job_id':os.environ['SLURM_JOB_ID'],
            'steps':20000,'single_seed_development_only':True,'not_full_diffusion_sampling':True}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
