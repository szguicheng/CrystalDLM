"""Single-noise conditional denoising pilot from discrete prototype conditions."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from compared_graph_network import ComparedGraphNetwork
from benchmark_graph_backbone import convert


def remove_origin(v,d):
    g=d['asu_graph'];m=d['multiplicity'];ng=len(d['lattice'])
    norm=torch.zeros(ng,device=v.device).index_add(0,g,m)
    mean=torch.zeros(ng,3,device=v.device).index_add(0,g,v*m[:,None])/norm[:,None]
    l=d['lattice'].transpose(-1,-2)
    pg=l@d['global_projection']@torch.linalg.inv(l)
    shift=torch.einsum('nij,nj->ni',pg,mean)
    return v-shift[g]


def corrupt(batch,generator=None):
    d=batch['condition'];g=d['asu_graph'];inv=torch.linalg.inv(d['lattice'])
    e=torch.randn(batch['target'].shape,device='cuda',generator=generator)*.1
    fractional=torch.einsum('ni,nij->nj',e,inv[g])
    projected=torch.einsum('nij,nj->ni',d['projection'],fractional)
    e=remove_origin(torch.einsum('ni,nij->nj',projected,d['lattice'][g]),d)
    return batch['target']+torch.einsum('ni,nij->nj',e,inv[g]),e


def per_graph(error,d):
    g=d['asu_graph'];m=d['multiplicity'];ng=len(d['lattice'])
    dof=d['projection'].diagonal(dim1=-2,dim2=-1).sum(-1)
    denominator=torch.zeros(ng,device=error.device).index_add(0,g,m*dof)
    assert (denominator>0).all()
    return torch.zeros(ng,device=error.device).index_add(0,g,m*error.square().sum(-1))/denominator


@torch.no_grad()
def evaluate(model,batches):
    generator=torch.Generator(device='cuda').manual_seed(9217)
    rows=[];max_parity=0.
    model.eval()
    for batch in batches:
        d=batch['condition'];values=[];base=[]
        for _ in range(8):
            noisy,e=corrupt(batch,generator)
            predicted=remove_origin(model(d,noisy,'orbit')[1],d)
            control=remove_origin(model(d,noisy,'asu')[1],d)
            max_parity=max(max_parity,float((predicted-control).abs().max()))
            assert torch.allclose(predicted,control,atol=3e-5,rtol=3e-4)
            values.append(per_graph(predicted+e,d));base.append(per_graph(e,d))
        values=torch.stack(values).mean(0).cpu().tolist();base=torch.stack(base).mean(0).cpu().tolist()
        rows.extend({'id':mid,'mse':v,'identity_mse':b} for mid,v,b in zip(batch['ids'],values,base))
    return {'mean_mse':float(np.mean([r['mse'] for r in rows])),
            'mean_identity_mse':float(np.mean([r['identity_mse'] for r in rows])),
            'asu_orbit_max_vector_difference':max_parity},rows


def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2);torch.manual_seed(17)
    root=Path('prototype_denoising')
    data=torch.load(root/'data.pt',map_location='cpu')
    for batches in data.values():
        for batch in batches:
            batch['condition']=convert(batch['condition'],torch.float32)
            batch['target']=batch['target'].cuda().float()
    model=ComparedGraphNetwork().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    rng=np.random.default_rng(17)
    batch_probability=np.array([len(b["ids"]) for b in data["train"]],dtype=float)
    batch_probability/=batch_probability.sum()
    history=[];start=time.perf_counter()
    # Fixed observed input: changing the target payload cannot change a prediction.
    batch=data['train'][0];observed,_=corrupt(batch)
    with torch.no_grad():
        fixed_input=observed.clone()
        before=model(batch['condition'],observed,'orbit')[1]
        repeated=model(batch['condition'],observed,'orbit')[1]
        original=batch['target'];batch['target']=torch.randn_like(original)
        after=model(batch['condition'],observed,'orbit')[1]
        batch['target']=original
        assert torch.equal(observed,fixed_input)
        preflight={'same_input_repeat_max_abs':float((before-repeated).abs().max()),
                   'changed_label_max_abs':float((before-after).abs().max()),
                   'observed_input_unchanged':True,'atol':3e-5,'rtol':3e-4}
        print(json.dumps({'preflight':preflight}),flush=True)
        assert torch.allclose(before,repeated,atol=3e-5,rtol=3e-4)
        assert torch.allclose(before,after,atol=3e-5,rtol=3e-4)
    torch.nn.init.zeros_(model.vector_weight[-1].weight)
    torch.nn.init.zeros_(model.vector_weight[-1].bias)
    for step in range(1,5001):
        model.train();batch=data['train'][int(rng.choice(len(data['train']),p=batch_probability))];d=batch['condition']
        noisy,e=corrupt(batch)
        predicted=remove_origin(model(d,noisy,'orbit')[1],d)
        loss=per_graph(predicted+e,d).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%1000==0:
            metrics,rows=evaluate(model,data['val'])
            history.append({'step':step,**metrics,'seconds':time.perf_counter()-start})
            (root/'history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(history[-1]),flush=True)
    torch.save(model.state_dict(),root/'final.pt')
    (root/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    difference=np.array([r['mse']-r['identity_mse'] for r in rows])
    bootstrap=np.random.default_rng(2026)
    ci=np.quantile([difference[bootstrap.integers(len(rows),size=len(rows))].mean() for _ in range(2000)],[.025,.975]).tolist()
    report={'status':'COMPLETED','job_id':os.environ['SLURM_JOB_ID'],'steps':5000,'metrics':metrics,
            'relative_mse_reduction':1-metrics['mean_mse']/metrics['mean_identity_mse'],
            'paired_structure_difference_95ci':ci,'validation_structures':len(rows),
            'train_structures':sum(len(b['ids']) for b in data['train']),
            'target_payload_not_model_condition':True,'preflight':preflight,'single_noise_level_only':True,
            'graph_and_lattice_are_explicit_conditions':True,'not_full_diffusion_sampling':True,
            'seconds':time.perf_counter()-start,'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
             for f in ['prototype_conditions.py','prepare_prototype_denoising.py','train_prototype_denoising.py','compared_graph_network.py']}}
    (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
