"""One frozen-model evaluation on the previously reserved cohort."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from compared_graph_network import ComparedGraphNetwork
from prototype_conditions import pack_condition
from benchmark_graph_backbone import convert
from train_prototype_denoising import corrupt,remove_origin,per_graph


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    corpus=Path('prototype_full_corpus');root=Path('prototype_full_model')
    manifest=json.loads((corpus/'replication_manifest.json').read_text())
    expected=json.loads(Path('input/manifest.json').read_text())
    assert manifest['source_csv_sha256']==expected['files']['val']['sha256']
    assert manifest['selected']==manifest['ready']+manifest['excluded_no_internal_dof']+len(manifest['failures'])
    audit=json.loads((corpus/'replication_failure_audit.json').read_text())
    assert all(d['reproduced_failure'] for d in audit['replication']['details'])
    assert hashlib.sha256((root/'final.pt').read_bytes()).hexdigest()==manifest['model_sha256']
    assert hashlib.sha256((root/'config.json').read_bytes()).hexdigest()==manifest['model_config_sha256']
    config=json.loads((root/'config.json').read_text())
    for name,sha in config['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    selected=[r for r in manifest['records'] if r['state']=='ready']
    assert not({r['id'] for r in selected}&set(config['ids']['train']))
    assert not({r['id'] for r in selected}&set(config['ids']['val']))
    model=ComparedGraphNetwork().cuda();model.load_state_dict(torch.load(root/'final.pt',map_location='cuda'));model.eval()
    rng=torch.Generator(device='cuda').manual_seed(manifest['noise_seed'])
    rows=[];parity=0.
    for offset in range(0,len(selected),32):
        objects=[torch.load(r['path'],map_location='cpu') for r in selected[offset:offset+32]]
        batch={'condition':convert(pack_condition([r['condition'] for r in objects]),torch.float32),
               'target':torch.as_tensor(np.concatenate([r['target'] for r in objects]),dtype=torch.float32,device='cuda')}
        d=batch['condition'];errors=[];base=[]
        for _ in range(manifest['noise_repeats']):
            noisy,e=corrupt(batch,rng)
            predicted=remove_origin(model(d,noisy,'orbit')[1],d)
            control=remove_origin(model(d,noisy,'asu')[1],d)
            parity=max(parity,float((predicted-control).abs().max()))
            assert torch.allclose(predicted,control,atol=3e-5,rtol=3e-4)
            errors.append(per_graph(predicted+e,d));base.append(per_graph(e,d))
        errors=torch.stack(errors).mean(0).cpu().tolist();base=torch.stack(base).mean(0).cpu().tolist()
        rows.extend({'id':o['id'],'mse':x,'identity_mse':y} for o,x,y in zip(objects,errors,base))
        if offset%1024==0:print(json.dumps({'evaluated':len(rows)}),flush=True)
    diff=np.array([r['mse']-r['identity_mse'] for r in rows]);boot=np.random.default_rng(2026)
    ci=np.quantile([diff[boot.integers(len(rows),size=len(rows))].mean() for _ in range(2000)],[.025,.975]).tolist()
    mse=float(np.mean([r['mse'] for r in rows]));identity=float(np.mean([r['identity_mse'] for r in rows]))
    report={'status':'COMPLETED','selected_candidates':manifest['selected'],'evaluated_structures':len(rows),
            'mean_mse':mse,'mean_identity_mse':identity,'relative_reduction':1-mse/identity,
            'difference_95ci':ci,'structures_improved':sum(r['mse']<r['identity_mse'] for r in rows),
            'same_weight_asu_max_difference':parity,'model_sha256':manifest['model_sha256'],
            'noise_seed':manifest['noise_seed'],'noise_repeats':manifest['noise_repeats'],
            'no_refitting':True,'not_full_diffusion_sampling':True,'job_id':os.environ['SLURM_JOB_ID']}
    (root/'replication_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (root/'replication_result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
