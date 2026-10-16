"""Read-only final-model diagnostics; no extra training or checkpoint selection."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from compared_graph_network import ComparedGraphNetwork
from benchmark_graph_backbone import convert
from train_prototype_denoising import evaluate,remove_origin,per_graph,corrupt


@torch.no_grad()
def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    torch.set_num_threads(2)
    root=Path('prototype_denoising');raw=torch.load(root/'data.pt',map_location='cpu')
    model=ComparedGraphNetwork().cuda();model.load_state_dict(torch.load(root/'final.pt',map_location='cuda'));model.eval()
    result={}
    for split,batches in raw.items():
        for batch in batches:
            batch['condition']=convert(batch['condition'],torch.float32);batch['target']=batch['target'].cuda().float()
        metrics,rows=evaluate(model,batches)
        clean=[];coordinate_error=0.
        for batch in batches:
            d=batch['condition'];target=batch['target']
            clean.append(per_graph(remove_origin(model(d,target,'orbit')[1],d),d).cpu().numpy())
            noisy,e=corrupt(batch)
            actual=torch.einsum('nij,nj->ni',d['projection'],noisy-target)
            actual=torch.einsum('ni,nij->nj',actual,d['lattice'][d['asu_graph']])
            coordinate_error=max(coordinate_error,float((actual-e).abs().max()))
        assert coordinate_error<1e-5
        error=np.array([r['mse'] for r in rows]);base=np.array([r['identity_mse'] for r in rows])
        order=np.argsort(error-base)[::-1]
        result[split]={**metrics,'clean_input_prediction_mse':float(np.concatenate(clean).mean()),
                       'actual_input_noise_matches_target_max_error':coordinate_error,
                       'structures_improved':int((error<base).sum()),'structures':len(rows),
                       'median_relative_error':float(np.median(error/base)),
                       'worst_10':[rows[int(i)] for i in order[:10]]}
    result['diagnostic_only_no_optimization']=True;result['job_id']=os.environ['SLURM_JOB_ID']
    (root/'diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
