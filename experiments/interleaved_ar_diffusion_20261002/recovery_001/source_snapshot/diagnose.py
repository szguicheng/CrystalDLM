"""Post-run diagnostic only: partial-mask recovery versus full-mask predictions."""
import argparse
import json
import os
from pathlib import Path
import time
import warnings

import torch
from transformers import AutoTokenizer
from model import Hybrid, unpack_block
from prepare import END, K, ROOT, STEPS, sha, tasks, write_json
from score import audit


def accuracy(predicted, target, selected):
    count = int(selected.sum())
    correct = int(((predicted == target) & selected).sum())
    return dict(correct=correct, count=count, accuracy=correct/count if count else None)


def training_model_hash(provenance):
    key = 'code_hashes' if 'code_hashes' in provenance else 'source_sha256'
    return provenance[key]['model.py']


def main():
    assert accuracy(torch.tensor([1,2,3]),torch.tensor([1,4,3]),torch.tensor([True,True,False])) == dict(correct=1,count=2,accuracy=.5)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'), 'Use an allocated compute GPU'
    assert not args.out.exists(), 'Never overwrite diagnostic evidence'
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(args.data/'tokenizer')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=Hybrid.pretrained(tok)
    model.load_state_dict(saved['model'])
    provenance=saved['config'];del saved
    model=model.cuda().eval()
    rows=[json.loads(s) for s in (args.data/'train.jsonl').read_text().splitlines()]
    rng=torch.Generator(device='cuda').manual_seed(31415)
    results=[];started=time.monotonic()
    for row in rows:
        ex=tasks(row,'train','plus')[0]
        prefix=tok.encode(ex['prefix']+ex['lead'],add_special_tokens=False)
        content=tok.encode(ex['crystal'],add_special_tokens=False)+[model.end_id]
        target=torch.tensor(content+[model.pad_id]*(K-len(content)),device='cuda')
        for k in [16,32]:
            hidden=torch.rand(K,device='cuda',generator=rng)<k/STEPS
            corrupted=torch.where(hidden,model.mask_id,target)
            ids=torch.cat([torch.tensor(prefix,device='cuda'),corrupted])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model.logits(ids,'hybrid',(len(prefix),len(prefix)+K),k,list(range(len(prefix),len(prefix)+K))).float()
            banned=list(set(tok.all_special_ids)-{model.end_id,model.pad_id})
            logits=logits.clone()
            logits[:,banned]=-torch.inf
            predictions=logits.argmax(-1)
            completed=torch.where(hidden,predictions,target).tolist()
            record=dict(id=row['id'],k=k,masked_content=accuracy(predictions,target,hidden&(target!=model.pad_id)&(target!=model.end_id)),
                        masked_padding=accuracy(predictions,target,hidden&(target==model.pad_id)),
                        predicted_end_count=completed.count(model.end_id),
                        raw_completion=tok.decode(completed,skip_special_tokens=False))
            try:
                text=tok.decode(unpack_block(completed,model.end_id,model.pad_id),skip_special_tokens=False)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');record['audit']=audit(text,row['condition'])
                record['condition_valid']=record['audit']['valid']
            except ValueError as error:
                record.update(condition_valid=False,error=str(error))
            results.append(record)
    summary={}
    for k in [16,32]:
        subset=[r for r in results if r['k']==k]
        correct=sum(r['masked_content']['correct'] for r in subset)
        count=sum(r['masked_content']['count'] for r in subset)
        summary[str(k)]=dict(content_top1_accuracy=correct/count,content_correct=correct,content_count=count,
                            condition_valid=sum(r['condition_valid'] for r in subset),n=len(subset))
    write_json(args.out,dict(status='complete',checkpoint_config=provenance,checkpoint_model_code_hash=training_model_hash(provenance),
             diagnostic_code_sha256=sha(Path(__file__).read_bytes()),summary=summary,records=results,
             elapsed_seconds=time.monotonic()-started,slurm_job=os.environ['SLURM_JOB_ID'],
             note='Teacher reference lead is supplied; k16 includes visible gold tokens. Diagnostic does not replace any official sampling result.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
