"""Fixed-weight, matched-release-count test of random versus learned-confidence unmasking."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from adapt import fingerprint
from diagnostics.cohort_probe import select_rows
from model import Hybrid, unpack_block
from prepare import K, MAXLEN, STEPS, sha, tasks, write_json
from run import attempt, records
from score import audit


def release_counts(seed,size=K,steps=STEPS):
    rng=torch.Generator().manual_seed(seed+1000);remaining=size;counts=[]
    for k in range(steps,0,-1):
        number=int((torch.rand(remaining,generator=rng)<1/k).sum())
        counts.append(number);remaining-=number
    assert remaining==0 and sum(counts)==size
    return counts


def choose_positions(scores,number,policy,rng):
    if policy=='left':return torch.arange(number,device=scores.device)
    if policy=='confidence':return scores.topk(number).indices
    assert policy=='random'
    return torch.randperm(len(scores),device=scores.device,generator=rng)[:number]


@torch.inference_mode()
def sample(model,tok,prefix,seed,policy,counts):
    device=model.time.weight.device;context=tok.encode(prefix,add_special_tokens=False)
    assert context[-1]==model.start_id and len(context)+K<=MAXLEN
    x=torch.tensor(context,device=device);current=torch.full((K,),model.mask_id,device=device,dtype=torch.long)
    values_rng=torch.Generator(device=device).manual_seed(seed)
    order_rng=torch.Generator(device=device).manual_seed(seed+2000)
    banned=list(set(tok.all_special_ids)-{model.end_id,model.pad_id});trace=[]
    for k,number in zip(range(STEPS,0,-1),counts):
        positions=(current==model.mask_id).nonzero().flatten()
        if not len(positions):break
        logits=model.logits(torch.cat([x,current]),'hybrid',(len(x),len(x)+K),k,(positions+len(x)).tolist()).float()
        logits[:,banned]=-torch.inf
        values=torch.multinomial((logits/.8).softmax(-1),1,generator=values_rng).flatten()
        scores=logits.softmax(-1).gather(1,values[:,None]).flatten()
        selected=choose_positions(scores,number,policy,order_rng)
        current[positions[selected]]=values[selected]
        trace.append(dict(k=k,number=number,positions=positions[selected].tolist()))
    raw=current.tolist();decoded=tok.decode(raw,skip_special_tokens=False)
    try:
        text=tok.decode(unpack_block(raw,model.end_id,model.pad_id),skip_special_tokens=False)
        return dict(boundary_valid=True,crystal=text,raw=decoded,trace=trace)
    except ValueError as error:
        return dict(boundary_valid=False,error=str(error),raw=decoded,trace=trace)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--policies',nargs='+',choices=['random','confidence','left'],default=['random','confidence'])
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    manifest=json.loads((args.initial/'manifest.json').read_text());train=records(args.data,'train');val=records(args.data,'val')
    cohorts={'train':select_rows(train,sorted([r['id'] for r in train],key=lambda i:sha(i.encode()))),
        'old_val':select_rows(val,manifest['previously_diagnosed_validation_ids']),
        'fresh_val':select_rows(val,manifest['fresh_validation_ids'])}
    checkpoint=args.initial/'hybrid/joint/checkpoint.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['update']==512 and saved['config']['mode']=='hybrid'
    assert saved['config']['source_sha256']['model.py']==sha((ROOT/'model.py').read_bytes())
    tok=AutoTokenizer.from_pretrained(args.data/'tokenizer');model=Hybrid.pretrained(tok)
    model.load_state_dict(saved['model']);provenance=saved['config'];del saved;model=model.cuda().eval()
    initial=fingerprint(model)
    write_json(args.out/'manifest.json',dict(cohorts={s:[r['id'] for r in rows] for s,rows in cohorts.items()},
        policies=args.policies,temperature=.8,seeds=[17,29,43],fixed_counts=True,updates_applied=0,
        checkpoint_config=provenance,checkpoint_sha256=sha(checkpoint.read_bytes()),initial_fingerprint=initial,
        source_sha256=sha(Path(__file__).read_bytes()),slurm_job=os.environ['SLURM_JOB_ID'],slurm_step=os.environ.get('SLURM_STEP_ID'),
        auditor_source_sha256=sha((ROOT/'score.py').read_bytes()),
        reference='https://github.com/ML-GSAI/LLaDA/blob/main/generate.py',
        note='Confidence heuristic changes release order, not token values. Matched-count random is distributionally the original kernel but has different RNG mapping. No oracle length, no repair, no retries; test split unused.'))
    groups=defaultdict(lambda:dict(n=0,valid=0,boundary_valid=0))
    with (args.out/'raw.jsonl').open('w') as log:
        for split,rows in cohorts.items():
            for row in rows:
                ex=tasks(row,'val','plus')[0];lead=attempt(lambda:model.text(ex['prefix'],tok,'hybrid',trigger=True))
                for seed in [17,29,43]:
                    counts=release_counts(seed)
                    for policy in args.policies:
                        result=attempt(lambda:sample(model,tok,ex['prefix']+lead['value'],seed,policy,counts)) if lead['ok'] else lead
                        boundary=bool(result['ok'] and result['value']['boundary_valid'])
                        checked=audit(result['value']['crystal'],row['condition']) if boundary else dict(valid=False)
                        key=f'{split}/{policy}';groups[key]['n']+=1;groups[key]['valid']+=int(checked['valid']);groups[key]['boundary_valid']+=int(boundary)
                        log.write(json.dumps(dict(split=split,id=row['id'],policy=policy,seed=seed,counts=counts,lead=lead,result=result,audit=checked))+'\n');log.flush()
                print(json.dumps(dict(split=split,id=row['id'],counts=dict(groups))),flush=True)
    final=fingerprint(model);assert final==initial,'diagnosis mutated model weights'
    write_json(args.out/'summary.json',dict(status='complete',groups=dict(groups),initial_fingerprint=initial,final_fingerprint=final,
        updates_applied=0,policies=args.policies,note=f'48 materials, 3 repeated seeds, {len(args.policies)} policies; {48*3*len(args.policies)} attempts. No full-loop or final-test conclusion.'))


if __name__=='__main__':main()
