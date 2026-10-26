"""No-update diagnosis: material novelty, generation wording, MASK level and temperature."""
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
from model import Hybrid, unpack_block
from prepare import K, STEPS, sha, tasks, write_json
from run import attempt, records
from score import audit


def select_rows(rows,ids):
    indexed={r['id']:r for r in rows}
    return [indexed[i] for i in ids[:16]]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.cohort/'manifest.json').read_text())
    training=records(args.data,'train');validation=records(args.data,'val')
    train_ids=sorted([r['id'] for r in training],key=lambda i:sha(i.encode()))[:16]
    cohorts={'train':select_rows(training,train_ids),'val':select_rows(validation,manifest['validation_ids'])}
    assert not {r['id'] for r in cohorts['train']} & {r['id'] for r in cohorts['val']}
    checkpoint=args.cohort/'hybrid/checkpoint.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved['update']==2048 and saved['config']['mode']=='hybrid'
    assert saved['config']['source_sha256']['model.py']==sha((ROOT/'model.py').read_bytes())
    tok=AutoTokenizer.from_pretrained(args.data/'tokenizer');model=Hybrid.pretrained(tok)
    provenance=model.restore_checkpoint(saved,expected_mode='hybrid');del saved
    model=model.cuda().eval();groups=defaultdict(lambda:dict(n=0,valid=0))
    write_json(args.out/'manifest.json',dict(cohorts={s:[r['id'] for r in rows] for s,rows in cohorts.items()},
        temperatures=[.8,.4,.1],seeds=[17,29,43],wordings=['train','val'],mask_levels=[16,32],
        checkpoint_config=provenance,source_sha256=sha(Path(__file__).read_bytes()),
        checkpoint_sha256=sha(checkpoint.read_bytes()),updates_applied=0,
        slurm_job=os.environ['SLURM_JOB_ID'],slurm_step=os.environ.get('SLURM_STEP_ID'),gpu=torch.cuda.get_device_name(),
        note='Frozen first 16 materials per cohort, no retries or answer repair. Partial-mask reference recovery is diagnostic only. Test split unused.'))
    with (args.out/'raw.jsonl').open('w') as log:
        for split,rows in cohorts.items():
            for row in rows:
                for wording in ['train','val']:
                    ex=tasks(row,wording,'plus')[0]
                    for k in [16,32]:
                        packed=model.pack(dict(ex,noise_step=k),tok,'hybrid',torch.Generator().manual_seed(31415))
                        g=packed['groups'][-1];ids=packed['ids'].cuda();start=len(ids)-K
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            logits=model.logits(ids,'hybrid',(start-model.prediction_offset,start+K),k,g['positions']).float()
                        logits=logits.clone()
                        logits[:,list(set(tok.all_special_ids)-{model.end_id,model.pad_id})]=-torch.inf
                        predictions=logits.argmax(-1);target=torch.tensor(g['labels'],device='cuda')
                        content=(target!=model.pad_id)&(target!=model.end_id)
                        completed=ids[-K:].clone();completed[torch.tensor(g['positions'],device='cuda')-start]=predictions
                        result=attempt(lambda:audit(tok.decode(unpack_block(completed.tolist(),model.end_id,model.pad_id),skip_special_tokens=False),row['condition']))
                        item=dict(kind='recovery',split=split,wording=wording,id=row['id'],k=k,
                            correct=int(((predictions==target)&content).sum()),count=int(content.sum()),result=result)
                        log.write(json.dumps(item)+'\n');log.flush()
                    lead=attempt(lambda:model.text(ex['prefix'],tok,'hybrid',trigger=True))
                    for temperature in [.8,.4,.1]:
                        for seed in [17,29,43]:
                            rng=torch.Generator(device='cuda').manual_seed(seed)
                            result=attempt(lambda:model.crystal(ex['prefix']+lead['value'],tok,'hybrid',rng,temperature)) if lead['ok'] else lead
                            checked=audit(result['value'][0],row['condition']) if result['ok'] else dict(valid=False)
                            item=dict(kind='sample',split=split,wording=wording,id=row['id'],temperature=temperature,seed=seed,lead=lead,result=result,audit=checked)
                            key=f'{split}/{wording}/{temperature}';groups[key]['n']+=1;groups[key]['valid']+=int(checked['valid'])
                            log.write(json.dumps(item)+'\n');log.flush()
                print(json.dumps(dict(split=split,id=row['id'],counts=dict(groups))),flush=True)
    write_json(args.out/'summary.json',dict(status='complete',sample_counts=dict(groups),updates_applied=0))


if __name__=='__main__':main()
