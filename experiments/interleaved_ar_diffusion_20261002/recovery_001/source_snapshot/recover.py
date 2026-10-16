"""Recover generation with more G exposure and condition-order augmentation, then joint training."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil

import torch
from transformers import AutoTokenizer
from adapt import train_phase
from cohort import balanced_stream, unseen_reading
from model import Hybrid
from prepare import ROOT, TEMPLATES, encoded_length, sha, tasks, write_json
from rescore import score_evaluation
from run import evaluate, records

WARM_UPDATES=2048
JOINT_UPDATES=512
GEN_PROMPTS=[TEMPLATES['train'][0][0],TEMPLATES['train'][1][0],
    'Use space group {sg}. The required unit cell composition is {formula}. Generate the crystal.',
    'A unit cell of composition {formula} is required. Generate it with space group number {sg}.',
    'Generate a structure meeting these conditions: space group {sg}; composition {formula} in the unit cell.',
    'Construct the requested crystal. Its cell composition must be {formula}, and its space group number must be {sg}.']


def generation_example(row,style):
    return dict(tasks(row,'train','plus')[0],prefix='User: '+GEN_PROMPTS[style].format(**row['condition'])+'\nAssistant:',
                source_id=row['id'],generation_style=style)


def generation_stream(rows,seed):
    rng=random.Random(seed)
    while True:
        order=list(range(len(rows)));rng.shuffle(order)
        for i in order:
            yield dict(generation_example(rows[i],rng.randrange(len(GEN_PROMPTS))),supervise_lead=False)


def recovery_stream(rows,seed):
    sequence=balanced_stream(rows,seed);rng=random.Random(seed+3000)
    by_text={r['c1']:r for r in rows}
    while True:
        group=[next(sequence) for _ in range(8)];assert group[0]['task']=='G'
        row=by_text[group[0]['crystal']];g=generation_example(row,rng.randrange(len(GEN_PROMPTS)))
        head=g['prefix'].split('\nAssistant:',1)[0]
        for e in group:
            tail=e['prefix'].split('\nAssistant:',1)[1]
            yield dict(e,prefix=head+'\nAssistant:'+tail,source_id=row['id'],generation_style=g['generation_style'])


def freeze_exposure(rows,tok,out):
    result={}
    for phase,sequence,updates in [('generation',generation_stream(rows,17),WARM_UPDATES),
                                   ('joint',recovery_stream(rows,17),JOINT_UPDATES)]:
        counts=Counter();styles=Counter();materials=Counter();maximum=0;file=out/f'{phase}_exposure.jsonl'
        with file.open('w') as f:
            for _ in range(updates*16):
                e=next(sequence);maximum=max(maximum,encoded_length(e,tok));assert maximum<=1536
                counts[e['task']]+=1;styles[e['generation_style']]+=1;materials[e['source_id']]+=1
                f.write(json.dumps(e)+'\n')
        result[phase]=dict(tasks=dict(counts),styles=dict(styles),material_exposure_min=min(materials.values()),
            material_exposure_max=max(materials.values()),materials=len(materials),max_length=maximum,sha256=sha(file.read_bytes()))
    write_json(out/'exposure_audit.json',result)


def evaluation(model,tok,mode,rows,out,materials,split='val'):
    out.mkdir();legacy=evaluate(model,tok,mode,rows,out,split)
    v2,scored=score_evaluation(out,materials)
    write_json(out/'rescored_summary_v2.json',dict(**v2,raw_sha256=sha((out/'evaluation.jsonl').read_bytes()),
        scorer_sha256=sha((ROOT/'rescore.py').read_bytes())))
    (out/'rescored_evaluation_v2.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    return dict(legacy=legacy,v2=v2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    assert json.loads((args.initial/'summary.json').read_text())['status']=='complete'
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    train=records(args.data,'train');val=sorted(records(args.data,'val'),key=lambda r:sha(r['id'].encode()))
    assert len(train)==1024 and len(val)==128
    assert not {r['id'] for r in train}&{r['id'] for r in val}
    old=records(ROOT/'smoke_data_002','train');materials={r['id']:r for r in train+val}
    tok=AutoTokenizer.from_pretrained(args.data/'tokenizer');freeze_exposure(train,tok,args.out)
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    write_json(args.out/'manifest.json',dict(train_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in val],
        previously_diagnosed_validation_ids=[r['id'] for r in val[:64]],fresh_validation_ids=[r['id'] for r in val[64:]],
        source_manifest_sha256=sha((args.data/'manifest.json').read_bytes()),warm_updates=WARM_UPDATES,
        joint_updates=JOINT_UPDATES,initial=str(args.initial),test_used=False,answer_repair=False,generation_prompts=GEN_PROMPTS))
    results={}
    for mode in ['hybrid','ar']:
        torch.manual_seed(17);torch.cuda.manual_seed_all(17);model=Hybrid.pretrained(tok)
        saved=torch.load(args.initial/mode/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert saved['config']['source_sha256']['model.py']==sha((ROOT/'model.py').read_bytes())
        model.load_state_dict(saved['model']);del saved;model=model.cuda();destination=args.out/mode
        torch.cuda.reset_peak_memory_stats()
        model,receipt=train_phase(model,tok,mode,train,destination/'generation','warmup',False,17,
            sequence=generation_stream(train,17),updates=WARM_UPDATES)
        receipt['peak_gpu_MB']=torch.cuda.max_memory_allocated()/2**20
        warm=evaluation(model,tok,mode,val[:64],destination/'generation/validation_64',materials)
        receipt.update(status='validated',validation=warm);write_json(destination/'generation/status.json',receipt)
        results[mode]=dict(generation=warm)
        write_json(args.out/'summary.json',dict(status='partial',results=results))
        torch.cuda.reset_peak_memory_stats()
        model,receipt=train_phase(model,tok,mode,train,destination/'joint','joint',True,17,
            sequence=recovery_stream(train,17),updates=JOINT_UPDATES)
        receipt['peak_gpu_MB']=torch.cuda.max_memory_allocated()/2**20
        joint=evaluation(model,tok,mode,val,destination/'joint/validation_128',materials)
        reading=unseen_reading(model,tok,mode,val,destination/'joint/validation_128')
        regression=evaluation(model,tok,mode,old,destination/'joint/old8_regression',materials,'train')
        receipt.update(status='validated',validation=joint,reading=reading,old8_regression=regression)
        write_json(destination/'joint/status.json',receipt)
        results[mode].update(joint=joint,reading=reading,old8_regression=regression)
        write_json(args.out/'summary.json',dict(status='partial',results=results))
        del model;torch.cuda.empty_cache()
    write_json(args.out/'summary.json',dict(status='complete',results=results,
        note='128 validation materials, including 64 fresh materials. Test256 unused; no claim of goal completion.'))


if __name__=='__main__':main()
