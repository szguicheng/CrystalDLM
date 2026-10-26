"""Train on 1024 materials with independent augmentation factors; validate unseen structures."""
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
from align import READ_TRAIN, READ_TEST, caption, edit_question, read_example
from model import Hybrid
from prepare import END, ROOT, START, edit_factor, encoded_length, sha, tasks, write_json
from run import attempt, evaluate, records
from rescore import reading_grade
from score import edit, parse

UPDATES=2048


def balanced_edit(row,direction,rng):
    source_kind=rng.choice(['c1','plus','minus']);source=row[source_kind]
    style=rng.randrange(6);history_style=rng.randrange(3);caption_style=rng.randrange(3)
    ex=tasks(row,'train',direction,rng.randrange(2))
    lead=[' I will generate the requested crystal.\n',' Here is the proposed crystal.\n',' Candidate crystal:\n'][history_style]
    prefix=(ex[0]['prefix']+lead+START+source+END+'\nUser: '+rng.choice(READ_TRAIN)+
            '\nAssistant:'+caption(source,caption_style)+'\nUser: '+edit_question(direction,style)+'\nAssistant:')
    return dict(task='E_varied',direction=direction,edit_style=style,source_kind=source_kind,
          history_style=history_style,caption_style=caption_style,prefix=prefix,lead=ex[2]['lead'],
          crystal=edit(source,edit_factor(direction)))


def balanced_stream(rows,seed):
    rng=random.Random(seed)
    while True:
        order=list(range(len(rows)));rng.shuffle(order)
        for i in order:
            row=rows[i];direction=rng.choice(['plus','minus'])
            yield from tasks(row,'train',direction,rng.randrange(2))
            for _ in range(2):
                other=rng.randrange(len(rows)-1);other+=other>=i
                current=rows[other];text=current[rng.choice(['c1','plus','minus'])]
                yield read_example(row,text,rng.choice(READ_TRAIN))
            yield balanced_edit(row,rng.choice(['plus','minus']),rng)
            yield read_example(row,row[rng.choice(['c1','plus','minus'])],rng.choice(READ_TRAIN))


def unseen_reading(model,tok,mode,rows,out):
    cases=[]
    with (out/'unseen_reading.jsonl').open('w') as f:
        for index,old in enumerate(rows):
            current=rows[(index+1)%len(rows)]
            question=READ_TEST[index%len(READ_TEST)]
            e=read_example(old,current['c1'],question)
            response=attempt(lambda:model.text(e['prefix'],tok,mode))
            grade=reading_grade(response['value'],current['c1']) if response['ok'] else dict(success=False,fields={})
            row=dict(old_id=old['id'],current_id=current['id'],question=question,response=response,grade=grade)
            f.write(json.dumps(row)+'\n');f.flush();cases.append(row)
            if (index+1)%16==0:print(json.dumps(dict(mode=mode,unseen_reading=index+1,total=len(rows))),flush=True)
    fields=Counter()
    for c in cases:fields.update({k:int(v) for k,v in c['grade']['fields'].items()})
    result=dict(n=len(cases),success=sum(c['grade']['success'] for c in cases),correct_by_field=dict(fields))
    write_json(out/'unseen_reading_summary.json',result)
    return result


def audit_exposure(rows,tok,out):
    kinds=Counter();factors=Counter();length=0
    sequence=balanced_stream(rows,17)
    with (out/'training_exposure.jsonl').open('w') as f:
        for _ in range(UPDATES*16):
            e=next(sequence);kinds[e['task']]+=1;length=max(length,encoded_length(e,tok))
            assert length<=1536
            if 'crystal' in e:
                assert len(tok.encode(e['crystal'],add_special_tokens=False))+1<=256
                parse(e['crystal'])
            if e['task']=='E_varied':factors[f"{e['direction']}:{e['edit_style']}"]+=1
            f.write(json.dumps(e)+'\n')
    assert all(factors[f'{d}:{i}']>0 for d in ['plus','minus'] for i in range(6))
    write_json(out/'exposure_audit.json',dict(kinds=dict(kinds),direction_style=dict(factors),max_length=length,
             exposure_sha256=sha((out/'training_exposure.jsonl').read_bytes())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    train=records(args.data,'train');val=records(args.data,'val')
    assert len(train)==1024 and len(val)==128
    assert {r['id'] for r in train}.isdisjoint(r['id'] for r in val)
    validation=sorted(val,key=lambda r:sha(r['id'].encode()))[:64]
    tok=AutoTokenizer.from_pretrained(args.data/'tokenizer')
    audit_exposure(train,tok,args.out)
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    write_json(args.out/'manifest.json',dict(train_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in validation],
        source_manifest_sha256=sha((args.data/'manifest.json').read_bytes()),updates=UPDATES,
        test_materials_used_for_training_or_selection=False,inference_answer_repair=False))
    results={}
    for mode in ['hybrid','ar']:
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        model=Hybrid.pretrained(tok)
        saved=torch.load(args.initial/mode/'checkpoint.pt',map_location='cpu',weights_only=True)
        model.restore_checkpoint(saved,expected_mode=mode);del saved;model=model.cuda()
        destination=args.out/mode
        model,receipt=train_phase(model,tok,mode,train,destination,'cohort',True,17,
                         sequence=balanced_stream(train,17),updates=UPDATES)
        validation_out=destination/'validation_64';validation_out.mkdir()
        core=evaluate(model,tok,mode,validation,validation_out,'val')
        reads=unseen_reading(model,tok,mode,validation,validation_out)
        regression_out=destination/'old8_regression';regression_out.mkdir()
        regression=evaluate(model,tok,mode,records(ROOT/'smoke_data_002','train'),regression_out,'train')
        receipt.update(status='validated',validation=core,reading=reads,old8_regression=regression)
        write_json(destination/'status.json',receipt)
        results[mode]=dict(validation=core,reading=reads,old8_regression=regression)
        write_json(args.out/'summary.json',dict(status='partial',results=results))
        del model;torch.cuda.empty_cache()
    write_json(args.out/'summary.json',dict(status='complete',results=results,
         note='64 genuinely unseen validation materials.256 test materials remain unused; goal not yet complete.'))
