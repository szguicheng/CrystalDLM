"""No-training diagnosis with common RNG and crossed pair/prompt factors."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from adapt import conflict_example, TRAIN_OFFSETS, TEST_OFFSETS
from model import Hybrid
from prepare import END, ROOT, TEMPLATES, sha, tasks, write_json
from rescore import reading_grade
from run import attempt, generate, records
from score import grade_edit, parse


def canonical_crystal(text):
    p=parse(text)
    def number(value,places):
        rendered=f'{value:.{places}f}'
        return rendered if float(rendered)==value else repr(value)
    symbol=text.strip().splitlines()[1].split(': ',1)[1].strip()
    lines=[p['formula'],'Spacegroup: '+symbol,
           'abc: '+' '.join(number(v,2) for v in p['lengths']),
           'angles: '+' '.join(number(v,2) for v in p['angles']),f"Sites ({p['declared_atoms']})"]
    for site in p['sites']:
        lines.append(site['element']+' '+' '.join(number(v,3) for v in site['coords'])+' '+site['label'])
    result='\n'.join(lines)
    assert parse(result)==p,'canonicalization changed the structure'
    return result


def actual_edit_prefix(row,evaluation,canonical=False):
    source=evaluation['generation']['value']['crystal']
    if canonical:source=canonical_crystal(source)
    ex=tasks(row,'train',evaluation['direction'])
    _,question,modify=TEMPLATES['train'][0]
    read=evaluation['actual_read']['value']
    scale='1.05' if evaluation['direction']=='plus' else '.95'
    return (ex[0]['prefix']+evaluation['generation']['value']['lead']+source+END+
            '\nUser: '+question+'\nAssistant:'+read+'\nUser: '+modify.format(scale=scale)+'\nAssistant:')


def load_model(data,checkpoint):
    tok=AutoTokenizer.from_pretrained(data/'tokenizer')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=Hybrid.pretrained(tok)
    model.load_state_dict(saved['model'])
    provenance=saved['config'];del saved
    return model.cuda().eval(),tok,provenance


def edit_diagnostic(model,tok,rows,legacy,out):
    source_records={r['id']:r for r in map(json.loads,legacy.read_text().splitlines())}
    results=[]
    with (out/'matched_edits.jsonl').open('w') as log:
        for row in rows:
            previous=source_records[row['id']]
            assert previous['G'] and previous.get('actual_read',{}).get('ok'),'all8 consistent-model generated inputs must be available'
            actual=previous['generation']['value']['crystal']
            ex=tasks(row,'train',previous['direction'])
            variants={'reference':(ex[2]['prefix'],row['c1']),
                      'actual':(actual_edit_prefix(row,previous),actual),
                      'canonical_structure':(actual_edit_prefix(row,previous,True),actual)}
            scale='1.05' if previous['direction']=='plus' else '.95'
            for repeat in range(3):
                seed=int(sha(f"common-history:{row['id']}:{repeat}".encode())[:8],16)
                for name,(prefix,target) in variants.items():
                    rng=torch.Generator(device='cuda').manual_seed(seed)
                    initial=sha(rng.get_state().cpu().numpy().tobytes())
                    produced=attempt(lambda:generate(model,tok,'hybrid',prefix,rng))
                    scored=grade_edit(produced['value']['crystal'],target,scale) if produced['ok'] else dict(success=False)
                    result=dict(id=row['id'],repeat=repeat,seed=seed,variant=name,
                         initial_rng_sha256=initial,prefix_sha256=sha(prefix.encode()),
                         reference_same_physics=parse(actual)==parse(row['c1']),
                         response=produced,grade=scored)
                    log.write(json.dumps(result)+'\n');log.flush();results.append(result)
            print(json.dumps(dict(task='matched_edits',materials_done=len(results)//9,total=8)),flush=True)
    summary={name:dict(n=24,success=sum(r['grade']['success'] for r in results if r['variant']==name)) for name in variants}
    for row in rows:
        for repeat in range(3):
            matched=[r for r in results if r['id']==row['id'] and r['repeat']==repeat]
            assert len({r['initial_rng_sha256'] for r in matched})==1
    write_json(out/'edit_summary.json',summary)
    return summary


def reading_diagnostic(model,tok,rows,out):
    results=[]
    with (out/'crossed_reading.jsonl').open('w') as log:
        for pair_kind,offsets in [('seen_pairs',TRAIN_OFFSETS),('heldout_pairs',TEST_OFFSETS)]:
            for index,old in enumerate(rows):
                for offset in offsets:
                    current=rows[(index+offset)%8]
                    train=conflict_example(old,current['c1'],'train')
                    test=conflict_example(old,current['c1'],'test')
                    query_only=train['prefix'].rsplit('\nUser:',1)[0]+'\nUser:'+test['prefix'].rsplit('\nUser:',1)[1]
                    for style,prefix in [('train',train['prefix']),('query_only',query_only),('full_test',test['prefix'])]:
                        response=attempt(lambda:model.text(prefix,tok,'hybrid'))
                        scored=reading_grade(response['value'],current['c1']) if response['ok'] else dict(success=False,fields={})
                        result=dict(old_id=old['id'],current_id=current['id'],pair_kind=pair_kind,prompt_style=style,
                                    prefix_sha256=sha(prefix.encode()),response=response,grade=scored)
                        log.write(json.dumps(result)+'\n');log.flush();results.append(result)
                print(json.dumps(dict(task='crossed_reading',pair_kind=pair_kind,old_materials_done=index+1,total=8)),flush=True)
    summary={}
    for pair_kind in ['seen_pairs','heldout_pairs']:
        summary[pair_kind]={}
        for style in ['train','query_only','full_test']:
            subset=[r for r in results if r['pair_kind']==pair_kind and r['prompt_style']==style]
            fields=Counter()
            for r in subset:fields.update({k:int(v) for k,v in r['grade']['fields'].items()})
            summary[pair_kind][style]=dict(n=len(subset),success=sum(r['grade']['success'] for r in subset),correct_by_field=dict(fields))
    assert len(results)==168
    write_json(out/'reading_summary.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'),'use a compute GPU'
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    rows=records(args.data,'train');started=time.monotonic()
    model,tok,edit_config=load_model(args.data,args.run/'hybrid/consistent/checkpoint.pt')
    edits=edit_diagnostic(model,tok,rows,args.run/'hybrid/consistent/evaluation.jsonl',args.out)
    del model,tok;torch.cuda.empty_cache()
    model,tok,read_config=load_model(args.data,args.run/'hybrid/feedback/checkpoint.pt')
    reading=reading_diagnostic(model,tok,rows,args.out)
    write_json(args.out/'summary.json',dict(status='complete',training_updates=0,edit_summary=edits,reading_summary=reading,
       edit_checkpoint_config=edit_config,reading_checkpoint_config=read_config,elapsed_seconds=time.monotonic()-started,
       slurm_job=os.environ['SLURM_JOB_ID'],code_sha256=sha(Path(__file__).read_bytes()),
       note='Controlled diagnosis only. Reference and canonical histories are probes, not hard-coded production answers.'))
