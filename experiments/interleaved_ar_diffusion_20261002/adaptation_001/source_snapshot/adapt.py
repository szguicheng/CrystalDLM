"""Fixed second pilot: crystal adaptation, then matched feedback/control branches."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer
from model import Hybrid
from prepare import END, EOS, ROOT, START, TEMPLATES, encoded_length, sha, tasks, write_json
from run import attempt, autocast, evaluate, records
from score import answer, audit, edit, grade_answer, parse

TRAIN_OFFSETS = (2, 3, 5)
TEST_OFFSETS = (1, 4, 6, 7)
WARMUP_UPDATES, JOINT_UPDATES = 600, 200


def conflict_example(old, current_text, split):
    condition = old['condition']
    gen, read, _ = TEMPLATES[split][0]
    instruction = ('Use the current crystal block as the source of truth even when earlier design conditions disagree. '
                   if split=='train' else 'Read the current replacement crystal, ignoring the earlier design conditions. ')
    prefix = ('User: '+gen.format(**condition)+'\nAssistant: I generated a crystal.\n'+START+
              current_text+END+'\nUser: '+instruction+read+'\nAssistant:')
    return dict(task='Q_conflict', prefix=prefix, target=' '+answer(current_text)+EOS)


def phase_stream(rows, phase, feedback, seed):
    rng=random.Random(seed);epoch=0
    while True:
        indices=list(range(len(rows)));rng.shuffle(indices)
        for index in indices:
            row=rows[index]
            direction='plus' if (int(sha(row['id'].encode())[:8],16)+epoch)%2 else 'minus'
            examples=tasks(row,'train',direction,epoch%2)
            if phase=='warmup':
                for e in [examples[0],examples[2]]:
                    yield dict(e,supervise_lead=False)
            else:
                if feedback and epoch%2:
                    offset=TRAIN_OFFSETS[(epoch//2)%len(TRAIN_OFFSETS)]
                    other=rows[(index+offset)%len(rows)]
                    variant=['c1','plus','minus'][(epoch//(2*len(TRAIN_OFFSETS)))%3]
                    examples[1]=conflict_example(row,other[variant],'train')
                yield from examples
        epoch+=1


def noise_step(phase, rng):
    if rng.random()<.5:
        return 32
    return rng.randint(16 if phase=='warmup' else 1,31)


def fingerprint(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def freeze_pairs(data,out):
    rows=records(data,'train');assert len(rows)==8
    assert not set(TRAIN_OFFSETS).intersection(TEST_OFFSETS)
    checks={}
    for row in rows:
        for field in ['c1','plus','minus']:
            checks[row['id']+':'+field]=audit(row[field])
        checks[row['id']+':1.03']=audit(edit(row['c1'],'1.03'))
    assert all(v['valid'] for v in checks.values()), 'a predefined counterfactual structure is invalid'
    tok=AutoTokenizer.from_pretrained(data/'tokenizer')
    training=[];testing=[]
    for index,old in enumerate(rows):
        for offset in TRAIN_OFFSETS:
            current=rows[(index+offset)%8]
            for variant in ['c1','plus','minus']:
                e=conflict_example(old,current[variant],'train')
                assert grade_answer(e['target'],current[variant])['success']
                assert encoded_length(e,tok)<=1536
                training.append(dict(old_id=old['id'],current_id=current['id'],variant=variant,**e))
        for offset in TEST_OFFSETS:
            current=rows[(index+offset)%8]
            for scale in ['1','1.03']:
                text=current['c1'] if scale=='1' else edit(current['c1'],scale)
                e=conflict_example(old,text,'test')
                assert encoded_length(e,tok)<=1536
                testing.append(dict(old_id=old['id'],current_id=current['id'],scale=scale,crystal=text,**e))
    train_pairs={(r['old_id'],r['current_id']) for r in training}
    test_pairs={(r['old_id'],r['current_id']) for r in testing}
    assert len(train_pairs)==24 and len(test_pairs)==32 and train_pairs.isdisjoint(test_pairs)
    for name,values in [('conflict_train',training),('conflict_test',testing)]:
        with (out/(name+'.jsonl')).open('w') as f:
            for value in values:f.write(json.dumps(value)+'\n')
    write_json(out/'data_manifest.json',dict(status='frozen',material_ids=[r['id'] for r in rows],
       source_manifest_sha256=sha((data/'manifest.json').read_bytes()),source_rows_sha256=sha((data/'train.jsonl').read_bytes()),
       train_offsets=TRAIN_OFFSETS,test_offsets=TEST_OFFSETS,training_unique_pairs=24,test_unique_pairs=32,
       train_examples=72,test_examples=64,training_variants=['c1','plus','minus'],test_scales=['1','1.03'],
       audits=checks,files={name:sha((out/(name+'.jsonl')).read_bytes()) for name in ['conflict_train','conflict_test']}))
    return rows


def train_phase(model,tok,mode,rows,out,phase,feedback,seed,warm_fingerprint=None):
    out.mkdir(parents=True,exist_ok=False)
    updates=WARMUP_UPDATES if phase=='warmup' else JOINT_UPDATES
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    sequence=phase_stream(rows,phase,feedback,seed)
    corruption_rng=torch.Generator().manual_seed(seed+1000)
    schedule_rng=random.Random(seed+2000)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    initial=fingerprint(model)
    if warm_fingerprint:assert initial==warm_fingerprint,'joint branch initialization differs from shared adaptation checkpoint'
    receipt=dict(mode=mode,phase=phase,feedback=feedback,updates=updates,seed=seed,status='training',
                 initial_fingerprint=initial,warm_fingerprint=warm_fingerprint,slurm_job=os.environ['SLURM_JOB_ID'],
                 host=socket.gethostname(),gpu=torch.cuda.get_device_name(),
                 noise_policy='50% full MASK; remaining uniform 16..31' if phase=='warmup' else '50% full MASK; remaining uniform 1..31',
                 source_sha256={p.name:sha(p.read_bytes()) for p in ROOT.glob('*.py')})
    write_json(out/'status.json',receipt)
    model.train();begun=time.monotonic();timings=[];noise_counts=Counter()
    with (out/'curve.jsonl').open('w') as log:
        for update in range(1,updates+1):
            torch.cuda.synchronize();start=time.monotonic()
            multiplier=update/40 if update<=40 else .5*(1+math.cos(math.pi*(update-40)/(updates-40)))
            for p in optimizer.param_groups:p['lr']=2e-5*multiplier
            optimizer.zero_grad(set_to_none=True);losses=[];text=[];crystal=[];kinds=Counter()
            for _ in range(8):
                batch=[next(sequence),next(sequence)]
                for e in batch:
                    kinds[e['task']]+=1
                    if mode=='hybrid' and 'crystal' in e:
                        e['noise_step']=noise_step(phase,schedule_rng);noise_counts[e['noise_step']]+=1
                with autocast():loss,stats=model.loss(batch,tok,mode,corruption_rng)
                assert torch.isfinite(loss),'nonfinite objective'
                (loss/8).backward();losses.append(float(loss.detach()))
                if stats['text'] is not None:text.append(stats['text'])
                if stats['crystal'] is not None:crystal.append(stats['crystal'])
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm)
            optimizer.step();torch.cuda.synchronize();seconds=time.monotonic()-start
            if update>20:timings.append(seconds)
            value=dict(update=update,loss=sum(losses)/len(losses),text_loss=sum(text)/len(text) if text else None,
                       crystal_loss=sum(crystal)/len(crystal) if crystal else None,seconds=seconds,task_counts=dict(kinds))
            log.write(json.dumps(value)+'\n');log.flush()
            if update%25==0 or update==updates:
                receipt.update(completed_updates=update,elapsed_seconds=time.monotonic()-begun,noise_counts=dict(noise_counts))
                write_json(out/'status.json',receipt)
                print(json.dumps(dict(mode=mode,phase=phase,feedback=feedback,**value)),flush=True)
    receipt.update(status='trained',training_seconds=time.monotonic()-begun,
                   seconds_per_update_median=sorted(timings)[len(timings)//2],final_fingerprint=fingerprint(model))
    write_json(out/'status.json',receipt)
    state=dict(model=model.state_dict(),config=receipt,update=updates)
    if phase!='warmup':state.update(optimizer=optimizer.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(state,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
    return model.eval(),receipt


def pair_evaluation(model,tok,mode,tests,out):
    rows=[]
    with (out/'conflict_evaluation.jsonl').open('w') as f:
        for case in tests:
            response=attempt(lambda:model.text(case['prefix'],tok,mode))
            grade=grade_answer(response['value'],case['crystal']) if response['ok'] else dict(success=False,fields={})
            row=dict(old_id=case['old_id'],current_id=case['current_id'],scale=case['scale'],response=response,grade=grade)
            f.write(json.dumps(row)+'\n');f.flush();rows.append(row)
            if len(rows)%16==0:print(json.dumps(dict(mode=mode,conflict_evaluated=len(rows),total=len(tests))),flush=True)
    summary={}
    for scale in ['1','1.03']:
        subset=[r for r in rows if r['scale']==scale]
        fields=Counter()
        for row in subset:fields.update({k:int(v) for k,v in row['grade']['fields'].items()})
        summary[scale]=dict(n=len(subset),success=sum(r['grade']['success'] for r in subset),correct_by_field=dict(fields))
    write_json(out/'conflict_summary.json',summary)
    return summary


def diagnostic(data,checkpoint,out):
    subprocess.run([sys.executable,str(ROOT/'diagnose.py'),'--data',str(data),
                    '--checkpoint',str(checkpoint),'--out',str(out)],check=True)
    return json.loads(out.read_text())['summary']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.data=args.data.resolve();args.out=args.out.resolve()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID'),'use an allocated GPU compute node'
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for p in [*ROOT.glob('*.py'),ROOT/'PROTOCOL_ADAPT_V2.md']:shutil.copy2(p,snapshot/p.name)
    rows=freeze_pairs(args.data,args.out)
    tests=[json.loads(s) for s in (args.out/'conflict_test.jsonl').read_text().splitlines()]
    results={}
    for mode in ['ar','hybrid']:
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        tok=AutoTokenizer.from_pretrained(args.data/'tokenizer');model=Hybrid.pretrained(tok).cuda()
        destination=args.out/mode;destination.mkdir()
        model,warm=train_phase(model,tok,mode,rows,destination/'warmup','warmup',False,17)
        del model;torch.cuda.empty_cache()
        results[mode]=dict(warmup=warm,branches={})
        if mode=='hybrid':results[mode]['warmup_diagnostic']=diagnostic(args.data,destination/'warmup/checkpoint.pt',destination/'warmup/diagnostics.json')
        for branch in ['consistent','feedback']:
            torch.manual_seed(17);torch.cuda.manual_seed_all(17)
            model=Hybrid.pretrained(tok)
            saved=torch.load(destination/'warmup/checkpoint.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model']);del saved;model=model.cuda()
            folder=destination/branch
            model,receipt=train_phase(model,tok,mode,rows,folder,'joint',branch=='feedback',17,warm['final_fingerprint'])
            original=evaluate(model,tok,mode,rows,folder,'train')
            conflicts=pair_evaluation(model,tok,mode,tests,folder)
            receipt.update(status='evaluated',original_evaluation=original,conflict_evaluation=conflicts)
            write_json(folder/'status.json',receipt)
            results[mode]['branches'][branch]=dict(original=original,conflicts=conflicts)
            del model;torch.cuda.empty_cache()
            if mode=='hybrid':results[mode]['branches'][branch]['diagnostic']=diagnostic(args.data,folder/'checkpoint.pt',folder/'diagnostics.json')
            write_json(args.out/'summary.json',dict(status='partial',results=results))
    passed={mode:all([r['branches']['feedback']['original']['stage0_passed'],
                    r['branches']['feedback']['original']['counts']['Q_swap']>=7,
                    r['branches']['feedback']['conflicts']['1']['success']>=28]) for mode,r in results.items()}
    write_json(args.out/'summary.json',dict(status='complete',results=results,mechanism_gate=passed,
       note='Training structures seen; counterfactual pairings and 1.03 perturbation held out. No 1024-material training launched.'))


if __name__=='__main__':main()
