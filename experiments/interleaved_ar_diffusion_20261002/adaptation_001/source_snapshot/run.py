"""Paired training and raw, denominator-preserving interleaved evaluation."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import time
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer
from model import Hybrid
from prepare import END, ROOT, START, TEMPLATES, sha, tasks, write_json
from score import audit, grade_answer, grade_edit, parse


def records(data, split):
    return [json.loads(line) for line in (data / f'{split}.jsonl').read_text().splitlines()]


def stream(rows, seed):
    rng = random.Random(seed)
    epoch = 0
    while True:
        indices = list(range(len(rows))); rng.shuffle(indices)
        for index in indices:
            row = rows[index]
            direction = 'plus' if (int(sha(row['id'].encode())[:8], 16)+epoch)%2 else 'minus'
            yield from tasks(row, 'train', direction, epoch%2)
        epoch += 1


def autocast():
    return torch.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()


def train(mode, data, out, updates, seed):
    out.mkdir(parents=True, exist_ok=False)
    assert torch.cuda.is_available(), 'Training must run on an allocated GPU compute node'
    assert os.environ.get('SLURM_JOB_ID'), 'Run inside a Slurm allocation'
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(data/'tokenizer')
    model = Hybrid.pretrained(tok).cuda().train()
    rng = torch.Generator().manual_seed(seed+1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    sequence = stream(records(data, 'train'), seed)
    receipt = dict(mode=mode, status='training', updates=updates, seed=seed,
                   initial_added_embedding_sha256=sha(model.base.get_input_embeddings().weight[-4:].detach().cpu().numpy().tobytes()),
                   host=socket.gethostname(), slurm_job=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
                   packages={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','pyxtal','pymatgen','spglib']},
                   code_hashes={p.name:sha(p.read_bytes()) for p in ROOT.glob('*.py')},
                   dataset_manifest_sha256=sha((data/'manifest.json').read_bytes()),
                   microbatch=2, accumulation=8, effective_batch=16)
    write_json(out/'status.json', receipt)
    timings = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (out/'curve.jsonl').open('w') as log:
        for update in range(1, updates+1):
            torch.cuda.synchronize(); begin = time.monotonic()
            warmup = min(40, updates)
            factor = update/warmup if update<=warmup else .5*(1+math.cos(math.pi*(update-warmup)/(updates-warmup)))
            for group in optimizer.param_groups:
                group['lr'] = 2e-5*factor
            optimizer.zero_grad(set_to_none=True)
            losses, texts, crystals = [], [], []
            for _ in range(8):
                examples = [next(sequence), next(sequence)]
                with autocast():
                    loss, stats = model.loss(examples, tok, mode, rng)
                assert torch.isfinite(loss), 'nonfinite objective'
                (loss/8).backward()
                losses.append(float(loss.detach()))
                if stats['text'] is not None: texts.append(stats['text'])
                if stats['crystal'] is not None: crystals.append(stats['crystal'])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            assert torch.isfinite(norm), 'nonfinite gradient norm'
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.monotonic()-begin
            if update>20: timings.append(seconds)
            entry = dict(update=update, loss=sum(losses)/len(losses), text_loss=sum(texts)/len(texts),
                         crystal_loss=sum(crystals)/len(crystals), seconds=seconds, lr=optimizer.param_groups[0]['lr'])
            log.write(json.dumps(entry)+'\n'); log.flush()
            if update%10==0 or update==updates:
                print(json.dumps(dict(mode=mode, **entry)), flush=True)
                receipt.update(completed_updates=update, elapsed_seconds=time.monotonic()-started,
                               peak_gpu_MB=torch.cuda.max_memory_allocated()/2**20)
                write_json(out/'status.json', receipt)
            if update==50:
                write_json(out/'throughput.json', dict(seconds_per_update_median=float(np.median(timings)),
                       estimated_800_update_seconds=float(np.median(timings))*800,
                       peak_gpu_MB=torch.cuda.max_memory_allocated()/2**20))
            if update%200==0 or update==updates:
                temporary = out/'checkpoint.tmp'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                      cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                      corruption_rng=rng.get_state(), config=receipt), temporary)
                temporary.replace(out/'checkpoint.pt')
    receipt.update(status='trained', training_seconds=time.monotonic()-started)
    write_json(out/'status.json', receipt)
    return model.eval(), tok, receipt


def attempt(function):
    try:
        with warnings.catch_warnings(), autocast():
            warnings.simplefilter('ignore')
            return dict(ok=True, value=function())
    except Exception as error:
        return dict(ok=False, error=f'{type(error).__name__}: {error}')


def generate(model, tok, mode, prefix, rng):
    lead = model.text(prefix, tok, mode, trigger=True)
    crystal, tokens = model.crystal(prefix+lead, tok, mode, rng)
    return dict(lead=lead, crystal=crystal, tokens=tokens)


def evaluate(model, tok, mode, rows, out, split):
    result = []
    keys = ['G','Q_reference','E_reference','closed_loop','flow_complete','Q_swap','Q_no_crystal']
    begun = time.monotonic()
    with (out/'evaluation.jsonl').open('w') as log:
        for index, row in enumerate(rows):
            torch.cuda.synchronize(); start = time.monotonic()
            before = model.forward_count
            seed = int(sha((row['id']+'20261002').encode())[:8], 16)
            rng = torch.Generator(device='cuda').manual_seed(seed)
            direction = 'plus' if int(sha(row['id'].encode())[:8],16)%2 else 'minus'
            scale = '1.05' if direction=='plus' else '.95'
            ex = tasks(row, split, direction)
            _, question, change = TEMPLATES[split][0]
            record = dict(id=row['id'], direction=direction, **{k:False for k in keys})
            g = attempt(lambda: generate(model, tok, mode, ex[0]['prefix'], rng)); record['generation']=g
            if g['ok']:
                checked = audit(g['value']['crystal'], row['condition'])
                record['generation_audit']=checked; record['G']=checked['valid']
            q = attempt(lambda: model.text(ex[1]['prefix'],tok,mode)); record['reference_read']=q
            if q['ok']:
                record['reference_read_grade']=grade_answer(q['value'],row['c1'])
                record['Q_reference']=record['reference_read_grade']['success']
            e = attempt(lambda: generate(model,tok,mode,ex[2]['prefix'],rng)); record['reference_edit']=e
            if e['ok']:
                record['reference_edit_grade']=grade_edit(e['value']['crystal'],row['c1'],scale)
                record['E_reference']=record['reference_edit_grade']['success']
            if record['G']:
                c1 = g['value']['crystal']
                history = ex[0]['prefix']+g['value']['lead']+c1+END+'\nUser: '+question+'\nAssistant:'
                cq = attempt(lambda: model.text(history,tok,mode)); record['actual_read']=cq
                if cq['ok']:
                    record['actual_read_grade']=grade_answer(cq['value'],c1)
                    editing = history+cq['value']+'\nUser: '+change.format(scale=scale)+'\nAssistant:'
                    ce = attempt(lambda: generate(model,tok,mode,editing,rng)); record['actual_edit']=ce
                    if ce['ok']:
                        c2 = ce['value']['crystal']
                        record['actual_edit_grade']=grade_edit(c2,c1,scale)
                        if audit(c2)['valid']:
                            reading = editing+ce['value']['lead']+c2+END+'\nUser: '+question+'\nAssistant:'
                            cq2 = attempt(lambda: model.text(reading,tok,mode)); record['actual_second_read']=cq2
                            if cq2['ok']:
                                record['flow_complete']=True
                                record['actual_second_read_grade']=grade_answer(cq2['value'],c2)
                                record['closed_loop'] = (record['actual_read_grade']['success'] and
                                   record['actual_edit_grade']['success'] and record['actual_second_read_grade']['success'])
            swap = next(other for other in rows[index+1:]+rows[:index+1]
                        if sum(parse(other['c1'])[f]!=parse(row['c1'])[f]
                               for f in ['counts','sg','atoms','orbits','lengths'])>=2)
            swap_question = 'Read the current replacement crystal, ignoring the earlier design conditions. '+question
            prior = ex[0]['prefix']+' I generated a crystal.\n'
            replacement = prior+START+swap['c1']+END+'\nUser: '+swap_question+'\nAssistant:'
            qs = attempt(lambda: model.text(replacement,tok,mode)); record['swap_read']=qs; record['swap_id']=swap['id']
            if qs['ok']: record['Q_swap']=grade_answer(qs['value'],swap['c1'])['success']
            absent = prior+'\nUser: '+swap_question+'\nAssistant:'
            qn = attempt(lambda: model.text(absent,tok,mode)); record['no_crystal_read']=qn
            if qn['ok']: record['Q_no_crystal']=grade_answer(qn['value'],swap['c1'])['success']
            torch.cuda.synchronize()
            record.update(seconds=time.monotonic()-start, forward_calls=model.forward_count-before)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            result.append(record)
            print(json.dumps(dict(mode=mode,evaluated=index+1,total=len(rows),counts={k:sum(r[k] for r in result) for k in keys})),flush=True)
    totals = {key:sum(r[key] for r in result) for key in keys}
    summary = dict(n=len(rows),counts=totals,rates={k:v/len(rows) for k,v in totals.items()},
                   elapsed_seconds=time.monotonic()-begun,attempts_retried=0,
                   seconds_median=float(np.median([r['seconds'] for r in result])),
                   seconds_p95=float(np.percentile([r['seconds'] for r in result],95)),
                   evaluation_split=split,training_materials_evaluated=split=='train')
    summary['stage0_passed'] = all(totals[k]>=math.ceil(.875*len(rows)) for k in ['G','Q_reference','E_reference','flow_complete'])
    write_json(out/'evaluation_summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--mode',choices=['ar','hybrid','pair'],default='pair')
    parser.add_argument('--seed',type=int,default=17)
    args = parser.parse_args()
    assert args.updates<=200, 'This runner is the stage-0 gate; stage-1 requires a separate frozen budget'
    args.out.mkdir(parents=True,exist_ok=False)
    rows = records(args.data,'train')
    assert len(rows)==8, 'stage-0 must use the frozen 8-material cohort'
    torch.set_num_threads(4)
    results = {}
    for mode in (['ar','hybrid'] if args.mode=='pair' else [args.mode]):
        destination=args.out/mode
        model,tok,receipt=train(mode,args.data,destination,args.updates,args.seed)
        result=evaluate(model,tok,mode,rows,destination,'train')
        receipt.update(status='evaluated',summary=result); write_json(destination/'status.json',receipt)
        results[mode]=result
        write_json(args.out/'summary.json',dict(status='partial',results=results))
        del model,tok; torch.cuda.empty_cache()
    write_json(args.out/'summary.json',dict(status='complete',results=results,
               stage0_pair_passed=all(r['stage0_passed'] for r in results.values()),
               note='8 training materials only; not held-out performance or material stability evidence'))


if __name__=='__main__':
    main()
