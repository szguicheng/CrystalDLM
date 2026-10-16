"""Same-batch generation/edit/reading learning across the full noise schedule."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

import torch
from transformers import AutoTokenizer
from adapt import train_phase
from cohort import unseen_reading
from model import Hybrid
from prepare import ROOT, encoded_length, sha, write_json
from recover import evaluation, recovery_stream
from run import records

UPDATES=4096
TASK_ORDER=[0]*8+[2,6]*2+[1,3,4,5]


def joint_stream(rows,seed):
    source=recovery_stream(rows,seed)
    while True:
        group=[next(source) for _ in range(8)]
        for index in TASK_ORDER:
            yield dict(group[index])


def freeze_exposure(rows,tok,out):
    counts=Counter();materials=Counter();styles=Counter();maximum=0
    source=joint_stream(rows,17);file=out/'training_exposure.jsonl'
    with file.open('w') as f:
        for _ in range(UPDATES*16):
            e=next(source);maximum=max(maximum,encoded_length(e,tok));assert maximum<=1536
            if 'crystal' in e:assert len(tok.encode(e['crystal'],add_special_tokens=False))+1<=256
            counts[e['task']]+=1;materials[e['source_id']]+=1;styles[e['generation_style']]+=1
            f.write(json.dumps(e)+'\n')
    assert counts['G']==32768 and counts['E']+counts['E_varied']==16384
    assert sum(counts.values())==65536 and len(materials)==1024 and set(materials.values())=={64}
    write_json(out/'exposure_audit.json',dict(tasks=dict(counts),styles=dict(styles),
        conditioning_materials=len(materials),examples_per_conditioning_material=64,
        G_per_material=32,E_per_material=16,Q_per_material=16,max_length=maximum,sha256=sha(file.read_bytes())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    assert json.loads((args.initial/'summary.json').read_text())['status']=='complete'
    torch.set_num_threads(4);args.out.mkdir(parents=True,exist_ok=False)
    train=records(args.data,'train');val=sorted(records(args.data,'val'),key=lambda r:sha(r['id'].encode()))
    assert len(train)==1024 and len(val)==128 and not {r['id'] for r in train}&{r['id'] for r in val}
    old=records(ROOT/'smoke_data_002','train');materials={r['id']:r for r in train+val}
    tok=AutoTokenizer.from_pretrained(args.data/'tokenizer');freeze_exposure(train,tok,args.out)
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    write_json(args.out/'manifest.json',dict(train_ids=[r['id'] for r in train],validation_ids=[r['id'] for r in val],
        validation_groups={'original64':[r['id'] for r in val[:64]],'later64':[r['id'] for r in val[64:]]},
        updates=UPDATES,optimizer_batch_mix={'G':8,'E':4,'Q':4},
        source_manifest_sha256=sha((args.data/'manifest.json').read_bytes()),initial=str(args.initial),
        physical_audit='orbit_membership_v3',reading_audit='named_axis_v2',test_used=False,answer_repair=False))
    results={}
    for mode in ['hybrid','ar']:
        torch.manual_seed(17);torch.cuda.manual_seed_all(17);model=Hybrid.pretrained(tok)
        saved=torch.load(args.initial/mode/'joint/checkpoint.pt',map_location='cpu',weights_only=True)
        assert saved['config']['source_sha256']['model.py']==sha((ROOT/'model.py').read_bytes())
        model.restore_checkpoint(saved,expected_mode=mode);previous=saved['config']['final_fingerprint'];del saved;model=model.cuda()
        folder=args.out/mode;torch.cuda.reset_peak_memory_stats()
        model,receipt=train_phase(model,tok,mode,train,folder,'joint',True,17,warm_fingerprint=previous,
            sequence=joint_stream(train,17),updates=UPDATES)
        receipt['peak_gpu_MB']=torch.cuda.max_memory_allocated()/2**20
        core=evaluation(model,tok,mode,val,folder/'validation_128',materials)
        reading=unseen_reading(model,tok,mode,val,folder/'validation_128')
        regression=evaluation(model,tok,mode,old,folder/'old8_regression',materials,'train')
        raw=[json.loads(l) for l in (folder/'validation_128/rescored_evaluation_v2.jsonl').read_text().splitlines()]
        keys=['G','Q_reference','E_reference','closed_loop','Q_swap','Q_no_crystal']
        strata={name:dict(n=64,counts={k:sum(r[k] for r in raw if r['id'] in ids) for k in keys})
            for name,ids in [('original64',{r['id'] for r in val[:64]}),('later64',{r['id'] for r in val[64:]})]}
        write_json(folder/'validation_128/stratified_summary.json',strata)
        receipt.update(status='validated',validation=core,reading=reading,old8_regression=regression,strata=strata)
        write_json(folder/'status.json',receipt);results[mode]=dict(validation=core,reading=reading,old8=regression,strata=strata)
        write_json(args.out/'summary.json',dict(status='partial',results=results))
        del model;torch.cuda.empty_cache()
    write_json(args.out/'summary.json',dict(status='complete',results=results,
        note='128 reused validation materials. Final test256 untouched. No automatic claim of goal completion.'))


if __name__=='__main__':main()
