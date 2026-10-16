"""Learn semantic prompt/history invariance without inference-time answer repair."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil

import torch
from transformers import AutoTokenizer
from model import Hybrid
from adapt import TRAIN_OFFSETS, TEST_OFFSETS, train_phase
from prepare import END, EOS, ROOT, START, edit_factor, encoded_length, sha, tasks, write_json
from rescore import reading_grade, score_branch
from run import attempt, evaluate, generate, records
from score import answer, audit, edit, parse, grade_edit

READ_TRAIN=(
 'Read the current crystal and give its composition, space group number, atom count, independent orbit count, and lattice lengths a, b, c.',
 'Describe the structure that is actually shown. Report its formula, space group number, number of cell atoms and independent orbits, and a, b, c.',
 'Use the current crystal block as the source of truth even when earlier design conditions disagree. Give the formula, space group number, atoms, orbits, and all three lattice lengths.',
 'Check the supplied crystal rather than the requested design. State its composition, space group number, cell atom count, orbit count, and a, b, c in angstroms.',
 'What does the displayed structure contain? Give composition, space group number, total atoms, independent orbits, and lattice lengths a, b and c.',
 'Summarize the latest crystal using its own data: composition, space group number, atoms per cell, orbit count and a, b, c.',
 'Earlier requested properties may differ from this crystal. Extract the actual composition and space group number, atom and orbit counts, and lattice lengths a, b, c.',
 'List the factual cell description from the current structure: formula, space group number, atoms, independent orbits and lattice lengths a, b, c.',
 'Tell me the actual formula and space group number. Also report the total number of atoms, number of independent orbits, and a, b, c for this crystal.',
 'Please inspect the provided crystal itself and return its composition, space group number, atom count, independent orbit count and a, b, c.',
 'Extract seven facts from the latest block: composition, space group number, atom count, independent orbit count, a, b and c.',
 'Report the measured cell description in the crystal block, not a repetition of my earlier conditions: formula, space group number, atoms, orbits, a, b, c.',
)
READ_TEST=(
 'For the structure currently in front of you, what are its chemical composition, numerical space group, cell population, independent orbit population, and the three cell lengths a, b, c?',
 'Please audit this candidate. I need the actual formula, space group number, counts of atoms and independent orbits, and cell lengths a, b, c.',
 'Forget the intended design for this answer and identify what the latest crystal encodes: composition, space group number, cell atoms, independent orbits, a, b, c.',
 'Give a factual account of the latest candidate, including its composition, space group number, total atom count, distinct orbit count and each lattice length a, b, c.',
)


def read_example(old,text,question):
    prefix=tasks(old,'train','plus')[0]['prefix']+' I generated a crystal.\n'+START+text+END+'\nUser: '+question+'\nAssistant:'
    return dict(task='Q_varied',prefix=prefix,target=' '+answer(text)+EOS)


def caption(text,index):
    p=parse(text);a,b,c=p['lengths']
    if index%3==0:return ' '+answer(text)+EOS
    if index%3==1:
        return (f" Space group: {p['sg']}. Formula: {p['formula']}. {p['atoms']} atoms and {p['orbits']} independent orbits. "
                f'a={a:.2f}, b={b:.2f}, c={c:.2f} angstroms.'+EOS)
    return (f" The composition is {p['formula']}; the space group is {p['sg']}. "
            f"There are {p['atoms']} atoms in {p['orbits']} orbits. The a, b, c are {a:.2f}, {b:.2f}, {c:.2f} angstroms."+EOS)


def edit_question(direction,index):
    factor=edit_factor(direction);percent='increase' if direction=='plus' else 'decrease'
    forms=(f'Multiply all lattice lengths by {factor}; keep every other field unchanged.',
           f'Apply a factor of {factor} to a, b and c, preserving angles and all orbit records.',
           f'{percent.capitalize()} all cell lengths by five percent, with all remaining structure fields fixed.',
           f'Scale every lattice length by {factor if direction=="plus" else ".95"}, keeping the angles, elements, Wyckoff sites and fractional coordinates unchanged.',
           f'Make each cell length {factor} times its present value. Do not change any other structural value.',
           f'Perform a uniform five-percent {"expansion" if direction=="plus" else "contraction"} of the cell lengths; preserve the rest of the structure.')
    return forms[index%len(forms)]


def augmented_edit(row,direction,cycle):
    source=[row['c1'],row['plus'],row['minus']][cycle%3]
    ex=tasks(row,'train',direction,cycle%2)
    lead=[' I will generate the requested crystal.\n',' Here is the proposed crystal.\n',' Candidate crystal:\n'][cycle%3]
    history=ex[0]['prefix']+lead+START+source+END+'\nUser: '+READ_TRAIN[cycle%len(READ_TRAIN)]+'\nAssistant:'+caption(source,cycle)
    prefix=history+'\nUser: '+edit_question(direction,cycle)+'\nAssistant:'
    return dict(task='E_varied',prefix=prefix,lead=ex[2]['lead'],crystal=edit(source,edit_factor(direction)))


def aligned_stream(rows,seed):
    rng=random.Random(seed);epoch=0
    while True:
        indices=list(range(len(rows)));rng.shuffle(indices)
        for index in indices:
            row=rows[index];direction='plus' if (index+epoch)%2 else 'minus'
            yield from tasks(row,'train',direction,epoch%2)
            for j in range(2):
                current=rows[(index+TRAIN_OFFSETS[(epoch+j)%3])%len(rows)]
                text=current[['c1','plus','minus'][(epoch+j)//3%3]]
                yield read_example(row,text,READ_TRAIN[(epoch*5+index+j)%len(READ_TRAIN)])
            yield augmented_edit(row,direction,epoch+index)
            yield read_example(row,row[['c1','plus','minus'][(epoch+index)%3]],READ_TRAIN[(epoch*7+index)%len(READ_TRAIN)])
        epoch+=1


def novel_reading(model,tok,mode,rows,out):
    cases=[]
    for index,old in enumerate(rows):
        for offset in TEST_OFFSETS:
            current=rows[(index+offset)%8]
            for question in READ_TEST:
                e=read_example(old,current['c1'],question)
                response=attempt(lambda:model.text(e['prefix'],tok,mode))
                scored=reading_grade(response['value'],current['c1']) if response['ok'] else dict(success=False,fields={})
                cases.append(dict(old_id=old['id'],current_id=current['id'],question=question,response=response,grade=scored))
        print(json.dumps(dict(task='novel_reading',mode=mode,old_materials_done=index+1,total=8)),flush=True)
    assert len(cases)==128
    with (out/'novel_reading.jsonl').open('w') as f:
        for c in cases:f.write(json.dumps(c)+'\n')
    fields=Counter()
    for c in cases:fields.update({k:int(v) for k,v in c['grade']['fields'].items()})
    result=dict(n=len(cases),success=sum(c['grade']['success'] for c in cases),correct_by_field=dict(fields))
    write_json(out/'novel_reading_summary.json',result)
    return result


def shorthand_editing(model,tok,mode,rows,out):
    results=[]
    for row in rows:
        e=tasks(row,'train','minus')[2]
        prefix=e['prefix'].replace('by 0.95,','by .95,')
        assert prefix!=e['prefix']
        rng=torch.Generator(device='cuda').manual_seed(int(sha((row['id']+'shorthand').encode())[:8],16))
        response=attempt(lambda:generate(model,tok,mode,prefix,rng))
        scored=grade_edit(response['value']['crystal'],row['c1'],'.95') if response['ok'] else dict(success=False)
        results.append(dict(id=row['id'],response=response,grade=scored))
    write_json(out/'shorthand_editing.json',dict(n=8,success=sum(r['grade']['success'] for r in results),records=results))
    return dict(n=8,success=sum(r['grade']['success'] for r in results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    rows=records(args.data,'train');tok=AutoTokenizer.from_pretrained(args.data/'tokenizer')
    tests=[json.loads(l) for l in (args.run/'conflict_test.jsonl').read_text().splitlines()]
    assert set(READ_TRAIN).isdisjoint(READ_TEST)
    for e in [next(aligned_stream(rows,17))]:assert encoded_length(e,tok)<=1536
    with (args.out/'training_exposure.jsonl').open('w') as f:
        stream=aligned_stream(rows,17)
        for _ in range(600*16):
            e=next(stream);assert encoded_length(e,tok)<=1536
            f.write(json.dumps(e)+'\n')
    write_json(args.out/'data_manifest.json',dict(source_rows_sha256=sha((args.data/'train.jsonl').read_bytes()),
         train_questions=READ_TRAIN,test_questions=READ_TEST,exposure_file_sha256=sha((args.out/'training_exposure.jsonl').read_bytes()),
         updates=600,effective_batch=16,inference_answer_repair=False))
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for p in ROOT.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    results={}
    for mode in ['ar','hybrid']:
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        model=Hybrid.pretrained(tok)
        checkpoint=args.run/mode/'feedback/checkpoint.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.restore_checkpoint(saved,expected_mode=mode);del saved;model=model.cuda()
        folder=args.out/mode
        model,receipt=train_phase(model,tok,mode,rows,folder,'align',True,17,
                     sequence=aligned_stream(rows,17),updates=600)
        original=evaluate(model,tok,mode,rows,folder,'train')
        from adapt import pair_evaluation
        conflicts=pair_evaluation(model,tok,mode,tests,folder)
        # Uniform v2 reader; all raw legacy metrics retained.
        mats={r['id']:r for r in rows};corrected=score_branch(folder,mats,tests)
        novel=novel_reading(model,tok,mode,rows,folder)
        shorthand=shorthand_editing(model,tok,mode,rows,folder)
        receipt.update(status='evaluated',original=original,corrected=corrected,novel=novel,shorthand=shorthand)
        write_json(folder/'status.json',receipt)
        results[mode]=dict(corrected=corrected,novel=novel,shorthand=shorthand)
        write_json(args.out/'summary.json',dict(status='partial',results=results))
        del model;torch.cuda.empty_cache()
    write_json(args.out/'summary.json',dict(status='complete',results=results,
         note='Learned prompt/history augmentation; no inference-time fact replacement.8 seen materials only.'))
