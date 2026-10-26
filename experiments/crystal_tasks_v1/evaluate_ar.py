"""Greedy Understanding readout and exact answer NLL from a saved adapter."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

from .configuration import protocol
from .tokenization import make_processor,encode_row
from .evaluation.references import load_split_rows
from .evaluation.understanding import summarize_understanding
from .evaluation.nll import score_model
from .evaluation.cli import _git_metadata,_sha256,_environment
from .train_ar import overfit_indices


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--split',choices=['training','validation','test'],required=True)
    parser.add_argument('--overfit32',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if len([x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x])!=1:
        raise ValueError('Select exactly one free GPU')
    if args.output.exists():raise FileExistsError(args.output)
    if args.overfit32 and args.split!='training':raise ValueError('Overfit check uses only training rows')
    p=protocol();started=datetime.now(timezone.utc).isoformat()
    rows=load_split_rows(Path(p['dataset_dir']),args.split)
    if args.overfit32:rows=[rows[i] for i in overfit_indices(len(rows))]
    adapter=json.loads((args.checkpoint/'adapter_config.json').read_text())
    assert Path(adapter['base_model_name_or_path']).resolve()==Path(p['model_path']).resolve()
    processor=make_processor('ar_understanding');tok=processor.tokenizer;tok.padding_side='left'
    base=AutoModelForCausalLM.from_pretrained(p['model_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,args.checkpoint,is_trainable=False).to('cuda:0').eval()
    args.output.mkdir(parents=True)
    meta={'status':'running','command':sys.argv,'configuration':vars(args),'git':_git_metadata(),
        'base_model':p['model_name'],'base_revision':p['model_revision'],'checkpoint':str(args.checkpoint),
        'dataset_manifest_sha256':_sha256(Path(p['dataset_dir'])/'manifest.json'),'seed':42,
        'environment':_environment(),'started_at':started,'rows':len(rows),
        'source_sha256':{str(x):_sha256(x) for x in [Path(__file__),Path(__file__).with_name('tokenization.py')]}}
    (args.output/'run_meta.json').write_text(json.dumps(meta,default=str,indent=2)+'\n')
    predictions=[]
    with torch.inference_mode(),(args.output/'predictions.jsonl').open('x') as stream:
        for offset in range(0,len(rows),args.batch_size):
            batch_rows=rows[offset:offset+args.batch_size];prompts=[]
            for row in batch_rows:
                encoded=encode_row(row,processor,'ar_understanding')
                prompt_len=sum(v==-100 for v in encoded['labels'])
                prompts.append(encoded['input_ids'][:prompt_len])
            batch=tok.pad({'input_ids':prompts,'attention_mask':[[1]*len(x) for x in prompts]},padding=True,return_tensors='pt').to('cuda:0')
            generated=model.generate(**batch,do_sample=False,temperature=None,top_p=None,top_k=None,
                max_new_tokens=128,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
            texts=tok.batch_decode(generated[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)
            for row,text in zip(batch_rows,texts,strict=True):
                result={'material_id':row['metadata']['material_id'],'seed':42,'response':text.strip(),
                    'reference_output':row['output'],'exact_match':text.strip()==row['output'].strip()}
                predictions.append(result);stream.write(json.dumps(result)+'\n');stream.flush()
            print('GENERATED',len(predictions),'/',len(rows),flush=True)
    joined=[]
    for row,pred in zip(rows,predictions,strict=True):
        f=row['metadata']['core_facts'];joined.append({'response':pred['response'],'reference':{'formula':f['cell_formula'],
            'space_group':f['space_group_number'],'cell_atom_count':f['cell_atom_count'],'independent_orbit_count':f['independent_orbit_count']}})
    summary=summarize_understanding(joined);summary.pop('grades')
    summary['exact_match']={'k':sum(x['exact_match'] for x in predictions),'n':len(rows)}
    nll,receipts=score_model(model,rows,processor,'ar_understanding','cuda:0');summary['nll']=nll
    summary['overfit_passed']=bool(args.overfit32 and summary['all_four']['k']==32 and summary['exact_match']['k']==32 and nll['nll_mean']<=.05)
    (args.output/'nll-receipts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in receipts))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    meta.update(status='complete',ended_at=datetime.now(timezone.utc).isoformat())
    (args.output/'run_meta.json').write_text(json.dumps(meta,default=str,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if args.overfit32 and not summary['overfit_passed']:raise SystemExit('Overfit gate did not pass; full training prohibited')


if __name__=='__main__':main()
