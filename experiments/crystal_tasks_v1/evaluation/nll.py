"""Teacher-forced AR NLL for a saved Qwen/PEFT task checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from ..configuration import protocol
from ..tokenization import make_processor, encode_row
from .losses import causal_nll, aggregate_causal_nll


def score_model(model, rows, processor, task, device):
    import torch
    model.eval()
    receipts=[]
    with torch.inference_mode():
        for row in rows:
            item=encode_row(row,processor,task)
            ids=torch.tensor([item['input_ids']],device=device)
            logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits.float()
            receipts.append(causal_nll(logits,torch.tensor([item['labels']],device=device),
                torch.tensor([item['segment_ids']],device=device)))
    return aggregate_causal_nll(receipts),receipts


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from .references import load_task_rows
    from .cli import _git_metadata

    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--task',choices=['ar_understanding','ar_design','hybrid_design'],required=True)
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len([x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()])!=1:
        raise ValueError('Select exactly one free GPU with CUDA_VISIBLE_DEVICES')
    started=datetime.now(timezone.utc).isoformat()
    adapter=json.loads((args.checkpoint/'adapter_config.json').read_text())
    p=protocol()
    if Path(adapter['base_model_name_or_path']).resolve()!=Path(p['model_path']).resolve():
        raise ValueError('Checkpoint base differs from frozen Qwen3-8B')
    task='ar_design' if args.task=='hybrid_design' else args.task
    rows=load_task_rows(args.dataset_dir,'understanding' if task=='ar_understanding' else 'design',args.split)
    processor=make_processor(task)
    base=AutoModelForCausalLM.from_pretrained(p['model_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa')
    model=PeftModel.from_pretrained(base,args.checkpoint,is_trainable=False).to('cuda:0')
    summary,receipts=score_model(model,rows,processor,task,'cuda:0')
    summary['interpretation']='AR branch token NLL; not diffusion likelihood'
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (args.output/'receipts.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in receipts))
    source_root=Path(__file__).resolve().parents[1]
    code_paths=[Path(__file__),Path(__file__).with_name('losses.py'),Path(__file__).with_name('references.py'),
        source_root/'tokenization.py',source_root/'configuration.py',source_root/'configs/shared.json']
    meta={'command':sys.argv,'configuration':vars(args),'git':_git_metadata(),
        'base_model':p['model_name'],'base_revision':p['model_revision'],
        'manifest_sha256':hashlib.sha256((args.dataset_dir/'manifest.json').read_bytes()).hexdigest(),
        'seed':p['training']['seed'],'started_at':started,'finished_at':datetime.now(timezone.utc).isoformat(),
        'checkpoint':str(args.checkpoint),'environment':{x:importlib.metadata.version(x) for x in ['torch','transformers','peft','llamafactory']},
        'source_sha256':{str(path.relative_to(source_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths}}
    (args.output/'run_meta.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')


if __name__=='__main__':
    main()
