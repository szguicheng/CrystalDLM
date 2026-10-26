from pathlib import Path
import json,time,torch
from transformers import set_seed
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model
from experiments.crystal_tasks_v1.configuration import training_config
from experiments.crystal_tasks_v1.tokenization import make_processor,encode_row
started=time.monotonic();cfg=training_config('ar_understanding');cfg.update(report_to=[],output_dir='/home/liugang/src/crystalDLM/experiments/crystal_tasks_v1/phase3_preflight')
model_args,_,_,finetuning_args,_=get_train_args(cfg)
processor=make_processor('ar_understanding');tok=processor.tokenizer
set_seed(42);model=load_model(tok,model_args,finetuning_args,is_trainable=True).to('cuda:0')
row=json.loads(Path('datasets/crystal_tasks_v1/understanding/training.jsonl').open().readline())
item=encode_row(row,processor,'ar_understanding')
batch={k:torch.tensor([item[k]],device='cuda:0') for k in ['input_ids','labels','attention_mask']}
model.train()
with torch.autocast('cuda',dtype=torch.bfloat16): loss=model(**batch).loss
assert torch.isfinite(loss)
loss.backward()
trainable=[(n,x) for n,x in model.named_parameters() if x.requires_grad]
assert all('lora_' in n or 'trainable_tokens_delta' in n for n,x in trainable)
assert all(x.grad is None or torch.isfinite(x.grad).all() for n,x in trainable)
assert any('trainable_tokens_delta' in n and x.grad is not None and x.grad.abs().sum()>0 for n,x in trainable)
result={'status':'passed','scope':'single-GPU original8B forward/backward only;no optimizer step','model_name':'Qwen/Qwen3-8B','tokens':len(item['input_ids']),'loss':float(loss.detach()),'trainable_parameters':sum(x.numel() for _,x in trainable),'gpu':0,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'seconds':time.monotonic()-started}
Path('experiments/crystal_tasks_v1/phase3_preflight/check8b.json').write_text(json.dumps(result,indent=2)+'\n');print('CHECK8B',json.dumps(result),flush=True)
