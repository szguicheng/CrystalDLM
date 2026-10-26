import json
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
import run
train=run._load_training_module()
torch.manual_seed(42)
base=Qwen3ForCausalLM(Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attn_implementation='sdpa'))
model=train.Qwen3ARBaseline(base,60,61,62,63,prediction_offset=0).eval()
with torch.no_grad(): model.time.weight.normal_(std=.1)
ids=[3,4,5]
sampler=run.CachedCausalSampler(model,None)
deltas=[]
with torch.inference_mode():
 cached=sampler.prefill(ids)
 for token in [None,60,8,9,61,10]:
  if token is not None:
   ids.append(token);cached=sampler.append(token)
  expected=model.logits(torch.tensor(ids),'ar',positions=[len(ids)-1])[0].float()
  torch.testing.assert_close(cached,expected,rtol=1e-5,atol=1e-6)
  deltas.append(float((cached-expected).abs().max()))
result=dict(passed=True,contexts=len(deltas),max_abs=max(deltas),dtype='float32',time_embeddings='nonzero_random')
from pathlib import Path
Path('experiments/structure_request_eval_20261017/parity-fp32.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
