import argparse,json
from pathlib import Path
import torch
import run
args=argparse.Namespace(model_path=run.DEFAULT_MODEL,data=run.DEFAULT_DATA,checkpoint=run.SHARED_RUN/"checkpoint.pt")
train,model,tok,identity,state,software=run._load_model(args)
ids=tok.encode(run.REQUESTS[0]["prompt"],add_special_tokens=False)
sampler=run.CachedCausalSampler(model,tok)
checks=[]
with torch.inference_mode():
 cached=sampler.prefill(ids)
 for label,token in [("prefix",None),("start",model.start_id),("crystal",0),("end",model.end_id),("tail",0)]:
  if token is not None:
   ids.append(token);cached=sampler.append(token)
  with torch.autocast("cuda",dtype=torch.bfloat16):
   expected=model.logits(torch.tensor(ids,device=model.time.weight.device),"ar",positions=[len(ids)-1])[0].float()
  lp=expected.log_softmax(-1);lq=cached.log_softmax(-1)
  r=dict(context=label,max_abs=float((cached-expected).abs().max()),mean_abs=float((cached-expected).abs().mean()),kl=float((lp.exp()*(lp-lq)).sum()),argmax_same=bool(cached.argmax()==expected.argmax()),expected_token=int(expected.argmax()),cached_token=int(cached.argmax()))
  checks.append(r);print(json.dumps(r),flush=True)
Path("experiments/structure_request_eval_20261017/parity-diagnostic.json").write_text(json.dumps(checks,indent=2))
