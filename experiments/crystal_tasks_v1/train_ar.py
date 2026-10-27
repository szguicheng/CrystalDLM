"""Run the approved AR task with LLaMA-Factory's model, data, and trainer."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

from .configuration import protocol, training_config, task_data
from .evaluation.cli import _git_metadata


def overfit_indices(count, seed=42):
    return sorted(random.Random(seed).sample(range(count), 32))


def design_overfit_indices(rows, seed=42):
    order=list(range(len(rows)))
    random.Random(seed).shuffle(order)
    selected=[]; seen=set()
    for index in order:
        request=(rows[index]['instruction'],rows[index]['input'])
        if request not in seen:
            selected.append(index);seen.add(request)
        if len(selected)==32:
            return sorted(selected)
    raise ValueError('Design overfit check requires32distincttrainingrequests')


def run_config(task, mode, run_dir, overfit_epochs=60):
    config = training_config(task)
    config.update(output_dir=str(run_dir), logging_dir=str(run_dir/'tensorboard'),
        metric_for_best_model='eval_token_nll', prediction_loss_only=True,
        plot_loss=True, save_safetensors=True)
    if mode=='overfit':
        config.update(num_train_epochs=overfit_epochs,eval_steps=30,save_steps=30,logging_steps=3,save_total_limit=3)
    return config


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    import torch
    import torch.distributed as dist
    from transformers import set_seed
    from llamafactory.hparams import get_train_args
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
    from llamafactory.model import load_model, load_tokenizer
    from llamafactory.extras.ploting import plot_loss
    from .training_metrics import CrystalTrainer, EpochCheckpointCallback

    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['ar_understanding','ar_design'],default='ar_understanding')
    parser.add_argument('--mode',choices=['overfit','full'],required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--overfit-epochs',type=int,default=60)
    args=parser.parse_args()
    p=protocol();world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    visible=[x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if world!=3 or len(visible)!=3:
        raise ValueError('Approved training requires three ranks and three explicitly selected GPUs')
    if not p['tasks'][args.task]['ready']:
        raise RuntimeError('Full task data/token audit is not ready')
    run_dir=args.run_dir.resolve()
    if rank==0:
        run_dir.mkdir(parents=True,exist_ok=False)
    config=run_config(args.task,args.mode,run_dir,args.overfit_epochs)
    model_args,data_args,training_args,finetuning_args,generating_args=get_train_args(config)
    if dist.is_initialized():dist.barrier()
    set_seed(p['training']['seed'])
    data=task_data(args.task)
    manifest=Path(data['dataset_dir'])/'manifest.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=data['manifest_sha256']:
        raise ValueError('Frozen task manifest changed')
    from .evaluation.references import load_task_rows
    task_name='design' if args.task=='ar_design' else 'understanding'
    verified_train=load_task_rows(data['dataset_dir'],task_name,'training')
    load_task_rows(data['dataset_dir'],task_name,'validation')
    tokenizer_module=load_tokenizer(model_args);tokenizer=tokenizer_module['tokenizer']
    template=get_template_and_fix_tokenizer(tokenizer,data_args)
    datasets=get_dataset(template,model_args,data_args,training_args,stage='sft',**tokenizer_module)
    assert len(datasets['train_dataset'])==36362 and len(datasets['eval_dataset'])==4501
    metadata=[json.loads(line) for line in (Path(data['dataset_dir'])/'metadata/training.jsonl').read_text().splitlines()]
    selected=overfit_indices(len(metadata)) if args.mode=='overfit' else list(range(len(metadata)))
    if args.mode=='overfit' and args.task=='ar_design':
        selected=design_overfit_indices(verified_train)
    del verified_train
    if args.mode=='overfit':
        datasets['train_dataset']=datasets['train_dataset'].select(selected)
        datasets['eval_dataset']=datasets['train_dataset']
    set_seed(p['training']['seed'])
    model=load_model(tokenizer,model_args,finetuning_args,is_trainable=True)
    assert model_args.adapter_name_or_path is None
    trainable=[(name,param) for name,param in model.named_parameters() if param.requires_grad]
    assert trainable and all('lora_' in name or 'trainable_tokens_delta' in name for name,_ in trainable)
    digest=hashlib.sha256()
    for name,param in trainable:
        digest.update(name.encode());digest.update(param.detach().float().cpu().numpy().tobytes())
    initial_hash=digest.hexdigest()
    if dist.is_initialized():
        hashes=[None]*world;dist.all_gather_object(hashes,initial_hash);assert len(set(hashes))==1
    meta={'status':'running','command':sys.argv,'configuration':config,'git':_git_metadata(),
        'base_model':p['model_name'],'base_revision':p['model_revision'],'dataset_dir':data['dataset_dir'],'manifest_sha256':data['manifest_sha256'],
        'seed':42,'gpu_ids':visible,'world_size':world,'global_batch_size':12,
        'training_rows':len(datasets['train_dataset']),'validation_rows':len(datasets['eval_dataset']),
        'validation_scope':'same32trainingdiagnostic' if args.mode=='overfit' else 'fullfrozenvalidation',
        'selected_training_indices':selected if args.mode=='overfit' else None,
        'selected_training_ids':[metadata[i]['material_id'] for i in selected] if args.mode=='overfit' else None,
        'selection_policy':'seed42distinctrequests' if args.mode=='overfit' and args.task=='ar_design' else 'seed42random32' if args.mode=='overfit' else 'fullfrozentraining',
        'initial_trainable_sha256':initial_hash,'trainable_parameters':sum(x.numel() for _,x in trainable),
        'environment':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft','accelerate','llamafactory']},
        'started_at':_now(),'checkpoint':None}
    if rank==0:
        _write(run_dir/'run_meta.json',meta);_write(run_dir/'config.json',config)
    collator=SFTDataCollatorWith4DAttentionMask(template=template,model=model,pad_to_multiple_of=8,
        label_pad_token_id=-100,block_diag_attn=False,attn_implementation='sdpa',
        compute_dtype=model_args.compute_dtype,**tokenizer_module)
    trainer=CrystalTrainer(model=model,args=training_args,finetuning_args=finetuning_args,
        data_collator=collator,callbacks=[EpochCheckpointCallback()] if args.mode=='full' else [],crystal_task=args.task,
        **datasets,**tokenizer_module)
    try:
        result=trainer.train(resume_from_checkpoint=None)
        trainer.save_model(str(run_dir/'final'));trainer.save_state()
        metrics=trainer.evaluate()
        if trainer.is_world_process_zero():
            meta.update(status='complete',ended_at=_now(),checkpoint=str(run_dir/'final'),
                best_checkpoint=trainer.state.best_model_checkpoint,best_metric=trainer.state.best_metric,
                global_step=trainer.state.global_step,epochs=trainer.state.epoch,
                training_metrics=result.metrics,validation_metrics=metrics)
            _write(run_dir/'run_meta.json',meta)
            segment_keys=['train_answer_ce','eval_answer_ce'] if args.task=='ar_understanding' else [
                'train_description_ce','eval_description_ce','train_crystal_ce','eval_crystal_ce']
            plot_loss(str(run_dir),keys=['loss','eval_loss','eval_token_nll',*segment_keys])
    except BaseException as error:
        if trainer.is_world_process_zero():
            meta.update(status='failed',ended_at=_now(),error=f'{type(error).__name__}: {error}')
            _write(run_dir/'run_meta.json',meta)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    main()
