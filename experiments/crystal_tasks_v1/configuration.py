"""Render native LLaMA-Factory arguments from the frozen matched protocol."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def protocol():
    return json.loads((ROOT / 'configs/shared.json').read_text())


def task_data(task):
    shared = protocol()
    selected = shared['tasks'][task]
    return {key: selected.get(key, shared[key]) for key in ('dataset_dir', 'manifest_sha256')}


def training_config(task):
    shared = protocol()
    selected = shared['tasks'][task]
    training = shared['training']
    assert training['world_size'] * training['per_device_train_batch_size'] * training['gradient_accumulation_steps'] == training['global_batch_size']
    config = {k: v for k, v in training.items() if k not in ('world_size', 'global_batch_size')}
    config.update(
        model_name_or_path=shared['model_path'], model_revision=shared['model_revision'],
        stage='sft', do_train=True, do_eval=True, finetuning_type='lora',
        dataset_dir=task_data(task)['dataset_dir'], dataset=selected['dataset'], eval_dataset=selected['eval_dataset'],
        template=shared['template'], enable_thinking=False, train_on_prompt=False,
        cutoff_len=selected['cutoff_len'], packing=False, val_size=0.0,
        lora_rank=shared['lora']['rank'], lora_alpha=shared['lora']['alpha'],
        lora_dropout=shared['lora']['dropout'], lora_target=','.join(shared['lora']['target_modules']),
        trainable_token_indices=shared['lora']['trainable_token_indices'],
        add_special_tokens=','.join(shared['tokens']), resize_vocab=True,
        flash_attn='sdpa', use_reentrant_gc=False,
        per_device_eval_batch_size=1, eval_strategy='steps', eval_steps=600,
        save_strategy='steps', save_steps=600, logging_steps=10, report_to='tensorboard',
        load_best_model_at_end=False, metric_for_best_model='eval_loss', greater_is_better=False,
        ddp_find_unused_parameters=False,
        overwrite_output_dir=False, preprocessing_num_workers=4,
        output_dir=f'/home/liugang/src/crystalDLM/experiments/crystal_tasks_v1/runs/{task}',
    )
    return config


if __name__ == '__main__':
    for task in ('ar_understanding', 'ar_design'):
        (ROOT / 'configs' / f'{task}.json').write_text(json.dumps(training_config(task), indent=2) + '\n')
