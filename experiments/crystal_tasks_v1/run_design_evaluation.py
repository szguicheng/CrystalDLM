"""Run matched NLL, fixed-budget sampling, GRACE, and shared design metrics."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from .configuration import protocol, task_data
from .evaluation.references import read_jsonl, load_task_rows
from .evaluation.plan import make_plan
from .evaluation.cli import _join_attempt_plan, _join_property_receipts, _git_metadata, _sha256

ROOT=Path(__file__).resolve().parents[2]
PY=ROOT/'experiments/crystal_tasks_v1/python.sh'


def merge_by_id(plan, shards):
    rows=[row for shard in shards for row in shard]
    indexed={row['attempt_id']:row for row in rows}
    expected=[row['attempt_id'] for row in plan]
    if len(indexed)!=len(rows) or len(set(expected))!=len(expected) or set(indexed)!=set(expected):
        raise ValueError('Shards do not cover the attempt plan exactly once')
    return [indexed[key] for key in expected]


def require_full_plan(plan, rows, seeds, attempts_per_condition):
    if plan!=make_plan(rows,seeds,attempts_per_condition):
        raise ValueError('Attempt plan differs from the full frozen test/seed/budget protocol')


def _write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _jsonl(path,rows):
    with path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')


def _jobs(specs,gpus,output):
    used={gpu for _,_,gpu,_ in specs}
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={str(int(line.split(',')[0])):int(line.split(',')[1]) for line in status.splitlines()}
    if any(memory[gpu]>=512 for gpu in used):raise RuntimeError('A selected evaluation GPU is occupied')
    processes=[]
    for name,module,gpu,arguments in specs:
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
        if module.endswith('.grace'):env['CRYSTAL_ENV_PREFIX']='/home/liugang/src/crystalDLM/envs/grace'
        log=(output/f'{name}.log').open('x')
        proc=subprocess.Popen(['bash',str(PY),'-m',module,*map(str,arguments)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append((name,proc,log))
    results={}
    for name,proc,log in processes:results[name]=proc.wait();log.close()
    _write(output/(specs[0][0].split('-')[0]+'-exits.json'),results)
    if any(results.values()):raise RuntimeError(f'Evaluation worker failed: {results}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--attempt-plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='0,1,2')
    args=parser.parse_args();gpus=args.gpus.split(',')
    if len(gpus)!=3 or len(set(gpus))!=3:raise ValueError('Three distinct GPU indices required')
    out=args.output.resolve()
    data=task_data('ar_design');p=protocol();plan=read_jsonl(args.attempt_plan)
    assert _sha256(Path(data['dataset_dir'])/'manifest.json')==data['manifest_sha256']
    require_full_plan(plan,load_task_rows(data['dataset_dir'],'design','test'),
        p['evaluation']['seeds'],p['evaluation']['attempts_per_condition_per_seed'])
    out.mkdir(parents=True,exist_ok=False)
    meta={'status':'running','command':sys.argv,'git':_git_metadata(),'base_model':p['model_name'],
        'checkpoint':str(args.checkpoint.resolve()),'dataset':data,'plan_sha256':_sha256(args.attempt_plan),
        'attempts':len(plan),'seeds':sorted({r['seed'] for r in plan}),
        'started_at':datetime.now(timezone.utc).isoformat()}
    _write(out/'run_meta.json',meta)
    common=['--checkpoint',args.checkpoint,'--dataset-dir',data['dataset_dir']]
    try:
        (out/'stage.txt').write_text('nll\n')
        _jobs([(f'nll-{split}','experiments.crystal_tasks_v1.evaluation.nll',gpu,
            [*common,'--task','ar_design','--split',split,'--output',out/f'nll_{split}'])
            for split,gpu in zip(['validation','test'],gpus)],gpus,out)
        (out/'stage.txt').write_text('generation\n')
        _jobs([(f'generation-{rank}','experiments.crystal_tasks_v1.generate_design',gpu,
            ['--checkpoint',args.checkpoint,'--mode','sample','--attempt-plan',args.attempt_plan,
             '--rank',rank,'--world-size',3,'--output',out/f'generation_{rank}'])
            for rank,gpu in enumerate(gpus)],gpus,out)
        attempts=merge_by_id(plan,[read_jsonl(out/f'generation_{rank}/attempts.jsonl') for rank in range(3)])
        attempts=_join_attempt_plan(attempts,plan);_jsonl(out/'attempts.jsonl',attempts)
        (out/'stage.txt').write_text('grace\n')
        _jobs([(f'grace-{rank}','experiments.crystal_tasks_v1.evaluation.grace',gpu,
            [*common,'--model',p['model_name'],'--attempts',out/f'generation_{rank}/attempts.jsonl',
             '--output',out/f'grace_{rank}']) for rank,gpu in enumerate(gpus)],gpus,out)
        properties=merge_by_id(plan,[read_jsonl(out/f'grace_{rank}/receipts.jsonl') for rank in range(3)])
        _join_property_receipts(attempts,properties);_jsonl(out/'property-receipts.jsonl',properties)
        (out/'stage.txt').write_text('shared_metrics\n')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        with (out/'metrics.log').open('x') as log:
            subprocess.run(['bash',str(PY),'-m','experiments.crystal_tasks_v1.evaluation.cli','design',
                *map(str,common),'--model',p['model_name'],'--attempt-plan',str(args.attempt_plan),
                '--attempts',str(out/'attempts.jsonl'),'--property-receipts',str(out/'property-receipts.jsonl'),
                '--output',str(out/'metrics')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((out/'metrics/summary.json').read_text())
        if summary['summary']['property_evaluation']['status']!='complete':
            raise RuntimeError('Property evaluation remains incomplete; inspect scoring errors')
        meta.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
        (out/'stage.txt').write_text('complete\n')
    except BaseException as error:
        meta.update(status='failed',finished_at=datetime.now(timezone.utc).isoformat(),error=f'{type(error).__name__}: {error}')
        raise
    finally:_write(out/'run_meta.json',meta)


if __name__=='__main__':main()
