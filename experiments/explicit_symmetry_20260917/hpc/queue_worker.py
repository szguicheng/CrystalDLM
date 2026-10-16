"""Run trusted experiment scripts sequentially inside one Slurm allocation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    assert os.environ.get('SLURM_JOB_ID')
    repo = Path.cwd()
    allocation = Path(sys.argv[1])
    allocation.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ['pending', 'running', 'finished']:
        (allocation / name).mkdir(exist_ok=True, mode=0o700)
    state = {'job_id': os.environ['SLURM_JOB_ID'], 'host': socket.gethostname(),
             'allocation_dir': str(allocation), 'tmux_socket': f"crystaldlm-{os.environ['SLURM_JOB_ID']}",
             'tmux_session': 'work', 'cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
             'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
             'started_at': now(), 'state': 'idle', 'idle_timeout_seconds': 1800}
    write_json(allocation / 'state.json', state)
    write_json(repo / 'hpc/active_allocation.json', state)
    last_activity = time.monotonic()
    while not (allocation / 'STOP').exists():
        pending = sorted((allocation / 'pending').glob('*.sh'))
        if not pending:
            if time.monotonic() - last_activity > state['idle_timeout_seconds']:
                state['state'] = 'idle_timeout'
                break
            time.sleep(2)
            continue
        task = pending[0]
        run = allocation / 'running' / task.stem
        run.mkdir()
        script = run / 'command.sh'
        task.rename(script)
        record = {'run_id': run.name, 'job_id': state['job_id'], 'host': state['host'],
                  'started_at': now(), 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
                  'state': 'running'}
        state.update(state='running', current_run=run.name, updated_at=now())
        write_json(run / 'receipt.json', record)
        write_json(allocation / 'state.json', state)
        write_json(repo / 'hpc/active_allocation.json', state)
        began = time.monotonic()
        environment = {**os.environ, 'CRYSTALDLM_RUN_ID': run.name}
        with (run / 'stdout.log').open('w') as stdout, (run / 'stderr.log').open('w') as stderr:
            result = subprocess.run(['bash', str(script)], cwd=repo, env=environment, stdout=stdout, stderr=stderr)
        record.update(state='completed' if result.returncode == 0 else 'failed',
                      exit_code=result.returncode, finished_at=now(), elapsed_seconds=time.monotonic() - began)
        write_json(run / 'receipt.json', record)
        run.rename(allocation / 'finished' / run.name)
        state.update(state='idle', current_run=None, updated_at=now())
        write_json(allocation / 'state.json', state)
        write_json(repo / 'hpc/active_allocation.json', state)
        print(json.dumps(record), flush=True)
        last_activity = time.monotonic()
    else:
        state['state'] = 'stopped'
    state['finished_at'] = now()
    write_json(allocation / 'state.json', state)
    write_json(repo / 'hpc/active_allocation.json', state)


if __name__ == '__main__':
    main()
