"""Prepare the previously reserved validation cohort after freezing the model."""
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import torch
from prepare_full_prototype_corpus import initialize,work


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(1);start=time.perf_counter()
    corpus=Path('prototype_full_corpus');(corpus/'replication').mkdir(exist_ok=True)
    model=Path('prototype_full_model/final.pt');config=Path('prototype_full_model/config.json')
    previous=torch.load('multiplicity_candidates/data.pt',map_location='cpu')
    ordered=sorted(previous['splits']['val'],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
    selected=[r['id'] for r in ordered[2048:]];assert len(selected)==6620
    development=json.loads((corpus/'val_selection.json').read_text())
    assert development==[r['id'] for r in ordered[:2048]] and not(set(selected)&set(development))
    source=Path('input/val.csv')
    with source.open() as stream:by_id={r['material_id']:r for r in csv.DictReader(stream)}
    rows=[by_id[mid] for mid in selected]
    lock={'selected_ids':selected,'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
          'model_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
          'noise_seed':32179,'noise_repeats':8,'no_fitting_or_checkpoint_selection':True,
          'source_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (corpus/'replication_lock.json').write_text(json.dumps(lock,indent=2))
    records=[]
    with (corpus/'replication_progress.jsonl').open('w') as log, ProcessPoolExecutor(
            max_workers=4,initializer=initialize,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(work,((r,'replication') for r in rows),chunksize=8)):
            records.append(result);log.write(json.dumps(result)+'\n')
            if (i+1)%500==0:
                log.flush();print(json.dumps({'processed':i+1,'ready':sum(r['state']=='ready' for r in records),
                   'failed':sum(r['state']=='failed' for r in records),'seconds':time.perf_counter()-start}),flush=True)
    report={'selected':len(rows),'ready':sum(r['state']=='ready' for r in records),
            'excluded_no_internal_dof':sum(r['state']=='no_internal_dof' for r in records),
            'failures':[r for r in records if r['state']=='failed'],'records':records,
            'job_id':os.environ['SLURM_JOB_ID'],'seconds':time.perf_counter()-start,**lock}
    (corpus/'replication_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['selected','ready','excluded_no_internal_dof','seconds']}),flush=True)


if __name__=='__main__':main()
