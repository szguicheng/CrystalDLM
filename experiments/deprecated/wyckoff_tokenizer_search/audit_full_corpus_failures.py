"""Classify preparation failures without modifying corpus membership or files."""
from collections import Counter
import csv
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from audit_symmetry_graph import process
from discrete_graph_prototype import split
from prototype_conditions import compile_condition


def main():
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(1)
    out=Path('prototype_full_corpus');reports={}
    for partition in ['train','val']:
        manifest=json.loads((out/f'{partition}_manifest.json').read_text())
        failed={r['id'] for r in manifest['failures']}
        with (Path('input')/f'{partition}.csv').open() as stream:
            rows=[r for r in csv.DictReader(stream) if r['material_id'] in failed]
        details=[]
        for row in rows:
            stage='standardization_and_graph_codec'
            try:
                stats,encoded=process(row,4.5)
                stage='integer_prototype'
                prototype,geometry=split(encoded)
                stage='condition_compilation'
                condition=compile_condition(prototype,geometry['lattice'],71)
                stage='target_reconstruction'
                recovered=condition['q']+np.einsum('nij,nj->ni',condition['projection'],np.asarray(geometry['free_vectors']))
                assert np.allclose(recovered,[n['fractional'] for n in encoded['header']['nodes']],atol=1e-6)
                assert int(condition['multiplicity'].sum())==stats['atoms']
                details.append({'id':row['material_id'],'reproduced_failure':False})
            except Exception as error:
                frames=traceback.extract_tb(error.__traceback__)
                frame=frames[-1]
                details.append({'id':row['material_id'],'reproduced_failure':True,'stage':stage,
                                'type':type(error).__name__,'message':str(error),
                                'file':Path(frame.filename).name,'line':frame.lineno,'expression':frame.line})
        assert len(details)==len(failed)
        reports[partition]={'selected':manifest['selected'],'ready':manifest['ready'],
                            'no_internal_dof':manifest['excluded_no_internal_dof'],
                            'failed':len(failed),'details':details,
                            'failure_groups':dict(Counter(f"{d.get('stage')}|{d.get('type')}|{d.get('expression')}" for d in details))}
    reports['membership_changed']=False;reports['job_id']=os.environ['SLURM_JOB_ID']
    (out/'failure_audit.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({k:{n:v for n,v in x.items() if n!='details'} if isinstance(x,dict) else x for k,x in reports.items()}),flush=True)


if __name__=='__main__':main()
