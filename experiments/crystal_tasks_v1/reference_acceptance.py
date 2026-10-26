"""CPU acceptance on complete reference banks and bounded real-structure fixtures."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .evaluation.references import load_split_rows, decode_reference_rows
from .evaluation.legacy import load_legacy_audit, extract_crystal
from .evaluation.design import evaluate_design


def main(dataset_dir, output):
    if output.exists():
        raise FileExistsError(output)
    started=time.monotonic()
    banks={};samples={};legacy=load_legacy_audit()
    for split in ('training','test'):
        rows=load_split_rows(dataset_dir,split)
        structures=[];selected=[]
        for index,row in enumerate(rows):
            structure=decode_reference_rows([row])[0]
            structures.append(structure)
            if len(selected)<32:
                body=extract_crystal(row['input']);audit=legacy.audit(body)
                if audit['valid']:
                    selected.append((row,structure,audit))
            if (index+1)%2000==0:
                print(split,index+1,flush=True)
        banks[split]=structures;samples[split]=selected
        print('REFERENCE_BANK_COMPLETE',split,len(structures),flush=True)
    assert len(banks['training'])==36362 and len(banks['test'])==4602
    results={}
    for split in ('training','test'):
        attempts=[]
        for row,structure,audit in samples[split]:
            attempts.append({'attempt_id':row['metadata']['material_id'],'seed':42,
                'request':{'space_group':audit['detected_sg'],'ehull_upper':.08},'output':row['input'],
                'property_receipt':{'score_status':'synthetic_metric_fixture','converged':True,
                    'post_relax_valid':True,'actual_space_group_number_post_relax':audit['detected_sg'],
                    'ehull_proxy_corrected_eV_per_atom':0.0,'relaxed_structure':structure.as_dict()}})
        refs=[s for _,s,_ in samples[split]]
        result=evaluate_design(attempts,banks['training'],refs)
        assert result['summary']['rates']['cov_recall']['rate']==1.0
        if split=='training':
            assert result['summary']['rates']['novel_unique_stable']['k']==0
        wrong=[{**a,'request':{**a['request'],'space_group':231}} for a in attempts]
        bad=evaluate_design(wrong,banks['training'],refs)
        assert bad['summary']['rates']['requested_sg_match']['k']==0
        results[split]={'fixtures':len(attempts),'material_ids':[a['attempt_id'] for a in attempts],
            'summary':result['summary'],'wrong_requested_sg_correctly_rejected':True}
    report={'checked_at':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,
        'decoded_reference_counts':{k:len(v) for k,v in banks.items()},
        'scope':'Full reference decode;32 real strict-valid fixtures/split;synthetic property receipts isolate metric logic;no GRACE or generation',
        'results':results}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('ACCEPTANCE_PASS',report['decoded_reference_counts'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.dataset_dir,args.output)
