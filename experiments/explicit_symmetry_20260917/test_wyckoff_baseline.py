"""Real CIF -> conventional Wyckoff table lookup -> independent decoder."""
import json
import os
from pathlib import Path
import subprocess
import sys
from pymatgen.io.cif import CifWriter

from test_program import parse,compare


assert os.environ.get('SLURM_JOB_ID')
root=Path(__file__).resolve().parent
assert (root/'wyckoff_baseline.py').exists(),'Wyckoff baseline codec is not implemented yet'
import wyckoff_baseline as baseline
table=baseline.reference_table()
assert len(table)==230
pilot=json.loads((root/'pilot/records.json').read_text())
out=root/'wyckoff_baseline_pilot';out.mkdir(exist_ok=True)
records=[]
for row in pilot:
    source=root/'corpus_512'/row['id']/'input.cif'
    full,standard,audit=baseline.encode_cif(source)
    original=parse(source)
    error=compare(original,baseline.decode(full))
    case=out/row['id'];case.mkdir(exist_ok=True)
    file=case/'wyckoff.tokens';file.write_text(full)
    (case/'wyckoff_standard.tokens').write_text(standard)
    target=case/'decoded.cif'
    if row['id'] in ['mp-30009','mp-1095422']:
        r=subprocess.run([sys.executable,str(root/'wyckoff_baseline.py'),'decode',str(file),str(target)],capture_output=True,text=True)
        assert r.returncode==0,r.stderr
    else:CifWriter(baseline.decode(file.read_text()),symprec=None,significant_figures=14).write_file(target)
    reread=compare(original,parse(target))
    records.append({'id':row['id'],'formula':row['formula'],'error_A':error,'reread_error_A':reread,**audit})
    print(json.dumps(records[-1]),flush=True)
(out/'records.json').write_text(json.dumps(records,indent=2))
print('10 Wyckoff baseline roundtrips passed')
