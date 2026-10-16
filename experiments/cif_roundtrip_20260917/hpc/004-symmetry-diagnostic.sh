#!/bin/bash
set -euo pipefail
/home/chenggui2/scratch/anaconda3/envs/matgl/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
import spglib
from pymatgen.io.cif import CifParser
from cif_codec import encode_cif,decode_bundle
from test_roundtrip import compare
rows=[]
for path in sorted(Path('results').glob('*/input.cif')):
    c=CifParser(str(path),frac_tolerance=0).parse_structures(primitive=False)[0]
    for tolerance in [1e-5,1e-7]:
        s=spglib.get_symmetry_dataset((c.lattice.matrix,c.frac_coords,c.atomic_numbers),symprec=tolerance)
        worst=0.
        for r,t in zip(s.rotations,s.translations):
            d=c.lattice.get_all_distances(c.frac_coords@r.T+t,c.frac_coords)
            same=np.array(c.atomic_numbers)[:,None]==np.array(c.atomic_numbers)[None,:]
            worst=max(worst,float(np.where(same,d,1e6).min(1).max()))
        row={'id':path.parent.name,'symprec_A':tolerance,'space_group':int(s.number),'operations':len(s.rotations),'worst_operation_site_residual_A':worst}
        if tolerance==1e-7:
            try:
                b=encode_cif(path,symprec=tolerance)
                c2,e=decode_bundle(b)
                _,_,err,_=compare(c,c2)
                row.update(strict_encode_decode='pass',error_A=err)
            except Exception as err: row.update(strict_encode_decode='fail',error=repr(err))
        rows.append(row)
Path('symmetry_diagnostic.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
PY
