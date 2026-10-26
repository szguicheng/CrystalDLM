from pathlib import Path
import hashlib,json,tarfile
root=Path('/home/liugang/src/crystalDLM/experiments/structure_request_eval_20261017/assets')
m=json.loads((root/'grace-weight-manifest.json').read_text());p=root/'GRACE-3L-OMAT-large-ft-AM-model.tar.gz'
assert p.stat().st_size==m['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
dest=root/'grace-model';dest.mkdir(exist_ok=True)
with tarfile.open(p) as t:t.extractall(dest,filter='data')
print('MODEL_PATH',dest/m['model'],flush=True)
import tensorpotential
import tensorflow as tf
from tensorpotential.calculator import TPCalculator
from ase.build import bulk
gpus=tf.config.list_physical_devices('GPU');assert len(gpus)==1
for g in gpus:tf.config.experimental.set_memory_growth(g,True)
calc=TPCalculator(str(dest/m['model']))
a=bulk('Si','diamond',a=5.43);a.calc=calc
energy=float(a.get_potential_energy());forces=a.get_forces();stress=a.get_stress()
import numpy as np
assert np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
r=dict(model=m,energy_eV=energy,atoms=len(a),max_force=float(np.linalg.norm(forces,axis=1).max()),stress_eV_A3=stress.tolist(),gpus=[g.name for g in gpus])
(root.parent/'grace-smoke.json').write_text(json.dumps(r,indent=2));print('GRACE_SMOKE',json.dumps(r),flush=True)
