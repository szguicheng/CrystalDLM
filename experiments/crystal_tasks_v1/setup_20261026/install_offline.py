import json,subprocess,sys
from pathlib import Path
p=Path('/home/liugang/src/crystalDLM/experiments/crystal_tasks_v1/setup_20261026')
paths=json.loads((p/'verified-wheel-paths.json').read_text())
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--report',str(p/'install-report.json'),*paths],check=True)
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--index-url','https://mirrors.aliyun.com/pypi/simple','-e','/home/liugang/src/crystalDLM/work/LLaMA-Factory-0.9.5'],check=True)
subprocess.run([sys.executable,'-m','pip','check'],check=True)
(p/'pip-after.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
import importlib.metadata as md
versions={x:md.version(x) for x in ['torch','transformers','tokenizers','peft','accelerate','llamafactory','datasets','tensorboard','numpy','pymatgen','pyxtal']}
(p/'versions-after.json').write_text(json.dumps(versions,indent=2)+'\n');print('INSTALLED',versions,flush=True)
