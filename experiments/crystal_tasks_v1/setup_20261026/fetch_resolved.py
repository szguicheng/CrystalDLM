from concurrent.futures import ThreadPoolExecutor
import importlib.util,json
from pathlib import Path
root=Path('/home/liugang/src/crystalDLM')
p=root/'research/liugang_setup_20261015/offline_dependencies.py'
spec=importlib.util.spec_from_file_location('project_wheel_fetch',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.CACHE=root/'.cache/crystal_tasks_v1_wheels';module.CACHE.mkdir(exist_ok=True)
plan=json.loads((root/'experiments/crystal_tasks_v1/setup_20261026/dependency-plan.json').read_text())
items=[x for x in plan['install'] if x['metadata']['name']!='llamafactory']
with ThreadPoolExecutor(max_workers=6) as pool: paths=list(pool.map(module.fetch,items))
(root/'experiments/crystal_tasks_v1/setup_20261026/verified-wheel-paths.json').write_text(json.dumps([str(x) for x in paths],indent=2)+'\n')
print('ALL_WHEELS_VERIFIED',len(paths),flush=True)
