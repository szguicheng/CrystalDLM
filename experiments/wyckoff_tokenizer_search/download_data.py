"""Download public pinned source data on the login node; no data computation."""
import hashlib
import json
import urllib.request
from pathlib import Path

COMMIT = 'e8e9add5df904284116af8a06c203555d908f290'
ROOT = f'https://raw.githubusercontent.com/SymmetryAdvantage/WyckoffTransformer/{COMMIT}/'
destination = Path('input')
destination.mkdir(exist_ok=True)
manifest = {'repository': 'https://github.com/SymmetryAdvantage/WyckoffTransformer',
            'commit': COMMIT, 'files': {}}
for split in ['train', 'val', 'test']:
    url = ROOT + f'data/mp_20/{split}.csv'
    target = destination / f'{split}.csv'
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=90) as response, target.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
    manifest['files'][split] = {'url': url, 'sha256': digest.hexdigest(),
                                 'bytes': target.stat().st_size}
    print(split, manifest['files'][split], flush=True)
(destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
