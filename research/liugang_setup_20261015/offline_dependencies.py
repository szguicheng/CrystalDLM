"""Fetch the pip-resolved wheels with curl, verify hashes, then install offline."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path('/home/liugang/src/crystalDLM')
CACHE = ROOT / '.cache/wheelhouse'
PLAN = ROOT / 'research/liugang_setup_20261015/dependency-plan.json'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(item):
    info = item['download_info']
    expected = info['archive_info']['hashes']['sha256']
    url = info['url']
    name = unquote(urlsplit(url).path.rsplit('/', 1)[1])
    assert name.endswith('.whl') and Path(name).name == name, name
    path = CACHE / name
    if not path.exists() or sha256(path) != expected:
        mirror = url.replace('https://files.pythonhosted.org/packages/',
                             'https://mirrors.aliyun.com/pypi/packages/')
        part = path.with_name(name + '.partial')
        subprocess.run(['curl', '-4', '-fsSL', '--retry', '3', '--max-time', '300',
                        '-C', '-', '-o', str(part), mirror], check=True)
        assert sha256(part) == expected, f'wheel SHA-256 mismatch: {name}'
        part.replace(path)
    print('VERIFIED_WHEEL', name, flush=True)
    return path


def main():
    plan = json.loads(PLAN.read_text())
    CACHE.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(fetch, plan['install']))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    *map(str, paths)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'research/liugang_setup_20261015/pip-freeze.txt').write_text(frozen)
    print('DEPENDENCIES_READY', len(paths), flush=True)


if __name__ == '__main__':
    main()
