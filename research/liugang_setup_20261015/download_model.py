"""Download the pinned Qwen3-8B snapshot and verify Git/LFS object hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/liugang/src/crystalDLM')
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
REPO = 'Qwen/Qwen3-8B'
DEST = ROOT / 'models' / ('Qwen3-8B-' + REVISION[:12])
ENDPOINT = 'https://hf-mirror.com'


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    digest = hashlib.sha256() if entry.get('lfs') else hashlib.sha1()
    if not entry.get('lfs'):
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    expected = entry['lfs']['oid'] if entry.get('lfs') else entry['oid']
    return digest.hexdigest() == expected


def download(entry):
    name = entry['path']
    path = DEST / name
    if verify(path, entry):
        print('VERIFIED_EXISTING', name, flush=True)
        return
    temporary = path.with_name(path.name + '.partial')
    url = entry.get('download_url', f'{ENDPOINT}/{REPO}/resolve/{REVISION}/{name}')
    print('DOWNLOADING', name, entry['size'], flush=True)
    subprocess.run(['curl', '-4', '-fsSL', '--retry', '4', '--retry-delay', '2',
                    '--connect-timeout', '20', '--speed-time', '120', '--speed-limit', '1024',
                    '-C', '-', '-o', str(temporary), url], check=True)
    assert verify(temporary, entry), f'hash/size mismatch: {name}'
    temporary.replace(path)
    print('VERIFIED_DOWNLOADED', name, flush=True)


def main():
    assert ROOT.is_dir() and not ROOT.is_symlink()
    DEST.mkdir(parents=True, exist_ok=True)
    url = f'{ENDPOINT}/api/models/{REPO}/revision/{REVISION}?blobs=true'
    metadata = json.loads(subprocess.check_output(['curl', '-4', '-fsSL', '--max-time', '30', url]))
    assert metadata['sha'] == REVISION
    modelscope_url = f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=false'
    modelscope_metadata = json.loads(subprocess.check_output(
        ['curl', '-4', '-fsSL', '--max-time', '30', modelscope_url]))
    modelscope_files = {e['Name']: e for e in modelscope_metadata['Data']['Files']}
    entries = []
    for item in metadata['siblings']:
        if item['rfilename'] in ('.gitattributes', 'README.md'):
            continue
        entry = {'path': item['rfilename'], 'size': item['size'], 'oid': item['blobId']}
        if item.get('lfs'):
            entry['lfs'] = {'oid': item['lfs']['sha256']}
            assert modelscope_files[entry['path']]['Sha256'] == entry['lfs']['oid']
            entry['download_url'] = f"https://modelscope.cn/models/{REPO}/resolve/master/{entry['path']}"
        entries.append(entry)
    assert len([e for e in entries if e['path'].endswith('.safetensors')]) == 5
    assert all('/' not in e['path'] and '..' not in e['path'] for e in entries)
    manifest = {'model': REPO, 'revision': REVISION, 'endpoint': ENDPOINT, 'files': entries}
    (DEST / 'download-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, entries))
    assert all(verify(DEST / entry['path'], entry) for entry in entries)
    (DEST / 'VERIFIED.json').write_text(json.dumps({'model': REPO, 'revision': REVISION,
                                                 'files': len(entries), 'all_hashes_verified': True}, indent=2) + '\n')
    print('SNAPSHOT_READY', DEST, flush=True)


if __name__ == '__main__':
    main()
