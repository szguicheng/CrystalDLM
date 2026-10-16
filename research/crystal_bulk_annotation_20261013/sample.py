"""Freeze one fifth of each existing train/val split; preserve every source fact."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select_ids(ids, seed):
    assert len(ids) == len(set(ids)), 'Duplicate material IDs'
    return sorted(random.Random(seed).sample(sorted(ids), len(ids) // 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='research/crystal_sharing_20261012/prepared_001')
    ap.add_argument('--output', default='research/crystal_bulk_annotation_20261013/subset_001')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    source, output = Path(args.source), Path(args.output)
    assert source.resolve() != output.resolve()
    source_manifest = json.loads((source / 'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'seed': args.seed, 'fraction': '1/5', 'rounding': 'floor',
                'algorithm': 'random.Random(seed).sample(sorted(material_ids), N // 5), independently per split',
                'source': str(source.resolve()), 'source_manifest_sha256': file_sha(source / 'manifest.json'),
                'splits': {}, 'test': 'unchanged, excluded from annotation subset'}
    selected_all = set()
    for split in ('train', 'val'):
        path = source / f'{split}-facts.jsonl.gz'
        input_sha = file_sha(path)
        assert input_sha == source_manifest['artifacts'][path.name]['sha256']
        ids = []
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                assert row['status'] == 'ok' and row['split'] == split
                ids.append(row['material_id'])
        assert len(ids) == source_manifest['eligible_split_counts'][split]
        selected = select_ids(ids, args.seed)
        assert selected == select_ids(list(reversed(ids)), args.seed)
        chosen = set(selected)
        assert not (chosen & selected_all)
        selected_all.update(chosen)
        id_text = ''.join(mid + '\n' for mid in selected)
        id_path = output / f'{split}-material-ids.txt'
        if id_path.exists():
            assert id_path.read_text() == id_text, 'Existing subset has different IDs'
        else:
            id_path.write_text(id_text)
        target = output / path.name
        temporary = target.with_suffix(target.suffix + '.tmp')
        count = 0
        with gzip.open(path, 'rb') as stream, temporary.open('wb') as raw_out:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw_out, mtime=0) as compressed:
                for line in stream:
                    if json.loads(line)['material_id'] in chosen:
                        compressed.write(line)
                        count += 1
        assert count == len(chosen) == len(ids) // 5
        if target.exists():
            assert file_sha(target) == file_sha(temporary), 'Existing subset facts differ'
            temporary.unlink()
        else:
            temporary.replace(target)
        manifest['splits'][split] = {'source_count': len(ids), 'selected_count': count,
            'source_sha256': input_sha, 'facts_path': target.name, 'facts_sha256': file_sha(target),
            'ids_path': id_path.name, 'ids_sha256': file_sha(id_path)}
    manifest['total_selected'] = len(selected_all)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Existing manifest differs'
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    # Sampling is independent of source row order, and never mixes original splits.
    assert select_ids([str(i) for i in range(15)], 42) == select_ids([str(i) for i in reversed(range(15))], 42)
    assert len(select_ids([str(i) for i in range(15)], 42)) == 3
    main()
