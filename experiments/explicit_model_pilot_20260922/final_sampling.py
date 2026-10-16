"""Freeze sampler using validation only, test it once against the original on fresh IDs."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from model import ROOT, NUMERIC, load_data
from sampling_followup import OUT, variant


def check_indexing():
    data, _, vocab = load_data('train', 'explicit')
    target, values, composition = [x[:8] for x in data]
    class FixedOracle(torch.nn.Module):
        def numeric(self, tokens):
            return (tokens[..., None] == torch.tensor([vocab.index(t) for t in NUMERIC], device='cuda')).any(-1)
        def forward(self, tokens, numeric_values, condition, rate):
            logits = torch.full((*tokens.shape, len(vocab)), -20., device='cuda')
            logits.scatter_(2, target[..., None], 20.)
            return logits, values
    model = FixedOracle()
    checks = []
    for mode in ['one_step', 'confidence', 'random', 'content_first', 'remask']:
        ids, numbers = variant(model, composition, mode)
        assert torch.equal(ids, target), mode
        assert torch.equal(numbers[model.numeric(target)], values[model.numeric(target)]), mode
        checks.append({'mode': mode, 'exact_ids': True, 'exact_numbers': True})
    (OUT / 'sampler_checks.json').write_text(json.dumps(checks, indent=2))


def main():
    check_indexing()
    entries = []
    for seed in [17, 29]:
        entries.extend(json.loads((OUT / 'sampler_v1' / f'val_512_{seed}/summary.json').read_text()))
    ranking = []
    for mode in ['one_step', 'confidence', 'random', 'content_first', 'remask']:
        rows = [r for r in entries if r['mode'] == mode]
        assert len(rows) == 4 and all(r['n'] == 512 for r in rows)
        ranking.append({'mode': mode, 'n': sum(r['n'] for r in rows),
                        **{key: sum(r['counts'][key] for r in rows) for key in ['usable', 'composition_match', 'decoded']}})
    ranking.sort(key=lambda r: (r['usable'], r['composition_match'], r['decoded'], r['mode']), reverse=True)
    chosen = ranking[0]['mode']
    config = {'selected_at': datetime.now(timezone.utc).isoformat(), 'selected': chosen,
              'selection_data': '512 validation structures x 2 representations x 2 fixed training seeds',
              'selection_order': ['usable', 'composition_match', 'decoded'], 'ranking': ranking,
              'fresh_data': json.loads((OUT / 'fresh_summary.json').read_text()),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ['model.py', 'sampling_followup.py', 'evaluate.py']},
              'checkpoint_sha256': {f'{rep}_{seed}': hashlib.sha256((ROOT / f'runs/{rep}_{seed}/best.pt').read_bytes()).hexdigest()
                                    for rep in ['explicit', 'wyckoff'] for seed in [17, 29]}}
    assert not (OUT / 'sampler_selection.json').exists(), 'Do not silently reselect after looking at fresh results'
    (OUT / 'sampler_selection.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config), flush=True)
    for seed in [17, 29]:
        subprocess.run([sys.executable, '-u', 'sampling_followup.py', '--count', '512', '--seed', str(seed),
                        '--split', 'fresh', '--label', 'fresh_frozen', '--modes', *dict.fromkeys(['confidence', chosen])],
                       check=True, cwd=ROOT)


if __name__ == '__main__': main()
