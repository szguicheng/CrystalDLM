"""Post-result diagnostic on training/validation only; never replace formal test metrics."""
import json
import signal
import sys
import warnings

import torch
from model import ROOT, Denoiser, load_data, render, sample

sys.path.insert(0, str(ROOT / 'codec'))
import program_codec as sp
import wyckoff_baseline as wp


def timeout(signum, frame): raise TimeoutError('Diagnostic decode timeout')


def main():
    torch.set_num_threads(4)
    wp.reference_table()
    signal.signal(signal.SIGALRM, timeout)
    results = []
    for representation, split, count, checkpoint_path in [
        ('explicit', 'train', 8, ROOT / 'runs/preflight.pt'),
        ('explicit', 'val', 64, ROOT / 'runs/explicit_17/best.pt'),
        ('wyckoff', 'val', 64, ROOT / 'runs/wyckoff_17/best.pt')]:
        data, records, vocab = load_data(split, representation)
        model = Denoiser(vocab).cuda()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint.get('model', checkpoint))
        data = [x[:count] for x in data]
        for steps in [1, 8, 32]:
            ids, numbers = sample(model, data[2], steps=steps)
            correct = ((ids == data[0]) & (data[0] != 0)).sum().item()
            total = (data[0] != 0).sum().item()
            decoded = 0; errors = []; examples = []
            for i, (row, values) in enumerate(zip(ids.tolist(), numbers.tolist())):
                text = render(vocab, row, values)
                signal.alarm(5)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        (sp if representation == 'explicit' else wp).decode(text)
                    decoded += 1
                except Exception as error: errors.append(f'{type(error).__name__}: {error}')
                finally: signal.alarm(0)
                if i < 3: examples.append({'id': records[i]['id'], 'program': text})
            entry = {'checkpoint': str(checkpoint_path.relative_to(ROOT)), 'split': split, 'n': count,
                     'steps': steps, 'decoded': decoded, 'nonpad_token_accuracy': correct / total,
                     'errors': errors, 'examples': examples}
            results.append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k not in ['errors', 'examples']}), flush=True)
    (ROOT / 'sampling_diagnostic.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__': main()
