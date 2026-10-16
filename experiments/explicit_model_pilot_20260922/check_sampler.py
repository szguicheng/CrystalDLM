"""Isolate sampler bookkeeping using a fixed oracle, not a learned network."""
import json
import torch
from model import ROOT, Denoiser, load_data, sample


class Oracle(torch.nn.Module):
    def __init__(self, targets, values, vocabulary):
        super().__init__()
        self.targets, self.values = targets, values
        self.vocabulary = vocabulary
        self.number_ids = torch.tensor([vocabulary.index(t) for t in ['<LENGTH>', '<ANGLE>', '<COORD>', '<OFFSET>', '<DELTA>']], device='cuda')

    def numeric(self, tokens): return (tokens[..., None] == self.number_ids).any(-1)

    def forward(self, tokens, values, composition, rate):
        logits = torch.full((*tokens.shape, len(self.vocabulary)), -20., device='cuda')
        logits.scatter_(2, self.targets[..., None], 20.)
        return logits, self.values


data, records, vocab = load_data('train', 'explicit')
y, v, c = [x[:8] for x in data]
model = Oracle(y, v, vocab)
out = []
for steps in [1, 8, 32]:
    ids, values = sample(model, c, steps=steps)
    assert torch.equal(ids, y)
    assert torch.equal(values[model.numeric(y)], v[model.numeric(y)])
    out.append({'steps': steps, 'exact_tokens': True, 'exact_numeric_values': True})
result = {'purpose': 'Oracle used only to isolate sampler indexing/reveal bookkeeping, never for training or generated metrics', 'checks': out}
(ROOT / 'sampler_bookkeeping_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
