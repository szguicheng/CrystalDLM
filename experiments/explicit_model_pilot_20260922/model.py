"""Composition-only bidirectional masked denoiser; no target-length attention mask."""
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
MAXLEN = 256
NUMERIC = ['<LENGTH>', '<ANGLE>', '<COORD>', '<OFFSET>', '<DELTA>']
FIXED = {'SP1', 'WYCKOFF_STD1', 'CELL', 'SG', 'HALL', 'SITES', 'ROUND', 'GENERATORS',
         'G', 'ORIGIN', 'FAMILY', 'RESIDUALS', 'FREE', 'PARAM', 'END'}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_data(split, representation, device='cuda'):
    vocab = json.loads((ROOT / 'data/vocab.json').read_text())
    lookup = {t: i for i, t in enumerate(vocab)}
    records = []
    with (ROOT / 'data/records.jsonl').open() as file:
        for line in file:
            r = json.loads(line)
            if r['split'] == split and r['status'] == 'ok': records.append(r)
    ids = np.zeros((len(records), MAXLEN), dtype=np.int64)
    values = np.zeros_like(ids, dtype=np.float32)
    for i, r in enumerate(records):
        data = r['packed'][representation]; n = len(data['tokens'])
        ids[i, :n] = [lookup.get(t, 2) for t in data['tokens']]
        values[i, :n] = data['values']
    comp = np.log1p(np.array([r['composition'] for r in records], dtype=np.float32))
    return tuple(torch.tensor(a, device=device) for a in (ids, values, comp)), records, vocab


class Denoiser(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.register_buffer('number_ids', torch.tensor([vocab.index(t) for t in NUMERIC]))
        self.register_buffer('content_ids', torch.tensor([i for i, t in enumerate(vocab)
                                                        if i > 2 and t not in FIXED and t not in NUMERIC]))
        d = 192
        self.embedding = nn.Embedding(len(vocab), d)
        self.position = nn.Parameter(torch.randn(1, MAXLEN + 2, d) * 0.02)
        self.value_embedding = nn.Linear(1, d)
        self.composition = nn.Sequential(nn.Linear(118, d), nn.GELU(), nn.Linear(d, d))
        self.time_embedding = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 6, 768, dropout=0.1, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.logits = nn.Linear(d, len(vocab))
        self.numbers = nn.Linear(d, 1)

    def numeric(self, ids):
        return (ids[..., None] == self.number_ids).any(-1)

    def forward(self, tokens, values, composition, rate):
        x = self.embedding(tokens)
        x = x + self.value_embedding(values[..., None]) * self.numeric(tokens)[..., None]
        context = torch.stack((self.composition(composition), self.time_embedding(rate[:, None])), dim=1)
        x = torch.cat((context, x), dim=1) + self.position
        # Deliberately no padding or target-length mask: the model must learn END.
        x = self.norm(self.encoder(x))[:, 2:]
        return self.logits(x), self.numbers(x).squeeze(-1)


def corrupt(tokens, values, rate, generator=None):
    hidden = torch.rand(tokens.shape, device=tokens.device, generator=generator) < rate[:, None]
    return tokens.masked_fill(hidden, 1), values.masked_fill(hidden, 0), hidden


def objective(model, logits, predicted, target, values, hidden):
    ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction='none')
    weights = hidden.float() * torch.where(target == 0, 0.1, 1.0)
    discrete = (ce * weights).sum() / weights.sum().clamp_min(1)
    numerical = hidden & model.numeric(target)
    mse = ((predicted.float() - values).square() * numerical).sum() / numerical.sum().clamp_min(1)
    return discrete + mse, discrete, mse


@torch.no_grad()
def validate(model, data, seed=803):
    model.eval()
    tokens, values, comp = data
    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    total = dict(ce=0., weight=0., squared_error=0., numeric=0, correct=0, count=0,
                 content_correct=0, content_count=0, unknown=0)
    by_type = {kind: {'absolute_error_normalized': 0., 'count': 0} for kind in NUMERIC}
    for start in range(0, len(tokens), 32):
        y, v, c = (x[start:start+32] for x in data)
        rate = torch.full((len(y),), 0.5, device=y.device)
        x, u, hidden = corrupt(y, v, rate, generator)
        with torch.autocast('cuda', dtype=torch.bfloat16): logits, predicted = model(x, u, c, rate)
        ce = F.cross_entropy(logits.float().transpose(1, 2), y, reduction='none')
        weights = hidden.float() * torch.where(y == 0, 0.1, 1.0)
        numeric = hidden & model.numeric(y)
        real = hidden & (y != 0)
        content = real & (y[..., None] == model.content_ids).any(-1)
        correct = logits.argmax(-1) == y
        total['ce'] += float((ce * weights).sum()); total['weight'] += float(weights.sum())
        total['squared_error'] += float(((predicted.float() - v).square() * numeric).sum())
        for key, mask in [('numeric', numeric), ('count', real), ('content_count', content)]:
            total[key] += int(mask.sum())
        total['correct'] += int((correct & real).sum())
        total['content_correct'] += int((correct & content).sum())
        total['unknown'] += int(((y == 2) & real).sum())
        for kind in NUMERIC:
            mask = hidden & (y == model.vocab.index(kind))
            by_type[kind]['absolute_error_normalized'] += float(((predicted.float() - v).abs() * mask).sum())
            by_type[kind]['count'] += int(mask.sum())
    ce = total['ce'] / total['weight']
    mse = total['squared_error'] / max(1, total['numeric'])
    return {'loss': ce + mse, 'ce': ce, 'numeric_rmse_normalized': math.sqrt(mse),
            'masked_nonpad_accuracy': total['correct'] / total['count'],
            'masked_content_accuracy': total['content_correct'] / max(1, total['content_count']),
            'counts': total, 'numeric_by_type': by_type}


def denormalize(kind, value):
    if kind == '<LENGTH>': return math.exp(value)
    if kind == '<ANGLE>': return 180 / (1 + math.exp(-value))
    if kind == '<COORD>': return (value + 0.5) % 1
    if kind == '<DELTA>': return value / 1e4
    return value


def render(vocab, ids, values):
    out = []
    for index, value in zip(ids, values):
        token = vocab[int(index)]
        out.append(format(denormalize(token, float(value)), '.14g') if token in NUMERIC else token)
        if token == 'END': break
    return ' '.join(out)


@torch.no_grad()
def sample(model, composition, steps=32):
    model.eval()
    count = len(composition)
    ids = torch.full((count, MAXLEN), 1, device=composition.device, dtype=torch.long)
    values = torch.zeros_like(ids, dtype=torch.float)
    for step in range(steps):
        hidden = ids == 1
        rate = hidden.float().mean(-1)
        with torch.autocast('cuda', dtype=torch.bfloat16): logits, predicted = model(ids, values, composition, rate)
        # MASK/UNK are not valid output symbols. Both representations use this same rule.
        logits[:, :, 1:3] = -torch.inf
        probabilities = logits.float().softmax(-1)
        confidence, proposed = probabilities.max(-1)
        confidence.masked_fill_(~hidden, -torch.inf)
        target_remaining = int(MAXLEN * math.cos((step + 1) / steps * math.pi / 2))
        reveal = int(hidden[0].sum()) - target_remaining
        positions = confidence.topk(reveal, dim=1).indices
        ids.scatter_(1, positions, proposed.gather(1, positions))
        values.scatter_(1, positions, predicted.float().gather(1, positions))
        values.masked_fill_(~model.numeric(ids), 0)
    assert not (ids == 1).any()
    return ids, values
