"""One shared language model with causal text and absorbing crystal diffusion."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from prepare import END, K, MASK, MAXLEN, MODEL, PAD, REVISION, START, STEPS

LEGACY_OFFSET0_MODEL_SHA256 = 'd7abb6bcc6da9b339408ee928caef4b80a4cf8a40204a0b61b8a68b7c9aca79a'


def attention(ids, mode, current_span, start_id, end_id):
    n = len(ids)
    regions = torch.zeros(n, dtype=torch.long, device=ids.device)
    active, counter = 0, 0
    stop = current_span[0] if current_span else n
    for i, value in enumerate(ids[:stop].tolist()):
        if value == start_id:
            counter += 1; active = counter
        elif active:
            regions[i] = active
            if value == end_id:
                active = 0
    if current_span:
        regions[current_span[0]:current_span[1]] = counter + 1
    allowed = torch.ones((n, n), dtype=torch.bool, device=ids.device).tril()
    if mode == 'hybrid':
        allowed |= (regions[:, None] == regions[None, :]) & (regions[:, None] > 0)
    return allowed[None, None], regions


def prediction_positions(start, indices, mode, prediction_offset=0):
    assert mode in ('ar', 'hybrid') and prediction_offset in (0, 1)
    offset = 1 if mode == 'ar' else prediction_offset
    return [start+i-offset for i in indices]


def target_positions(start, count, mode, prediction_offset=0):
    return prediction_positions(start, range(count), mode, prediction_offset)


def reverse_step(current, predicted, k, mask_id, generator=None):
    hidden = current == mask_id
    release = torch.rand(current.shape, device=current.device, generator=generator) < 1/k
    return torch.where(hidden & release, predicted, current)


def unpack_block(ids, end_id, pad_id):
    if ids.count(end_id) != 1:
        raise ValueError('crystal block requires exactly one END')
    end = ids.index(end_id)
    if pad_id in ids[:end] or any(v != pad_id for v in ids[end+1:]):
        raise ValueError('invalid crystal PAD boundary')
    return ids[:end]


class Hybrid(nn.Module):
    def __init__(self, base, start_id, end_id, mask_id, pad_id, prediction_offset=0):
        super().__init__()
        assert prediction_offset in (0, 1)
        self.base = base
        self.time = nn.Embedding(STEPS+1, base.config.hidden_size)
        nn.init.zeros_(self.time.weight)
        self.start_id, self.end_id, self.mask_id, self.pad_id = start_id, end_id, mask_id, pad_id
        self.prediction_offset = prediction_offset
        self.forward_count = 0

    @classmethod
    def pretrained(cls, tok, prediction_offset=0):
        base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                    torch_dtype=torch.float32, attn_implementation='sdpa')
        base.resize_token_embeddings(len(tok), mean_resizing=False)
        return cls(base, *[tok.convert_tokens_to_ids(s) for s in [START, END, MASK, PAD]],
                   prediction_offset=prediction_offset)

    def restore_checkpoint(self, saved, expected_mode=None, expected_offset=None):
        config = saved['config']
        assert config['mode'] in ('ar', 'hybrid')
        if expected_mode is not None:
            assert config['mode'] == expected_mode
        if 'prediction_offset' in config:
            offset = config['prediction_offset']
        else:
            sources = config.get('source_sha256', {})
            assert sources.get('model.py') == LEGACY_OFFSET0_MODEL_SHA256
            offset = 0
        assert type(offset) is int and offset in (0, 1)
        if expected_offset is not None:
            assert type(expected_offset) is int and offset == expected_offset
        self.prediction_offset = offset
        self.load_state_dict(saved['model'], strict=True)
        return config

    def hidden(self, ids, allowed, times):
        self.forward_count += 1
        embedded = self.base.get_input_embeddings()(ids)
        embedded = embedded + self.time(times.clamp_min(0)) * (times >= 0).unsqueeze(-1)
        additive = torch.zeros_like(allowed, dtype=embedded.dtype).masked_fill(~allowed, torch.finfo(embedded.dtype).min)
        output = self.base.model(inputs_embeds=embedded, attention_mask={'full_attention': additive}, use_cache=False)
        return output.last_hidden_state

    def logits(self, ids, mode, span=None, k=0, positions=None):
        allowed, regions = attention(ids, mode, span, self.start_id, self.end_id)
        times = torch.where(regions > 0, 0, -1)
        if span:
            times[span[0]:span[1]] = k
        hidden = self.hidden(ids[None], allowed, times[None])[0]
        return self.base.lm_head(hidden if positions is None else hidden[positions])

    def pack(self, example, tok, mode, generator):
        encode = lambda s: tok.encode(s, add_special_tokens=False)
        prefix = encode(example['prefix'])
        groups = []
        if 'crystal' in example:
            lead = encode(example['lead'])
            start = len(prefix) + len(lead)
            assert lead[-1] == self.start_id
            clean = encode(example['crystal']) + [self.end_id]
            assert len(clean) <= K
            length = len(clean)
            clean += [self.pad_id] * (K-length)
            target = torch.tensor(clean)
            if mode == 'hybrid':
                k = int(example['noise_step']) if 'noise_step' in example else int(torch.randint(1, STEPS+1, (1,), generator=generator))
                assert 1 <= k <= STEPS
            else:
                k = 0
            hidden = torch.rand(K, generator=generator) < k/STEPS
            corrupted = torch.where(hidden, self.mask_id, target) if mode == 'hybrid' else target
            ids = torch.tensor(prefix+lead+corrupted.tolist())
            offset = self.prediction_offset if mode == 'hybrid' else 0
            span = (start-offset, start+K)
            if example.get('supervise_lead', True):
                groups.append(dict(kind='text', positions=target_positions(len(prefix), len(lead), 'ar'),
                                   labels=lead, weights=[1.]*len(lead), normalizer=len(lead)))
            if mode == 'hybrid':
                weights = torch.where(target == self.pad_id, .1, 1.)
                positions = hidden.nonzero().flatten().tolist()
                groups.append(dict(kind='crystal', positions=prediction_positions(start, positions, mode, self.prediction_offset),
                                   labels=target[hidden].tolist(), weights=(weights[hidden]*STEPS/k).tolist(),
                                   normalizer=float(weights.sum()), k=k))
            else:
                groups.append(dict(kind='crystal', positions=target_positions(start, length, 'ar'),
                                   labels=clean[:length], weights=[1.]*length, normalizer=length, k=0))
        else:
            target = encode(example['target'])
            ids = torch.tensor(prefix+target)
            span, k = None, 0
            groups.append(dict(kind='text', positions=target_positions(len(prefix), len(target), 'ar'),
                               labels=target, weights=[1.]*len(target), normalizer=len(target)))
        assert len(ids) <= MAXLEN, 'trajectory too long; never truncate'
        allowed, regions = attention(ids, mode, span, self.start_id, self.end_id)
        times = torch.where(regions > 0, 0, -1)
        if span:
            times[span[0]:span[1]] = k
        return dict(ids=ids, allowed=allowed, times=times, groups=groups)

    def loss(self, examples, tok, mode, generator):
        packed = [self.pack(e, tok, mode, generator) for e in examples]
        n = max(len(p['ids']) for p in packed)
        batch = len(packed)
        device = self.time.weight.device
        ids = torch.full((batch, n), self.pad_id, dtype=torch.long, device=device)
        times = torch.full_like(ids, -1)
        allowed = torch.zeros((batch, 1, n, n), dtype=torch.bool, device=device)
        for i, p in enumerate(packed):
            size = len(p['ids'])
            ids[i, :size], times[i, :size] = p['ids'].to(device), p['times'].to(device)
            allowed[i, :, :size, :size] = p['allowed'][0].to(device)
            if size < n:
                indices = torch.arange(size, n, device=device)
                allowed[i, 0, indices, indices] = True
        hidden = self.hidden(ids, allowed, times)
        loss = hidden.sum()*0
        stats = {'text': [], 'crystal': []}
        for i, p in enumerate(packed):
            for g in p['groups']:
                if not g['positions']:
                    continue
                logits = self.base.lm_head(hidden[i, g['positions']]).float()
                labels = torch.tensor(g['labels'], device=device)
                weights = torch.tensor(g['weights'], device=device)
                value = (F.cross_entropy(logits, labels, reduction='none') * weights).sum()/g['normalizer']
                loss = loss + value/batch
                stats[g['kind']].append(float(value.detach()))
        return loss, {key: sum(values)/len(values) if values else None for key, values in stats.items()}

    @torch.inference_mode()
    def text(self, prefix, tok, mode, trigger=False, limit=160):
        device = self.time.weight.device
        ids = tok.encode(prefix, add_special_tokens=False)
        generated = []
        banned = [self.mask_id, self.pad_id, self.end_id]
        if not trigger:
            banned.append(self.start_id)
        for _ in range(limit):
            x = torch.tensor(ids+generated, device=device)
            if len(x) >= MAXLEN:
                raise ValueError('text context capacity exceeded')
            logits = self.logits(x, mode, positions=[len(x)-1])[0].float()
            logits[banned] = -torch.inf
            value = int(logits.argmax())
            generated.append(value)
            if value == tok.eos_token_id or (trigger and value == self.start_id):
                break
        response = tok.decode(generated, skip_special_tokens=False)
        if trigger and (not generated or generated[-1] != self.start_id):
            raise ValueError('AR failed to trigger crystal generation: '+response)
        if not trigger and (not generated or generated[-1] != tok.eos_token_id):
            raise ValueError('AR response failed to terminate: '+response)
        return response

    @torch.inference_mode()
    def crystal(self, prefix, tok, mode, generator, temperature=.8):
        device = self.time.weight.device
        context = tok.encode(prefix, add_special_tokens=False)
        assert context[-1] == self.start_id
        assert len(context)+K <= MAXLEN
        banned = list(set(tok.all_special_ids) - {self.end_id, self.pad_id})
        if mode == 'hybrid':
            current = torch.full((K,), self.mask_id, device=device, dtype=torch.long)
            offset = self.prediction_offset
            span = (len(context)-offset, len(context)+K)
            for k in range(STEPS, 0, -1):
                positions = (current == self.mask_id).nonzero().flatten()
                if not len(positions):
                    break
                x = torch.tensor(context, device=device)
                queries = prediction_positions(len(x), positions.tolist(), mode, self.prediction_offset)
                logits = self.logits(torch.cat([x, current]), mode, span, k, queries).float()
                logits[:, banned] = -torch.inf
                values = torch.multinomial((logits/temperature).softmax(-1), 1, generator=generator).flatten()
                predicted = current.clone(); predicted[positions] = values
                current = reverse_step(current, predicted, k, self.mask_id, generator)
            raw = current.tolist()
        else:
            raw = []
            for _ in range(K):
                x = torch.tensor(context+raw, device=device)
                logits = self.logits(x, mode, positions=[len(x)-1])[0].float()
                logits[banned] = -torch.inf
                value = int(torch.multinomial((logits/temperature).softmax(-1), 1, generator=generator))
                raw.append(value)
                if value == self.end_id:
                    raw += [self.pad_id]*(K-len(raw))
                    break
        try:
            content = unpack_block(raw, self.end_id, self.pad_id)
        except ValueError as error:
            raw_text = tok.decode(raw, skip_special_tokens=False)
            failure = ValueError(str(error) + ': ' + raw_text)
            failure.raw_tokens, failure.raw_text = raw, raw_text
            raise failure from error
        return tok.decode(content, skip_special_tokens=False), raw
