"""One shared language model with causal text and absorbing crystal diffusion."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from prepare import END, K, MASK, MAXLEN, MODEL, PAD, REVISION, START, STEPS

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
    if mode == 'hybrid' and current_span:
        allowed[current_span[0]:current_span[1], current_span[0]:current_span[1]] = True
    return allowed[None, None], regions


def block_attention(ids, context_len, block_size):
    size = (len(ids) - context_len) // 2
    assert size > 0 and len(ids) == context_len + 2 * size and size % block_size == 0
    clean_start, noisy_start = context_len, context_len + size
    allowed = torch.ones((len(ids), len(ids)), dtype=torch.bool, device=ids.device).tril()
    for block_start in range(noisy_start, len(ids), block_size):
        block_end = block_start + block_size
        block_index = (block_start - noisy_start) // block_size
        allowed[block_start:block_end] = False
        allowed[block_start:block_end, :context_len] = True
        allowed[block_start:block_end, clean_start:clean_start + block_index * block_size] = True
        allowed[block_start:block_end, block_start:block_end] = True
    return allowed[None, None]


def crystal_prefix_times(ids, start_id, end_id, pad_id):
    times = torch.full((len(ids),), -1, dtype=torch.long, device=ids.device)
    active = padding = False
    for i, value in enumerate(ids.tolist()):
        if value == start_id:
            active, padding = True, False
        elif active:
            times[i] = 0
            if value == end_id:
                active, padding = False, True
        elif padding and value == pad_id:
            times[i] = 0
        else:
            padding = False
    return times


class CrystalBlockError(ValueError):
    def __init__(self, message, raw_ids):
        self.raw_ids = list(raw_ids)
        super().__init__(message)


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
    def __init__(self, base, start_id, end_id, mask_id, pad_id, prediction_offset=0, block_size=K):
        super().__init__()
        assert prediction_offset == 0
        assert type(block_size) is int and block_size > 0 and K % block_size == 0
        self.base = base
        self.time = nn.Embedding(STEPS+1, base.config.hidden_size)
        nn.init.zeros_(self.time.weight)
        self.start_id, self.end_id, self.mask_id, self.pad_id = start_id, end_id, mask_id, pad_id
        self.prediction_offset = prediction_offset
        self.block_size = block_size
        self.forward_count = 0

    @classmethod
    def pretrained(cls, tok, prediction_offset=0, block_size=K):
        base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
                    torch_dtype=torch.float32, attn_implementation='sdpa')
        base.resize_token_embeddings(len(tok), mean_resizing=False)
        return cls(base, *[tok.convert_tokens_to_ids(s) for s in [START, END, MASK, PAD]],
                   prediction_offset=prediction_offset, block_size=block_size)

    def restore_checkpoint(self, saved, expected_mode=None, expected_offset=None, expected_block_size=None):
        config = saved['config']
        assert config['mode'] in ('ar', 'hybrid')
        if expected_mode is not None:
            assert config['mode'] == expected_mode
        offset = config['prediction_offset']
        assert type(offset) is int and offset == 0
        if expected_offset is not None:
            assert type(expected_offset) is int and offset == expected_offset
        assert type(config['block_size']) is int and config['block_size'] == self.block_size
        if expected_block_size is not None:
            assert type(expected_block_size) is int and config['block_size'] == expected_block_size
        self.load_state_dict(saved['model'], strict=True)
        return config

    def hidden(self, ids, allowed, times, position_ids=None):
        self.forward_count += 1
        embedded = self.base.get_input_embeddings()(ids)
        embedded = embedded + self.time(times.clamp_min(0)) * (times >= 0).unsqueeze(-1)
        additive = torch.zeros_like(allowed, dtype=embedded.dtype).masked_fill(~allowed, torch.finfo(embedded.dtype).min)
        output = self.base.model(inputs_embeds=embedded, attention_mask={'full_attention': additive},
                                 position_ids=position_ids, use_cache=False)
        return output.last_hidden_state

    def logits(self, ids, mode, span=None, k=0, positions=None, times=None, position_ids=None):
        allowed, regions = attention(ids, mode, span, self.start_id, self.end_id)
        if times is None:
            if mode == 'hybrid':
                prefix = crystal_prefix_times(ids[:span[0]] if span else ids,
                                              self.start_id, self.end_id, self.pad_id)
                times = (torch.cat((prefix, torch.full((len(ids) - len(prefix),), -1,
                                                       dtype=torch.long, device=ids.device)))
                         if span else prefix)
            else:
                times = torch.where(regions > 0, 0, -1)
            if span:
                times[span[0]:span[1]] = k
        if position_ids is None:
            position_ids = torch.arange(len(ids), device=ids.device)
        hidden = self.hidden(ids[None], allowed, times[None], position_ids[None])[0]
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
            context = prefix + lead
            if mode == 'hybrid':
                noisy_start = start + K
                ids = torch.tensor(context + clean + corrupted.tolist())
                allowed = block_attention(ids, start, self.block_size)
                times = torch.cat((crystal_prefix_times(torch.tensor(context), self.start_id, self.end_id, self.pad_id),
                                   torch.zeros(K, dtype=torch.long), torch.full((K,), k, dtype=torch.long)))
                position_ids = torch.tensor(list(range(start)) + list(range(start, start + K)) * 2)
                positions = (noisy_start + hidden.nonzero().flatten()).tolist()
                weights = torch.where(target == self.pad_id, .1, 1.)
                groups.append(dict(kind='crystal', positions=positions,
                                   labels=target[hidden].tolist(), weights=(weights[hidden]*STEPS/k).tolist(),
                                   normalizer=float(weights.sum()), k=k))
            else:
                noisy_start = start
                ids = torch.tensor(context+corrupted.tolist())
                span = (start, start+K)
                allowed, regions = attention(ids, mode, span, self.start_id, self.end_id)
                times = torch.where(regions > 0, 0, -1)
                times[span[0]:span[1]] = k
                position_ids = torch.arange(len(ids))
            if example.get('supervise_lead', True):
                groups.append(dict(kind='text', positions=target_positions(len(prefix), len(lead), 'ar'),
                                   labels=lead, weights=[1.]*len(lead), normalizer=len(lead)))
            if mode == 'ar':
                groups.append(dict(kind='crystal', positions=target_positions(start, length, 'ar'),
                                   labels=clean[:length], weights=[1.]*length, normalizer=length, k=0))
        else:
            target = encode(example['target'])
            ids = torch.tensor(prefix+target)
            k = 0
            groups.append(dict(kind='text', positions=target_positions(len(prefix), len(target), 'ar'),
                               labels=target, weights=[1.]*len(target), normalizer=len(target)))
            assert len(ids) <= MAXLEN, 'trajectory too long; never truncate'
            allowed, regions = attention(ids, mode, None, self.start_id, self.end_id)
            times = (crystal_prefix_times(ids, self.start_id, self.end_id, self.pad_id)
                     if mode == 'hybrid' else torch.where(regions > 0, 0, -1))
            position_ids = torch.arange(len(ids))
        assert len(ids) <= MAXLEN, 'trajectory too long; never truncate'
        return dict(ids=ids, allowed=allowed, times=times, position_ids=position_ids, groups=groups)

    def loss(self, examples, tok, mode, generator):
        packed = [self.pack(e, tok, mode, generator) for e in examples]
        n = max(len(p['ids']) for p in packed)
        batch = len(packed)
        device = self.time.weight.device
        ids = torch.full((batch, n), self.pad_id, dtype=torch.long, device=device)
        times = torch.full_like(ids, -1)
        position_ids = torch.zeros_like(ids)
        allowed = torch.zeros((batch, 1, n, n), dtype=torch.bool, device=device)
        for i, p in enumerate(packed):
            size = len(p['ids'])
            ids[i, :size], times[i, :size] = p['ids'].to(device), p['times'].to(device)
            position_ids[i, :size] = p['position_ids'].to(device)
            allowed[i, :, :size, :size] = p['allowed'][0].to(device)
            if size < n:
                indices = torch.arange(size, n, device=device)
                allowed[i, 0, indices, indices] = True
        hidden = self.hidden(ids, allowed, times, position_ids)
        loss = hidden.sum()*0
        stats = {'text': [], 'crystal': [], 'crystal_body': [], 'crystal_end': [], 'crystal_pad': []}
        metrics = {name: {'count': 0, 'correct': 0, 'nll_sum': 0.} for name in ('body', 'end', 'pad')}
        for i, p in enumerate(packed):
            for g in p['groups']:
                if not g['positions']:
                    continue
                logits = self.base.lm_head(hidden[i, g['positions']]).float()
                labels = torch.tensor(g['labels'], device=device)
                weights = torch.tensor(g['weights'], device=device)
                nll = F.cross_entropy(logits, labels, reduction='none')
                token_loss = nll * weights
                value = token_loss.sum()/g['normalizer']
                loss = loss + value/batch
                stats[g['kind']].append(float(value.detach()))
                if g['kind'] == 'crystal':
                    predicted = logits.argmax(-1)
                    for name, selected in (('body', (labels != self.end_id) & (labels != self.pad_id)),
                                           ('end', labels == self.end_id), ('pad', labels == self.pad_id)):
                        stats['crystal_' + name].append(float((token_loss * selected).sum().detach()/g['normalizer']))
                        metrics[name]['count'] += int(selected.sum())
                        metrics[name]['correct'] += int(((predicted == labels) & selected).sum())
                        metrics[name]['nll_sum'] += float(nll[selected].sum().detach())
        summary = {key: sum(values)/len(values) if values else None for key, values in stats.items()}
        summary['crystal_metrics'] = metrics
        return loss, summary

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
            raw = []
            for start in range(0, K, self.block_size):
                current = torch.full((self.block_size,), self.mask_id, device=device, dtype=torch.long)
                for k in range(STEPS, 0, -1):
                    positions = (current == self.mask_id).nonzero().flatten()
                    if not len(positions):
                        break
                    x = torch.tensor(context + raw, device=device)
                    span = (len(x), len(x) + self.block_size)
                    queries = [span[0] + i for i in positions.tolist()]
                    times = torch.cat((crystal_prefix_times(torch.tensor(context, device=device), self.start_id,
                                                              self.end_id, self.pad_id),
                                       torch.zeros(len(raw), dtype=torch.long, device=device),
                                       torch.full((self.block_size,), k, dtype=torch.long, device=device)))
                    logits = self.logits(torch.cat((x, current)), mode, span, k, queries, times=times).float()
                    logits[:, banned] = -torch.inf
                    values = torch.multinomial((logits/temperature).softmax(-1), 1, generator=generator).flatten()
                    predicted = current.clone(); predicted[positions] = values
                    current = reverse_step(current, predicted, k, self.mask_id, generator)
                raw.extend(current.tolist())
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
            raise CrystalBlockError(str(error) + ': ' + tok.decode(raw, skip_special_tokens=False), raw) from error
        return tok.decode(content, skip_special_tokens=False), raw
