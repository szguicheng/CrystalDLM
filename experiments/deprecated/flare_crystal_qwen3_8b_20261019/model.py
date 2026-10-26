"""Crystal-only FLARE-style supervision with explicitly hidden shifted targets."""
BLOCK_SIZE = 4


def layout(clean, start, mask, mask_id, block_size=BLOCK_SIZE):
    assert 0 < start < len(clean) and len(mask) == len(clean) - start
    ids, positions = list(clean), list(range(len(clean)))
    queries, labels, targets, blocks = [], [], [], []
    for begin in range(start, len(clean), block_size):
        end = min(begin + block_size, len(clean))
        offset = len(ids)
        # Duplicate the preceding token so every target has a preceding noisy query.
        ids.append(clean[begin - 1])
        positions.append(begin - 1)
        for target in range(begin, end):
            hidden = mask[target - start]
            ids.append(mask_id if hidden else clean[target])
            positions.append(target)
            if hidden:
                queries.append(offset + target - begin)
                labels.append(clean[target])
                targets.append(target)
        blocks.append((offset, len(ids), begin))
    n = len(ids)
    allowed = [[False] * n for _ in range(n)]
    for q in range(len(clean)):
        allowed[q][:q + 1] = [True] * (q + 1)
    for left, right, begin in blocks:
        for q in range(left, right):
            allowed[q][:begin] = [True] * begin
            allowed[q][left:right] = [True] * (right - left)
    return dict(ids=ids, positions=positions, allowed=allowed, queries=queries,
                labels=labels, target_positions=targets)


def forward_hidden(model, ids, allowed, positions, crystal_start):
    import torch
    device = model.time.weight.device
    ids = torch.tensor(ids, dtype=torch.long, device=device)[None]
    positions = torch.tensor(positions, dtype=torch.long, device=device)[None]
    embedded = model.base.get_input_embeddings()(ids)
    embedded = embedded + model.time.weight[0] * (positions >= crystal_start).unsqueeze(-1)
    permitted = torch.tensor(allowed, dtype=torch.bool, device=device)[None, None]
    additive = torch.zeros_like(permitted, dtype=embedded.dtype).masked_fill(
        ~permitted, torch.finfo(embedded.dtype).min)
    return model.base.model(inputs_embeds=embedded, position_ids=positions,
        attention_mask={'full_attention': additive}, use_cache=False).last_hidden_state[0]


def objective(model, tok, example, generator, arm):
    import torch
    from torch.nn import functional as F
    assert arm in ('ar', 'flare')
    prefix = tok.encode(example['prefix'], add_special_tokens=False)
    lead = tok.encode(example['lead'], add_special_tokens=False)
    crystal = tok.encode(example['crystal'], add_special_tokens=False) + [model.end_id]
    assert lead[-1] == model.start_id
    clean = prefix + lead + crystal
    start = len(prefix) + len(lead)
    assert len(crystal) <= 544
    ar_positions = list(range(len(prefix) - 1, len(clean) - 1))
    ar_labels = torch.tensor(clean[len(prefix):], device=model.time.weight.device)
    if arm == 'ar':
        allowed = [[k <= q for k in range(len(clean))] for q in range(len(clean))]
        hidden = forward_hidden(model, clean, allowed, list(range(len(clean))), start)
        loss = F.cross_entropy(model.base.lm_head(hidden[ar_positions]).float(), ar_labels)
        return loss, dict(clean_ar=float(loss.detach()), denoise=0.)
    probabilities = torch.rand((len(crystal) + BLOCK_SIZE - 1) // BLOCK_SIZE,
                               generator=generator).repeat_interleave(BLOCK_SIZE)[:len(crystal)]
    mask = (torch.rand(len(crystal), generator=generator) < probabilities).tolist()
    diff_sum = 0.
    for view, selected in enumerate([mask, [not x for x in mask]]):
        plan = layout(clean, start, selected, model.mask_id)
        hidden = forward_hidden(model, plan['ids'], plan['allowed'], plan['positions'], start)
        if view == 0:
            ar_loss = F.cross_entropy(model.base.lm_head(hidden[ar_positions]).float(), ar_labels)
        if plan['queries']:
            labels = torch.tensor(plan['labels'], device=hidden.device)
            diff_sum = diff_sum + F.cross_entropy(
                model.base.lm_head(hidden[plan['queries']]).float(), labels, reduction='sum')
    # Per-crystal token contributes exactly one denoising target across the two views.
    diff_loss = diff_sum / len(ar_labels)
    return ar_loss + diff_loss, dict(clean_ar=float(ar_loss.detach()), denoise=float(diff_loss.detach()))


def denoise_logits(model, prefix, current, crystal_start):
    """Identical shifted-block visibility to training; no clean current-block copy."""
    clean_count = len(prefix)
    ids = prefix + [prefix[-1]] + current
    n = len(ids)
    allowed = [[False] * n for _ in range(n)]
    for q in range(clean_count):
        allowed[q][:q + 1] = [True] * (q + 1)
    for q in range(clean_count, n):
        allowed[q] = [True] * n
    positions = list(range(clean_count)) + list(range(clean_count - 1, clean_count + len(current)))
    hidden = forward_hidden(model, ids, allowed, positions, crystal_start)
    return model.base.lm_head(hidden[clean_count:clean_count + len(current)]).float()
