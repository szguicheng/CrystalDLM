"""Process-local CPU assembly for model attention and time metadata."""
import sys

import torch


def fast_attention(ids, mode, current_span, start_id, end_id):
    n = len(ids)
    regions = [0] * n
    active, counter = 0, 0
    stop = current_span[0] if current_span else n
    for i, value in enumerate(ids[:stop].tolist()):
        if value == start_id:
            counter += 1
            active = counter
        elif active:
            regions[i] = active
            if value == end_id:
                active = 0
    if current_span:
        regions[current_span[0]:current_span[1]] = [counter + 1] * (current_span[1] - current_span[0])
    regions = torch.tensor(regions, dtype=torch.long, device=ids.device)
    allowed = torch.ones((n, n), dtype=torch.bool, device=ids.device).tril()
    if mode == 'hybrid' and current_span:
        allowed[current_span[0]:current_span[1], current_span[0]:current_span[1]] = True
    return allowed[None, None], regions


def fast_crystal_prefix_times(ids, start_id, end_id, pad_id):
    times = [-1] * len(ids)
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
    return torch.tensor(times, dtype=torch.long, device=ids.device)


def install(model):
    module = sys.modules[model.__class__.__module__]
    original = module.attention, module.crystal_prefix_times
    module.attention = fast_attention
    module.crystal_prefix_times = fast_crystal_prefix_times
    return original
