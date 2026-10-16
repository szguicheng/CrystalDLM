"""Equivalence check for process-local fast metadata functions."""
import random
import sys
import types

import torch

from fast_inference_metadata import fast_attention, fast_crystal_prefix_times, install


def baseline_attention(ids, mode, current_span, start_id, end_id):
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


def baseline_crystal_prefix_times(ids, start_id, end_id, pad_id):
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


def same(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual, expected), (actual, expected)


def check(device='cpu'):
    start_id, end_id, pad_id = 7, 8, 0
    fixtures = [
        [],
        [0, 1, 2, 0],
        [1, 7, 2, 3, 8, 0, 0, 4],
        [7, 1, 7, 2, 8, 0, 0, 3],  # nested START retags the active region
        [2, 7, 1, 0, 3],  # unclosed region includes PAD and later tokens
        [7, 1, 8, 0, 0, 2, 0],  # only contiguous PAD after END stays in the prefix
    ]
    rng = random.Random(601006)
    random_ids = [
        [rng.choice((pad_id, start_id, end_id, 1, 2, 3, 4)) for _ in range(rng.randrange(25))]
        for _ in range(24)
    ]
    for values in fixtures + random_ids:
        ids = torch.tensor(values, dtype=torch.long, device=device)
        spans = [None, (0, len(values)), (len(values), len(values))]
        for _ in range(3):
            a = rng.randrange(len(values) + 1)
            b = rng.randrange(a, len(values) + 1)
            spans.append((a, b))
        same(fast_crystal_prefix_times(ids, start_id, end_id, pad_id),
             baseline_crystal_prefix_times(ids, start_id, end_id, pad_id))
        for span in spans:
            for mode in ('ar', 'hybrid'):
                actual = fast_attention(ids, mode, span, start_id, end_id)
                expected = baseline_attention(ids, mode, span, start_id, end_id)
                same(actual[0], expected[0])
                same(actual[1], expected[1])


def check_install():
    name = '_fast_metadata_install_check'
    module = types.ModuleType(name)
    module.attention = baseline_attention
    module.crystal_prefix_times = baseline_crystal_prefix_times
    sentinel = module.sentinel = object()
    sys.modules[name] = module
    Model = type('Model', (), {'__module__': name})
    originals = install(Model())
    assert originals == (baseline_attention, baseline_crystal_prefix_times)
    assert module.attention is fast_attention
    assert module.crystal_prefix_times is fast_crystal_prefix_times
    assert module.sentinel is sentinel
    del sys.modules[name]


if __name__ == '__main__':
    torch.set_num_threads(1)
    check()
    check_install()
    print('fast inference metadata equivalence passed on CPU')
