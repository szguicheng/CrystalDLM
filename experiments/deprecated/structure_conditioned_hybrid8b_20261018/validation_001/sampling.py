"""Two matched confidence samplers; optional revisiting of committed positions."""
import math


def remask_low(current, confidence, mask_id, fraction):
    current, confidence = list(current), list(confidence)
    known = [i for i, token in enumerate(current) if token != mask_id]
    changed = sorted(known, key=lambda i: (confidence[i], i))[:math.ceil(fraction * len(known))]
    for i in changed:
        current[i], confidence[i] = mask_id, -1.
    return current, confidence, changed


def commit_best(current, confidence, positions, predictions, probabilities, known_target):
    current, confidence = list(current), list(confidence)
    count = known_target - (len(current) - len(positions))
    assert 0 <= count <= len(positions)
    selected = sorted(range(len(positions)), key=lambda j: (-probabilities[j], positions[j]))[:count]
    for j in selected:
        current[positions[j]], confidence[positions[j]] = predictions[j], probabilities[j]
    return current, confidence, [positions[j] for j in selected]


def diffuse(model, prefix_ids, capacity, steps, generator, banned_ids, temperature,
            top_p, sample_rows, revisit=False):
    import torch

    device = model.time.weight.device
    current, confidence = [model.mask_id] * capacity, [-1.] * capacity
    start = len(prefix_ids)
    trace = []
    for step in range(steps):
        previous = list(current)
        reconsidered = []
        if revisit and step:
            current, confidence, reconsidered = remask_low(current, confidence, model.mask_id, .1)
        positions = [i for i, token in enumerate(current) if token == model.mask_id]
        # Condition on the actual corruption after remasking, not the old nominal step.
        k = max(1, min(steps, round(steps * len(positions) / capacity)))
        ids = torch.tensor(prefix_ids + current, dtype=torch.long, device=device)
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model.logits(ids, 'hybrid', span=(start, start + capacity), k=k,
                                  positions=[start + i for i in positions])
            predictions = sample_rows(logits, generator, banned_ids, temperature, top_p)
            scores = logits.float() / temperature
            scores[:, list(banned_ids)] = -torch.inf
            probabilities = scores.softmax(-1).gather(1, predictions[:, None]).squeeze(1).tolist()
        target = capacity * (step + 1) // steps
        current, confidence, committed = commit_best(current, confidence, positions,
            predictions.tolist(), probabilities, target)
        assert sum(t != model.mask_id for t in current) == target
        trace.append(dict(step=step + 1, noise_step=k, masked_before=len(positions),
                          committed=len(committed), reconsidered=len(reconsidered),
                          revised=sum(current[i] != previous[i] and current[i] != model.mask_id
                                      for i in reconsidered)))
    assert model.mask_id not in current
    return current, trace
