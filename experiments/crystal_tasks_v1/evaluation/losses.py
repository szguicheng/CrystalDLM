"""Token-weighted autoregressive and denoising loss metrics."""


SEGMENTS = {0: "input", 1: "description", 2: "crystal", 3: "answer"}


def _nll_metrics(nll_sum, token_count):
    return {
        "nll_sum": nll_sum,
        "token_count": token_count,
        "nll_mean": nll_sum / token_count if token_count else None,
    }


def causal_nll(logits, labels, segment_ids=None):
    import torch.nn.functional as F

    vocab_size = logits.shape[-1]
    shifted_logits = logits[:, :-1, :].reshape(-1, vocab_size)
    shifted_labels = labels[:, 1:].reshape(-1)
    token_nll = F.cross_entropy(
        shifted_logits, shifted_labels, reduction="none", ignore_index=-100
    )
    valid = shifted_labels.ne(-100)

    def metrics(mask):
        count = int(mask.sum().item())
        loss_sum = float(token_nll[mask].sum().item())
        return _nll_metrics(loss_sum, count)

    result = metrics(valid)
    result["segments"] = {}
    if segment_ids is not None:
        shifted_segments = segment_ids[:, 1:].reshape(-1)
        result["segments"] = {
            name: metrics(valid & shifted_segments.eq(segment))
            for segment, name in SEGMENTS.items()
        }
    return result


def aggregate_causal_nll(batches):
    batches = list(batches)
    result = _nll_metrics(
        sum((batch["nll_sum"] for batch in batches), 0.0),
        sum(batch["token_count"] for batch in batches),
    )
    result["segments"] = {}
    for name in SEGMENTS.values():
        metrics = [batch["segments"][name] for batch in batches if name in batch["segments"]]
        if metrics:
            result["segments"][name] = _nll_metrics(
                sum((item["nll_sum"] for item in metrics), 0.0),
                sum(item["token_count"] for item in metrics),
            )
    return result


def denoising_metrics(ce_sum, crystal_tokens, text_tokens):
    total_tokens = crystal_tokens + text_tokens
    return {
        "L_crystal_denoise": ce_sum / crystal_tokens if crystal_tokens else None,
        "L_denoise_current": ce_sum / total_tokens if total_tokens else None,
    }


def fixed_complementary_masks(crystal_tokens, seed, sample_id):
    import hashlib
    import torch

    key = f"{seed}\0{sample_id}".encode("utf-8")
    seed_value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed_value)
    first = torch.zeros(crystal_tokens, dtype=torch.bool)
    first[torch.randperm(crystal_tokens, generator=generator)[: crystal_tokens // 2]] = True
    return first, ~first


def complementary_denoising_ce(logits_pair, targets, masks, text_tokens):
    import torch
    import torch.nn.functional as F

    logits_a, logits_b = logits_pair
    targets = targets.reshape(-1)
    mask_a, mask_b = (
        mask.to(device=targets.device, dtype=torch.bool).reshape(-1) for mask in masks
    )
    masks = (mask_a, mask_b)
    if any(mask.numel() != targets.numel() for mask in masks):
        raise ValueError("complementary masks must match the crystal targets")
    coverage = masks[0].to(torch.int8) + masks[1].to(torch.int8)
    if not torch.all(coverage == 1).item():
        raise ValueError("complementary masks must cover each crystal target exactly once")

    ce_sum = sum(
        float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1])[mask],
                targets[mask],
                reduction="sum",
            ).item()
        )
        for logits, mask in ((logits_a, mask_a), (logits_b, mask_b))
    )
    crystal_tokens = int(targets.numel())
    return {
        **denoising_metrics(ce_sum, crystal_tokens, text_tokens),
        "ce_sum": ce_sum,
        "crystal_tokens": crystal_tokens,
        "text_tokens": text_tokens,
        "total_tokens": crystal_tokens + text_tokens,
    }
