"""Token-weighted training and validation metrics for the AR task trainers."""

import torch
import torch.nn.functional as F
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from transformers import TrainerCallback

from .configuration import protocol


_SEGMENTS = (("description", 1), ("crystal", 2), ("answer", 3))
_TASK_SEGMENTS = {
    "ar_understanding": {"answer"},
    "ar_design": {"description", "crystal"},
}
_CRYSTAL_START_ID, _CRYSTAL_END_ID = protocol()["lora"]["trainable_token_indices"][:2]


def _segment_ids_from_labels(labels, task):
    if task not in _TASK_SEGMENTS:
        raise ValueError(f"Unsupported crystal task: {task}")
    segments = torch.zeros_like(labels)
    valid = labels.ne(-100)
    if task == "ar_understanding":
        segments[valid] = 3
        return segments

    starts = labels.eq(_CRYSTAL_START_ID).sum(dim=1)
    ends = labels.eq(_CRYSTAL_END_ID).sum(dim=1)
    if not torch.all((starts == 1) & (ends == 1)).item():
        raise ValueError("Each design target must contain one crystal START and END token")

    is_start, is_end = labels.eq(_CRYSTAL_START_ID), labels.eq(_CRYSTAL_END_ID)
    if not torch.all(is_start.long().argmax(dim=1) < is_end.long().argmax(dim=1)).item():
        raise ValueError('Crystal END must follow START')
    in_crystal = is_start.cumsum(dim=1).gt(0) & (is_end.cumsum(dim=1) - is_end.long()).eq(0) & ~is_start
    segments[valid] = 1
    segments[valid & in_crystal] = 2
    return segments


def _batch_token_stats(logits, labels, task, segment_ids=None):
    """Return per-example CE sums/counts in description, crystal, answer order."""
    if task not in _TASK_SEGMENTS:
        raise ValueError(f"Unsupported crystal task: {task}")
    if segment_ids is None:
        segment_ids = _segment_ids_from_labels(labels, task)

    with torch.no_grad():
        shifted_logits = logits.detach()[:, :-1, :]
        shifted_labels = labels.detach()[:, 1:]
        shifted_segments = segment_ids.detach()[:, 1:]
        valid = shifted_labels.ne(-100)
        supervised = torch.zeros_like(valid)
        for name, segment_id in _SEGMENTS:
            if name in _TASK_SEGMENTS[task]:
                supervised |= shifted_segments.eq(segment_id)
        if not torch.equal(valid, supervised):
            raise ValueError("Every supervised label must belong to the selected task segments")

        batch_size = labels.shape[0]
        row_ids = torch.arange(batch_size, device=labels.device).unsqueeze(1).expand_as(valid)
        stats = torch.zeros((batch_size, len(_SEGMENTS) * 2), device=logits.device, dtype=torch.float64)
        for index, (name, segment_id) in enumerate(_SEGMENTS):
            if name not in _TASK_SEGMENTS[task]:
                continue
            mask = valid & shifted_segments.eq(segment_id)
            token_nll = F.cross_entropy(
                shifted_logits[mask].float(), shifted_labels[mask], reduction="none"
            )
            stats[:, index * 2].index_add_(0, row_ids[mask], token_nll.to(torch.float64))
            stats[:, index * 2 + 1] = mask.sum(dim=1).to(torch.float64)
        return stats


def _format_token_stats(stats, prefix, include_token_nll=False):
    if not torch.isfinite(stats).all().item():
        raise FloatingPointError("Non-finite token CE statistics")

    result = {}
    total_sum = stats.new_zeros(())
    total_count = stats.new_zeros(())
    for index, (name, _) in enumerate(_SEGMENTS):
        ce_sum, count = stats[index * 2], stats[index * 2 + 1]
        count_value = int(count.item())
        if count_value:
            result[f"{prefix}_{name}_ce"] = float((ce_sum / count).item())
        result[f"{prefix}_{name}_tokens"] = count_value
        total_sum += ce_sum
        total_count += count

    if include_token_nll:
        if total_count.item() == 0:
            raise ValueError("Cannot compute validation token NLL with no supervised tokens")
        result[f"{prefix}_token_nll"] = float((total_sum / total_count).item())
    return result


class CrystalTrainer(CustomSeq2SeqTrainer):
    """LLaMA-Factory trainer with token-weighted per-segment CE monitoring."""

    def __init__(self, *args, crystal_task, **kwargs):
        if crystal_task not in _TASK_SEGMENTS:
            raise ValueError(f"Unsupported crystal task: {crystal_task}")
        super().__init__(*args, **kwargs)
        self.crystal_task = crystal_task
        self._train_metric_totals = torch.zeros(len(_SEGMENTS) * 2, device=self.args.device, dtype=torch.float64)
        self._eval_metric_totals = torch.zeros_like(self._train_metric_totals)
        self._eval_metric_prefix = None
        self._last_eval_batch_stats = None

    def compute_loss(self, model, inputs, *args, **kwargs):
        labels = inputs["labels"].detach()
        call_kwargs = dict(kwargs)
        if args:
            return_outputs = args[0]
            call_args = (True, *args[1:])
            call_kwargs.pop("return_outputs", None)
        else:
            return_outputs = call_kwargs.get("return_outputs", False)
            call_args = args
            call_kwargs["return_outputs"] = True

        loss, outputs = super().compute_loss(model, inputs, *call_args, **call_kwargs)
        batch_stats = _batch_token_stats(outputs.logits, labels, self.crystal_task)
        if not (torch.isfinite(loss.detach()).all() & torch.isfinite(batch_stats).all()).item():
            raise FloatingPointError("Non-finite native loss or token CE")

        if self._eval_metric_prefix is not None:
            self._last_eval_batch_stats = batch_stats
        elif model.training:
            self._train_metric_totals.add_(batch_stats.sum(dim=0))
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        if self._eval_metric_prefix is None:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
            )

        self._last_eval_batch_stats = None
        result = super().prediction_step(
            model, inputs, prediction_loss_only=True, ignore_keys=ignore_keys, **gen_kwargs
        )
        if self._last_eval_batch_stats is None:
            raise RuntimeError("Evaluation did not produce token CE statistics")
        # gather_for_metrics trims Accelerate's final distributed-sampler padding rows.
        gathered_stats = self.accelerator.gather_for_metrics(self._last_eval_batch_stats)
        self._eval_metric_totals.add_(gathered_stats.sum(dim=0))
        return result

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):
        self._eval_metric_totals.zero_()
        self._eval_metric_prefix = metric_key_prefix
        prediction_loss_only = self.args.prediction_loss_only
        self.args.prediction_loss_only = True
        try:
            return super().evaluate(
                eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
                **gen_kwargs,
            )
        finally:
            self._eval_metric_prefix = None
            self._last_eval_batch_stats = None
            self._eval_metric_totals.zero_()
            self.args.prediction_loss_only = prediction_loss_only

    def log(self, logs, *args, **kwargs):
        if self._eval_metric_prefix is not None and f"{self._eval_metric_prefix}_loss" in logs:
            # Per-batch gather_for_metrics already made these totals identical on every rank.
            logs.update(_format_token_stats(self._eval_metric_totals, self._eval_metric_prefix, True))
            self._eval_metric_totals.zero_()
        elif "loss" in logs:
            totals = self.accelerator.reduce(self._train_metric_totals, reduction="sum")
            logs.update(_format_token_stats(totals, "train"))
            self._train_metric_totals.zero_()
        return super().log(logs, *args, **kwargs)


class EpochCheckpointCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        control.should_save = True
        return control
