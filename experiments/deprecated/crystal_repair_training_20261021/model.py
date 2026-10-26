"""Repair, original generation, and factual-reading objectives on one model."""
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = ROOT / "experiments/flare_crystal_qwen3_8b_20261019/model.py"
_spec = importlib.util.spec_from_file_location("repair_flare_pilot_objective", PILOT_PATH)
PILOT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PILOT)


def objective_tasks(model, tok, repair, original_tasks, generator):
    """Return .5 R_joint + .25 G_joint + .25 Q_AR using the frozen FLARE loss."""
    if len(original_tasks) != 2 or [item.get("task") for item in original_tasks] != ["G", "Q"]:
        raise ValueError("original_tasks must contain the frozen G then Q examples")
    generation, reading = original_tasks
    repair_loss, repair_stats = PILOT.objective(model, tok, repair, generator, "flare")
    generation_loss, generation_stats = PILOT.objective(
        model, tok, generation, generator, "flare")
    reading_loss, _ = model.loss([reading], tok, "ar", generator)
    loss = .5 * repair_loss + .25 * generation_loss + .25 * reading_loss
    return loss, {
        "repair_joint": float(repair_loss.detach()),
        "generation_joint": float(generation_loss.detach()),
        "reading_ar": float(reading_loss.detach()),
        "repair_clean_ar": repair_stats["clean_ar"],
        "repair_denoise": repair_stats["denoise"],
        "generation_clean_ar": generation_stats["clean_ar"],
        "generation_denoise": generation_stats["denoise"],
    }


def teacher_forced_start_correct(model, tok, example):
    """Check whether the correction lead predicts its final START token."""
    import torch

    prefix = tok.encode(example["prefix"], add_special_tokens=False)
    lead = tok.encode(example["lead"], add_special_tokens=False)
    crystal = tok.encode(example["crystal"], add_special_tokens=False) + [model.end_id]
    if not lead or lead[-1] != model.start_id:
        raise ValueError("repair lead must end in CRYSTAL_START")
    clean = prefix + lead + crystal
    crystal_start = len(prefix) + len(lead)
    n = len(clean)
    allowed = [[column <= row for column in range(n)] for row in range(n)]
    hidden = PILOT.forward_hidden(model, clean, allowed, list(range(n)), crystal_start)
    preceding = crystal_start - 2
    return int(model.base.lm_head(hidden[preceding]).float().argmax().item()) == model.start_id


def json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def trainable_state(model):
    return {name: param.detach().cpu().contiguous().clone()
            for name, param in model.named_parameters() if param.requires_grad}


def validate_checkpoint_identity(saved, expected_arm="repair", parent_checkpoint_sha256=None,
                                 parent_trainable_sha256=None, metadata=None,
                                 world_size=None, allow_partial=False):
    if not isinstance(saved, dict) or saved.get("schema") != 1 or saved.get("arm") != expected_arm:
        raise ValueError("checkpoint schema or arm differs")
    saved_metadata = saved.get("metadata")
    if (not isinstance(saved_metadata, dict)
            or saved.get("metadata_sha256") != json_sha256(saved_metadata)):
        raise ValueError("checkpoint metadata is missing or failed its hash check")
    required_metadata = {
        "config", "config_sha256", "arm", "training_world_size", "source_identity",
        "pure_ar_trainable_sha256", "parent_trainable_sha256", "parent_checkpoint_sha256",
        "parent_checkpoint_metadata_sha256", "parent_checkpoint_path", "runtime_versions",
        "dataset_hashes", "train_material_ids", "validation_material_ids",
        "epoch_order_sha256", "validation_ids_sha256", "source_sha256",
    }
    if not required_metadata <= saved_metadata.keys():
        raise ValueError("checkpoint metadata is incomplete")
    config = saved_metadata.get("config")
    if (not isinstance(config, dict)
            or saved_metadata.get("config_sha256") != json_sha256(config)):
        raise ValueError("checkpoint config is missing or failed its hash check")
    if not {"total_updates", "materials_per_update", "train_materials"} <= config.keys():
        raise ValueError("checkpoint update plan is incomplete")
    if saved_metadata.get("arm") != expected_arm:
        raise ValueError("checkpoint metadata arm differs")
    if parent_checkpoint_sha256 is not None and saved.get(
            "parent_checkpoint_sha256") != parent_checkpoint_sha256:
        raise ValueError("checkpoint parent file hash differs")
    if parent_trainable_sha256 is not None and saved.get(
            "parent_trainable_sha256") != parent_trainable_sha256:
        raise ValueError("checkpoint parent trainable state differs")
    if saved_metadata.get("parent_checkpoint_sha256") != saved.get("parent_checkpoint_sha256"):
        raise ValueError("checkpoint parent file hash differs from metadata")
    if saved_metadata.get("parent_trainable_sha256") != saved.get("parent_trainable_sha256"):
        raise ValueError("checkpoint parent state hash differs from metadata")
    if metadata is not None and saved_metadata != metadata:
        raise ValueError("checkpoint metadata differs from current code, data, or protocol")
    completed = saved.get("completed_updates", 0)
    total = config.get("total_updates", 0)
    if (type(total) is not int or type(config["materials_per_update"]) is not int
            or type(config["train_materials"]) is not int or type(completed) is not int
            or total < 1 or config["materials_per_update"] < 1 or config["train_materials"] < 1
            or saved.get("total_updates") != total
            or not 0 < completed <= total
            or (not allow_partial and completed != total)):
        raise ValueError("checkpoint is partial or has an invalid update count")
    expected_next = min(completed * config["materials_per_update"], config["train_materials"])
    if saved.get("next_material_index") != expected_next:
        raise ValueError("checkpoint next material index differs")
    saved_world_size = saved.get("world_size")
    if type(saved_world_size) is not int or saved_world_size not in (1, 3):
        raise ValueError("checkpoint world size is invalid")
    if saved_metadata.get("training_world_size") != saved_world_size:
        raise ValueError("checkpoint metadata world size differs")
    if world_size is not None and saved_world_size != world_size:
        raise ValueError("checkpoint world size differs")
    if not isinstance(saved.get("rng_states"), list) or len(saved["rng_states"]) != saved_world_size:
        raise ValueError("checkpoint does not contain RNG state for every training rank")


def load_checkpoint(model, path, train, expected_arm="repair", *,
                    parent_checkpoint_sha256=None, parent_trainable_sha256=None,
                    metadata=None, world_size=None, allow_partial=False):
    """Restore trainable repair state after verifying its parent, metadata, and tensors."""
    torch = train.torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_identity(saved, expected_arm, parent_checkpoint_sha256,
                                 parent_trainable_sha256, metadata, world_size, allow_partial)
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    state = saved.get("trainable_state", {})
    names = sorted(params)
    if saved.get("trainable_parameter_names") != names or sorted(state) != names:
        raise ValueError("checkpoint trainable parameter names differ")
    for name, param in params.items():
        if tuple(state[name].shape) != tuple(param.shape):
            raise ValueError(f"checkpoint trainable parameter shape differs: {name}")
    if train.state_sha256(state) != saved.get("trainable_state_sha256"):
        raise ValueError("checkpoint trainable state failed its hash check")
    with torch.no_grad():
        for name, param in params.items():
            param.copy_(state[name].to(device=param.device, dtype=param.dtype))
    if train.trainable_sha256(model) != saved["trainable_state_sha256"]:
        raise ValueError("restored trainable state failed its hash check")
    return saved
