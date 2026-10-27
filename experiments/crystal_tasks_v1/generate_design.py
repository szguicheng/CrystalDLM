"""Generate fixed-protocol AR design attempts from a saved HF PEFT adapter."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from .configuration import protocol, task_data


CHAT_EOS_ID = 151645
OVERFIT_COUNT = 32
_DESIGN_INSTRUCTION = re.compile(
    r"Generate a crystal with space group number (\d+) and energy above hull no greater than ([\d.]+) eV/atom\."
)
ROOT = Path(__file__).resolve().parents[2]


def design_instruction(request):
    space_group = request.get("space_group")
    ehull_upper = request.get("ehull_upper")
    if type(space_group) is not int or not 1 <= space_group <= 230:
        raise ValueError("request.space_group must be an integer from 1 to 230")
    if type(ehull_upper) not in (int, float) or not math.isfinite(ehull_upper):
        raise ValueError("request.ehull_upper must be finite")
    return (
        f"Generate a crystal with space group number {space_group} and energy above hull "
        f"no greater than {ehull_upper} eV/atom."
    )


def format_user_prompt(instruction, input_text=""):
    if not isinstance(instruction, str) or not isinstance(input_text, str):
        raise ValueError("instruction and input must be strings")
    return instruction + ("\n" + input_text if input_text else "")


def encode_prompt(instruction, input_text, processor):
    messages = [
        {"role": "user", "content": format_user_prompt(instruction, input_text)},
        {"role": "assistant", "content": ""},
    ]
    source, _ = processor.template.encode_oneturn(processor.tokenizer, messages)
    if len(source) > processor.data_args.cutoff_len:
        raise ValueError(f"Prompt requires {len(source)} tokens; cutoff is {processor.data_args.cutoff_len}")
    return source


def derive_attempt_seed(seed, condition_id, attempt_index):
    payload = json.dumps(
        [seed, condition_id, attempt_index], separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def shard_rows(rows, rank, world_size):
    if type(rank) is not int or type(world_size) is not int or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size), with world_size >= 1")
    return list(rows[rank::world_size])


def decode_generated_tokens(tokenizer, token_ids):
    token_ids = [int(token) for token in token_ids]
    raw_count = len(token_ids)
    trailing_padding = 0
    pad_id = tokenizer.pad_token_id
    while pad_id is not None and pad_id != CHAT_EOS_ID and token_ids and token_ids[-1] == pad_id:
        token_ids.pop()
        trailing_padding += 1
    terminal_eos = bool(token_ids and token_ids[-1] == CHAT_EOS_ID)
    if terminal_eos:
        token_ids.pop()
    text = tokenizer.decode(
        token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )
    return text, {
        "raw_generated_token_count": raw_count,
        "generated_token_count": len(token_ids),
        "terminal_chat_eos": terminal_eos,
        "terminal_stop_token_id": CHAT_EOS_ID if terminal_eos else None,
        "trailing_chat_padding_tokens": trailing_padding,
    }


def _request_from_instruction(instruction):
    match = _DESIGN_INSTRUCTION.fullmatch(instruction)
    if match is None:
        raise ValueError(f"design instruction differs from the frozen request format: {instruction!r}")
    request = {"space_group": int(match[1]), "ehull_upper": float(match[2])}
    design_instruction(request)
    return request


def _validate_plan(rows):
    if not rows:
        raise ValueError("attempt plan is empty")
    attempt_ids, condition_requests, identities = set(), {}, set()
    for index, row in enumerate(rows):
        if not isinstance(row.get("attempt_id"), str) or not row["attempt_id"]:
            raise ValueError(f"attempt plan row {index} needs a nonempty attempt_id")
        if row["attempt_id"] in attempt_ids:
            raise ValueError(f"duplicate attempt_id in plan: {row['attempt_id']}")
        attempt_ids.add(row["attempt_id"])
        condition_id = row.get("condition_id")
        seed, attempt_index = row.get("seed"), row.get("attempt_index")
        if not isinstance(condition_id, str) or not condition_id:
            raise ValueError(f"attempt plan row {index} needs a nonempty condition_id")
        if type(seed) is not int or type(attempt_index) is not int or attempt_index < 0:
            raise ValueError(f"attempt plan row {index} needs integer seed and nonnegative attempt_index")
        request = row.get("request")
        if not isinstance(request, dict) or set(request) != {"space_group", "ehull_upper"}:
            raise ValueError(f"attempt plan row {index} has an invalid request")
        design_instruction(request)
        condition_key = (request["space_group"], float(request["ehull_upper"]))
        if condition_id in condition_requests and condition_requests[condition_id] != condition_key:
            raise ValueError(f"condition_id {condition_id!r} maps to more than one request")
        condition_requests[condition_id] = condition_key
        identity = (condition_id, seed, attempt_index)
        if identity in identities:
            raise ValueError(f"duplicate condition/seed/attempt index in plan: {identity}")
        identities.add(identity)
    return rows


def _receipt_map(receipts, label):
    indexed = {}
    for receipt in receipts:
        attempt_id = receipt["attempt_id"]
        if attempt_id in indexed:
            raise ValueError(f"duplicate attempt_id in {label}: {attempt_id}")
        indexed[attempt_id] = receipt
    return indexed


def grade_overfit(predictions, generated_receipts, gold_receipts, nll_summary):
    from .evaluation.legacy import extract_crystal

    predicted = {row["material_id"]: row for row in predictions}
    if len(predicted) != len(predictions):
        raise ValueError("duplicate material_id in overfit predictions")
    generated = _receipt_map(generated_receipts, "generated receipts")
    gold = _receipt_map(gold_receipts, "gold receipts")
    expected_ids = set(predicted)
    if generated.keys() != expected_ids or gold.keys() != expected_ids:
        raise ValueError("overfit prediction and structure receipt identities differ")

    exact_full = exact_body = strict_valid = requested_sg = strict_and_sg = gold_valid = 0
    invalid_gold = []
    for material_id, row in predicted.items():
        response, reference = row["response"], row["reference_output"]
        exact_full += response.strip() == reference.strip()
        try:
            exact_body += extract_crystal(response).strip() == extract_crystal(reference).strip()
        except ValueError:
            pass
        receipt = generated[material_id]
        valid = receipt["strict_valid"] is True
        sg_match = receipt["requested_sg_match"] is True
        strict_valid += valid
        requested_sg += sg_match
        strict_and_sg += valid and sg_match
        if gold[material_id]["strict_valid"] is True:
            gold_valid += 1
        else:
            invalid_gold.append(material_id)

    n = len(predictions)
    nll_mean = nll_summary.get("nll_mean")
    nll_pass = type(nll_mean) in (int, float) and math.isfinite(nll_mean) and nll_mean <= 0.05
    distinct_conditions = len({
        (row["request"]["space_group"], float(row["request"]["ehull_upper"]))
        for row in predictions
    })
    return {
        "exact_full_output": {"k": exact_full, "n": n},
        "exact_crystal_body": {"k": exact_body, "n": n},
        "strict_valid": {"k": strict_valid, "n": n},
        "requested_sg_match": {"k": requested_sg, "n": n},
        "strict_valid_requested_sg": {"k": strict_and_sg, "n": n},
        "gold_strict_valid": {"k": gold_valid, "n": n},
        "gold_invalid_material_ids": invalid_gold,
        "nll": nll_summary,
        "nll_pass": nll_pass,
        "distinct_conditions": distinct_conditions,
        "condition_goal": "32 distinct (space_group, ehull_upper) conditions; selection recorded by training run",
        "passed": bool(
            n == OVERFIT_COUNT and exact_full == OVERFIT_COUNT and exact_body == OVERFIT_COUNT
            and strict_valid == OVERFIT_COUNT and requested_sg == OVERFIT_COUNT and nll_pass
        ),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _write_jsonl(stream, row):
    stream.write(json.dumps(row, allow_nan=False, default=str) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def persist_attempt_or_fail(stream, attempts, attempt):
    _write_jsonl(stream, attempt)
    attempts.append(attempt)
    if attempt["generation_error"] is not None:
        raise RuntimeError(f"Generation failed for {attempt['attempt_id']}: {attempt['generation_error']}")


def _cuda_device():
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to exactly one explicitly selected GPU")
    return visible


def _training_items(dataset_dir, training_run_dir, seed):
    from .evaluation.references import load_split_rows, load_task_rows

    run_meta_path = Path(training_run_dir) / "run_meta.json"
    training_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    indices = training_meta.get("selected_training_indices")
    material_ids = training_meta.get("selected_training_ids")
    if (
        not isinstance(indices, list) or len(indices) != OVERFIT_COUNT
        or not isinstance(material_ids, list) or len(material_ids) != OVERFIT_COUNT
        or any(type(index) is not int for index in indices)
        or len(set(indices)) != OVERFIT_COUNT or len(set(material_ids)) != OVERFIT_COUNT
    ):
        raise ValueError("training run metadata must contain the saved 32 selected indices and IDs")
    design_rows = load_task_rows(dataset_dir, "design", "training")
    reference_rows = load_split_rows(dataset_dir, "training")
    if len(design_rows) != len(reference_rows):
        raise ValueError("design rows and shared Understanding metadata have different training counts")
    items, conditions = [], set()
    for index, material_id in zip(indices, material_ids, strict=True):
        if not 0 <= index < len(design_rows):
            raise ValueError(f"selected training index out of range: {index}")
        metadata = reference_rows[index]["metadata"]
        if metadata["material_id"] != material_id:
            raise ValueError(
                f"training run selected ID mismatch at index {index}: "
                f"run={material_id}, shared metadata={metadata['material_id']}"
            )
        row = {**design_rows[index], "metadata": metadata}
        request = _request_from_instruction(row["instruction"])
        conditions.add((request["space_group"], request["ehull_upper"]))
        items.append({
            "attempt_id": material_id,
            "condition_id": f"overfit-{index}",
            "seed": seed,
            "attempt_index": 0,
            "request": request,
            "row": row,
        })
    if len(conditions) != OVERFIT_COUNT:
        raise ValueError("saved AR-design overfit selection does not contain 32 distinct requests")
    return items, training_meta, _sha256(run_meta_path)


def _sample_items(attempt_plan):
    from .evaluation.references import read_jsonl

    rows = _validate_plan(read_jsonl(attempt_plan))
    return [
        {
            **row,
            "row": {"instruction": design_instruction(row["request"]), "input": ""},
        }
        for row in rows
    ], _sha256(attempt_plan)


def _source_hashes():
    script_path = Path(__file__).resolve()
    package_dir = script_path.parent
    source_paths = [
        script_path, package_dir / "configuration.py", package_dir / "tokenization.py",
        package_dir / "train_ar.py", package_dir / "evaluate_ar.py",
        package_dir / "evaluation/design.py", package_dir / "evaluation/nll.py",
        package_dir / "evaluation/references.py", package_dir / "evaluation/legacy.py",
        package_dir / "evaluation/cli.py", package_dir / "evaluation/plan.py",
    ]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in source_paths}


def _base_model(p, checkpoint):
    adapter_config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    if Path(adapter_config["base_model_name_or_path"]).resolve() != Path(p["model_path"]).resolve():
        raise ValueError("Checkpoint base differs from the frozen protocol model path")
    return {
        "name": p["model_name"], "revision": p["model_revision"],
        "path": p["model_path"], "adapter_sha256": _directory_sha256(checkpoint),
    }


def _generate_one(item, processor, model, torch, mode, sampling, max_new_tokens):
    started = time.perf_counter()
    seed = derive_attempt_seed(item["seed"], item["condition_id"], item["attempt_index"]) if mode == "sample" else item["seed"]
    try:
        prompt = encode_prompt(item["row"]["instruction"], item["row"].get("input", ""), processor)
        input_ids = torch.tensor([prompt], device="cuda:0")
        options = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "eos_token_id": CHAT_EOS_ID,
            "pad_token_id": processor.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if mode == "sample":
            options.update(sampling)
        else:
            options["do_sample"] = False
        torch.manual_seed(seed)
        with torch.inference_mode():
            sequence = model.generate(**options)[0, input_ids.shape[1]:].tolist()
        response, stop = decode_generated_tokens(processor.tokenizer, sequence)
        stop["stop_reason"] = (
            "chat_eos" if stop["terminal_chat_eos"]
            else "max_new_tokens" if stop["generated_token_count"] >= max_new_tokens
            else "returned_without_chat_eos"
        )
        return response, seed, stop, time.perf_counter() - started, None
    except Exception as error:
        return "", seed, {
            "raw_generated_token_count": 0, "generated_token_count": 0,
            "terminal_chat_eos": False, "terminal_stop_token_id": None,
            "trailing_chat_padding_tokens": 0, "stop_reason": "generation_error",
        }, time.perf_counter() - started, f"{type(error).__name__}: {error}"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("overfit", "sample"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-run-dir", type=Path)
    parser.add_argument("--attempt-plan", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    return parser


def main(argv=None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from .evaluation.cli import _environment, _git_metadata
    from .evaluation.nll import score_model
    from .tokenization import encode_row, make_processor

    args = build_parser().parse_args(argv)
    if args.mode == "overfit" and (args.training_run_dir is None or args.attempt_plan is not None):
        raise ValueError("overfit mode requires --training-run-dir and does not use --attempt-plan")
    if args.mode == "sample" and (args.attempt_plan is None or args.training_run_dir is not None):
        raise ValueError("sample mode requires --attempt-plan and does not use --training-run-dir")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.output}")
    visible = _cuda_device()
    if args.world_size < 1 or args.world_size > 3 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size), with 1 <= world_size <= 3")

    p = protocol()
    sampling = {
        "do_sample": True,
        **{name: p["evaluation"]["ar_design_sampling"][name] for name in ("temperature", "top_p", "top_k")},
    }
    data = task_data("ar_design")
    dataset_dir = Path(data["dataset_dir"])
    manifest_sha256 = _sha256(dataset_dir / "manifest.json")
    if manifest_sha256 != data["manifest_sha256"]:
        raise ValueError("AR-design dataset manifest differs from configuration.task_data")
    checkpoint = args.checkpoint.resolve()
    base = _base_model(p, checkpoint)
    if args.mode == "overfit":
        items, training_meta, training_meta_sha256 = _training_items(
            dataset_dir, args.training_run_dir, p["training"]["seed"],
        )
        plan_sha256 = None
        plan_path = None
    else:
        items, plan_sha256 = _sample_items(args.attempt_plan)
        training_meta = None
        training_meta_sha256 = None
        plan_path = str(args.attempt_plan.resolve())
    shard = shard_rows(items, args.rank, args.world_size)
    if not shard:
        raise ValueError(f"rank {args.rank} has no assigned rows from {len(items)} planned rows")

    args.output.mkdir(parents=True, exist_ok=False)
    meta = {
        "schema": 1, "status": "running", "mode": args.mode,
        "command": sys.argv if argv is None else [sys.argv[0], *argv],
        "config": {
            "checkpoint": str(checkpoint), "output": str(args.output.resolve()),
            "training_run_dir": str(args.training_run_dir.resolve()) if args.training_run_dir else None,
            "attempt_plan": plan_path, "rank": args.rank, "world_size": args.world_size,
        },
        "worker": {"rank": args.rank, "world_size": args.world_size, "cuda_visible_devices": visible},
        "git": _git_metadata(), "base_model": base,
        "dataset_dir": str(dataset_dir), "dataset_manifest_sha256": manifest_sha256,
        "attempt_plan_sha256": plan_sha256, "plan_attempt_count": len(items),
        "shard_attempt_ids": [item["attempt_id"] for item in shard],
        "training_run_meta_sha256": training_meta_sha256,
        "training_selected_ids": training_meta.get("selected_training_ids") if training_meta else None,
        "environment": _environment(), "source_sha256": _source_hashes(),
        "config_sha256": {
            name: _sha256(ROOT / "experiments/crystal_tasks_v1/configs" / name)
            for name in ("shared.json", "ar_design.json")
        },
        "decoding": {
            "sampling": sampling if args.mode == "sample" else {"do_sample": False},
            "eos_token_id": CHAT_EOS_ID, "preserve_special_tokens": True,
        },
        "started_at": datetime.now(timezone.utc).isoformat(), "ended_at": None,
    }
    _write_json(args.output / "run_meta.json", meta)
    attempts, predictions = [], []

    try:
        processor = make_processor("ar_design")
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        if tokenizer.eos_token_id != CHAT_EOS_ID:
            raise ValueError(f"Expected chat EOS {CHAT_EOS_ID}, found {tokenizer.eos_token_id}")
        if args.mode == "overfit":
            prompt_check_row = items[0]["row"]
        else:
            from .evaluation.references import load_task_rows

            prompt_check_row = load_task_rows(dataset_dir, "design", "training")[0]
        encoded_check = encode_row(prompt_check_row, processor, "ar_design")
        prompt_length = sum(label == -100 for label in encoded_check["labels"])
        prompt_source = encode_prompt(
            prompt_check_row["instruction"], prompt_check_row["input"], processor,
        )
        if prompt_source != encoded_check["input_ids"][:prompt_length]:
            raise ValueError("LLaMA Factory inference source differs from the masked training prompt")
        meta["prompt_contract_verified"] = True
        _write_json(args.output / "run_meta.json", meta)
        model_base = AutoModelForCausalLM.from_pretrained(
            p["model_path"], local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(model_base, checkpoint, is_trainable=False).to("cuda:0").eval()
        cutoff_len = processor.data_args.cutoff_len
        meta["max_new_tokens"] = cutoff_len
        meta["decoding"]["sampling"] = {**meta["decoding"]["sampling"], "max_new_tokens": cutoff_len}
        _write_json(args.output / "run_meta.json", meta)
        if args.mode == "sample":
            print(f"Sampling settings: {json.dumps(sampling, sort_keys=True)}", flush=True)

        with (args.output / "attempts.jsonl").open("x", encoding="utf-8") as attempt_stream:
            prediction_stream = (
                (args.output / "predictions.jsonl").open("x", encoding="utf-8")
                if args.mode == "overfit" else None
            )
            try:
                for index, item in enumerate(shard, 1):
                    response, actual_seed, stop, elapsed, error = _generate_one(
                        item, processor, model, torch, args.mode, sampling, cutoff_len,
                    )
                    attempt = {
                        "attempt_id": item["attempt_id"], "condition_id": item["condition_id"],
                        "seed": item["seed"], "attempt_index": item["attempt_index"],
                        "request": item["request"], "output": response,
                        "actual_attempt_seed": actual_seed, "elapsed_seconds": elapsed,
                        **stop, "generation_error": error,
                    }
                    persist_attempt_or_fail(attempt_stream, attempts, attempt)
                    if prediction_stream is not None:
                        reference = item["row"]["output"]
                        prediction = {
                            "material_id": item["attempt_id"], "seed": item["seed"],
                            "request": item["request"], "response": response,
                            "reference_output": reference,
                            "exact_full_output": response.strip() == reference.strip(),
                        }
                        try:
                            from .evaluation.legacy import extract_crystal
                            prediction["exact_crystal_body"] = (
                                extract_crystal(response).strip() == extract_crystal(reference).strip()
                            )
                        except ValueError:
                            prediction["exact_crystal_body"] = False
                        _write_jsonl(prediction_stream, prediction)
                        predictions.append(prediction)
                    print(f"GENERATED {index}/{len(shard)}", flush=True)
            finally:
                if prediction_stream is not None:
                    prediction_stream.close()

        if args.mode == "overfit":
            from .evaluation.design import evaluate_design

            rows = [item["row"] for item in shard]
            nll_summary, nll_receipts = score_model(model, rows, processor, "ar_design", "cuda:0")
            generated_eval = evaluate_design(attempts, training_references=[], test_references=[])
            gold_attempts = [
                {**attempt, "output": item["row"]["output"]}
                for attempt, item in zip(attempts, shard, strict=True)
            ]
            gold_eval = evaluate_design(gold_attempts, training_references=[], test_references=[])
            grade = grade_overfit(
                predictions, generated_eval["receipts"], gold_eval["receipts"], nll_summary,
            )
            if grade["gold_invalid_material_ids"]:
                print(
                    "WARNING: exact PLaID gold is strictly invalid for "
                    + ", ".join(grade["gold_invalid_material_ids"]),
                    file=sys.stderr, flush=True,
                )
            summary = {
                **grade,
                "generated_structure_counts": generated_eval["summary"]["rates"],
                "nll_receipts": len(nll_receipts),
                "shard": {"rank": args.rank, "world_size": args.world_size, "rows": len(shard)},
                "overfit_passed": grade["passed"] if args.world_size == 1 else None,
            }
            with (args.output / "nll-receipts.jsonl").open("x", encoding="utf-8") as stream:
                for item, receipt in zip(shard, nll_receipts, strict=True):
                    _write_jsonl(stream, {"material_id": item["attempt_id"], **receipt})
            _write_json(args.output / "summary.json", summary)
        else:
            summary = {
                "attempt_count": len(attempts), "plan_attempt_count": len(items),
                "attempt_ids": [attempt["attempt_id"] for attempt in attempts],
                "shard": {"rank": args.rank, "world_size": args.world_size},
            }
            _write_json(args.output / "summary.json", summary)

        meta.update(
            status="complete", ended_at=datetime.now(timezone.utc).isoformat(),
            generated_attempt_count=len(attempts),
            overfit_passed=summary.get("overfit_passed") if args.mode == "overfit" else None,
        )
        _write_json(args.output / "run_meta.json", meta)
        if args.mode == "overfit" and args.world_size == 1 and not grade["passed"]:
            raise SystemExit("Overfit gate did not pass; full training must not start")
    except SystemExit:
        raise
    except BaseException as error:
        meta.update(
            status="failed", ended_at=datetime.now(timezone.utc).isoformat(),
            generated_attempt_count=len(attempts), error=f"{type(error).__name__}: {error}",
        )
        _write_json(args.output / "run_meta.json", meta)
        raise


if __name__ == "__main__":
    main()
