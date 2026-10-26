"""Measure full chat lengths with the existing tokenizer; never load weights."""
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer
from prepare import DATA, ROOT, rows, write_json

model_path = ROOT / "models/Qwen3-8B-b968826d9c46"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
registry = json.loads((DATA / "dataset_info.json").read_text())
report = {"tokenizer": str(model_path), "enable_thinking": False,
          "scope": "native tokenizer chat template; not an executed LLaMA-Factory training pipeline",
          "raw_jsonl_contains_model_eos": False, "datasets": {}}
for name, config in registry.items():
    lengths = []
    for row in rows(DATA / config["file_name"]):
        prompt = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
        ids = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["output"]}],
            tokenize=True, enable_thinking=False)
        lengths.append(len(ids))
    report["datasets"][name] = {"rows": len(lengths), "min": min(lengths),
        "median": statistics.median(lengths), "max": max(lengths),
        "above_2048": sum(n > 2048 for n in lengths)}
    print(name, report["datasets"][name], flush=True)
write_json(DATA / "token-length-audit.json", report)
