"""Manifest-verified task rows and complete frozen PLaID reference banks."""
import hashlib
import json
from pathlib import Path

from .legacy import extract_crystal, load_legacy_audit


_SPLITS = {
    "training": ("training", "train"),
    "validation": ("validation", "val"),
    "test": ("test",),
}
_COUNT_KEY = {"training": "train", "validation": "val", "test": "test"}


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row in {path} at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL in {path} at line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object in {path} at line {line_number}")
            rows.append(row)
    return rows


def _manifest(dataset_dir):
    path = Path(dataset_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {path}")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("artifacts"), dict):
        raise ValueError(f"manifest has no artifact map: {path}")
    return path, manifest, hashlib.sha256(raw).hexdigest()


def _artifact_record(manifest, suffix):
    matches = [
        (key, value) for key, value in manifest["artifacts"].items()
        if key.replace("\\", "/") == suffix
        or key.replace("\\", "/").endswith("/" + suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"manifest must contain exactly one artifact ending in {suffix}; found {len(matches)}")
    key, record = matches[0]
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"manifest artifact lacks sha256: {key}")
    return key, record


def _expected_count(manifest, task, split):
    counts = manifest.get("counts_per_task")
    if not isinstance(counts, dict) or type(counts.get(_COUNT_KEY[split])) is not int:
        raise ValueError(f"manifest counts_per_task has no count for {task}/{split}")
    return counts[_COUNT_KEY[split]]


def _task_path(root, task, split):
    path = root / task / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"frozen {task}/{split} JSONL not found: {path}")
    return path


def _verify_file(path, manifest, suffix, *, task, split):
    key, record = _artifact_record(manifest, suffix)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise ValueError(f"sha256 mismatch for {task}/{split}: {path} (manifest key {key})")
    return record


def load_task_rows(dataset_dir, task="understanding", split="test"):
    """Return raw frozen task JSONL rows after manifest hash and count checks."""
    if task not in ("understanding", "design"):
        raise ValueError("task must be 'understanding' or 'design'")
    if split not in _SPLITS:
        raise ValueError("split must be 'training', 'validation', or 'test'")
    root = Path(dataset_dir)
    _, manifest, _ = _manifest(root)
    path = _task_path(root, task, split)
    _verify_file(
        path, manifest, f"{task}/{split}.jsonl", task=task, split=split,
    )
    rows = read_jsonl(path)
    expected = _expected_count(manifest, task, split)
    if len(rows) != expected:
        raise ValueError(f"{task}/{split} row count mismatch: expected {expected}, found {len(rows)}")
    return rows


def _metadata_path(root, split):
    path = root / "metadata" / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"frozen metadata JSONL not found: {path}")
    return path


def load_split_rows(dataset_dir, split):
    """Return task rows paired positionally with their identity and core facts."""
    rows = load_task_rows(dataset_dir, "understanding", split)
    root = Path(dataset_dir)
    _, manifest, _ = _manifest(root)
    metadata_path = _metadata_path(root, split)
    _verify_file(
        metadata_path, manifest, f"metadata/{split}.jsonl",
        task="understanding metadata", split=split,
    )
    metadata = read_jsonl(metadata_path)
    expected = _expected_count(manifest, "understanding", split)
    if len(metadata) != expected or len(metadata) != len(rows):
        raise ValueError(
            f"understanding/{split} metadata count mismatch: data={len(rows)}, metadata={len(metadata)}, expected={expected}"
        )

    joined = []
    seen = set()
    source_split = _SPLITS[split][-1]
    for index, (row, meta) in enumerate(zip(rows, metadata)):
        if type(meta.get("row_index")) is not int or meta["row_index"] != index:
            raise ValueError(
                f"understanding/{split} metadata order mismatch at row_index={index}: "
                f"found {meta.get('row_index')!r}"
            )
        if meta.get("source_split") != source_split:
            raise ValueError(
                f"understanding/{split} metadata split mismatch at row_index={index}: "
                f"found {meta.get('source_split')!r}"
            )
        material_id = meta.get("material_id")
        if not isinstance(material_id, str) or not material_id:
            raise ValueError(f"missing material_id in {split} metadata row_index={index}")
        if material_id in seen:
            raise ValueError(f"duplicate material_id {material_id} in {split} metadata")
        seen.add(material_id)
        facts = meta.get("core_facts")
        if not isinstance(facts, dict) or any(
            name not in facts for name in (
                "cell_formula", "space_group_number", "cell_atom_count", "independent_orbit_count",
            )
        ):
            raise ValueError(f"incomplete core_facts for {material_id} at row_index={index}")
        if not all(isinstance(row.get(name), str) for name in ("instruction", "input", "output")):
            raise ValueError(f"invalid task row for {material_id} at row_index={index}")
        joined.append({**row, "metadata": meta})
    return joined


def decode_reference_rows(rows):
    """Decode every raw structure; one failure invalidates the complete bank."""
    parser = load_legacy_audit().parse_fn_wyckoff
    structures = []
    for index, row in enumerate(rows):
        meta = row.get("metadata", {})
        material_id = meta.get("material_id", f"row-{index}")
        try:
            text = row["input"]
            if "<CRYSTAL_START>" not in text or "<CRYSTAL_END>" not in text:
                raise ValueError("missing PLaID crystal markers")
            body = extract_crystal(text)
            structure = parser(body, to_cif=False)
            if structure is None:
                raise ValueError("PLaID parser returned no structure")
        except Exception as error:
            raise ValueError(
                f"failed to decode frozen reference {material_id} at row_index={index}: "
                f"{type(error).__name__}: {error}"
            ) from error
        structures.append(structure)
    return structures


def load_reference_banks(dataset_dir, extra_training_references=None):
    training_rows = load_split_rows(dataset_dir, "training")
    test_rows = load_split_rows(dataset_dir, "test")
    training = decode_reference_rows(training_rows)
    test = decode_reference_rows(test_rows)
    if extra_training_references is not None:
        path = Path(extra_training_references)
        parser = load_legacy_audit().parse_fn_wyckoff
        for index, row in enumerate(read_jsonl(path)):
            value = row.get("input")
            if not isinstance(value, str):
                raise ValueError(f"extra reference row {index} in {path} needs an input string")
            try:
                structure = parser(extract_crystal(value), to_cif=False)
                if structure is None:
                    raise ValueError("PLaID parser returned no structure")
            except Exception as error:
                raise ValueError(
                    f"failed to decode extra training reference row {index} in {path}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            training.append(structure)
    return training, test

