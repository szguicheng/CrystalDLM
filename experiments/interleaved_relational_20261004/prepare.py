"""Build the fixed 24-case relational-CNL AR/diffusion pilot data."""
from collections import Counter
from copy import deepcopy
from decimal import Decimal
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RELATIONAL_ROOT = PROJECT / "experiments/naturecry_relational_20261004"
INLINE_ROOT = PROJECT / "experiments/naturecry_inline_20261003"
ROUNDTRIP_ROOT = PROJECT / "experiments/naturecry_roundtrip_20261001/run_001"
OLD_DATA_ROOT = PROJECT / "experiments/interleaved_ar_diffusion_20261002/data_002"
DATA_DIR = ROOT / "data_pilot_001"
TOKENIZER_PATH = OLD_DATA_ROOT / "tokenizer"

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
START, END, MASK, PAD = ("<CRYSTAL_START>", "<CRYSTAL_END>", "<CRYSTAL_MASK>", "<CRYSTAL_PAD>")
EOS = "<|im_end|>"
K, STEPS, MAXLEN = 2560, 32, 8192

if str(RELATIONAL_ROOT) not in sys.path:
    sys.path.insert(0, str(RELATIONAL_ROOT))
import relational_codec

if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
if __name__ == "__main__":
    sys.modules["prepare"] = sys.modules[__name__]

TABLE = relational_codec.geometry.verify_public_table()
source_facts_from_plan = relational_codec.oldcodec.source_facts_from_plan

TEMPLATES = {
    split: [
        (
            "Generate a crystal whose input cell has composition {formula} in space group {sg}.",
            "Read the current relational crystal text. Give its exact input-cell composition, space group number, total atoms, independent orbit count, and input-cell lattice lengths a, b, c.",
            "Multiply only the original input-cell lengths a, b, and c by {scale}. Keep angles, composition, atom count, space group, orbits, free parameters, frame, rounding, and corrections exactly unchanged.",
        )
    ]
    for split in ("train", "heldout")
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_json(path):
    return json.loads(path.read_text())


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def tokenizer():
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True)
    tok.add_special_tokens({"additional_special_tokens": [START, END, MASK, PAD]})
    assert tok.eos_token == EOS
    assert all(tok.convert_tokens_to_ids(token) != tok.unk_token_id for token in (START, END, MASK, PAD))
    return tok


TOK = tokenizer()


def _formula(composition):
    pieces = []
    for element, amount in sorted(composition.items()):
        value = float(amount)
        suffix = "" if value == 1 else str(int(value)) if value.is_integer() else format(value, ".15g")
        pieces.append(element + suffix)
    return "".join(pieces)


def _scaled_plan(plan, factor):
    target = deepcopy(plan)
    decimal_factor = Decimal(factor)
    target["lattice"][:3] = [float(Decimal(repr(length)) * decimal_factor) for length in plan["lattice"][:3]]
    return target


def edit_factor(direction):
    return {"plus": "1.05", "minus": "0.95"}[direction]


def _render_plan(plan):
    return relational_codec.encode(source_facts_from_plan(plan), table=TABLE)


def _condition(plan):
    return {"formula": _formula(plan["composition"]), "sg": int(plan["space_group_number"])}


def _structure(plan):
    return relational_codec.geometry.execute_geometry_plan(plan, table=TABLE)


def _scaled_reference(structure, factor):
    return Structure(
        Lattice(structure.lattice.matrix * float(Decimal(factor))),
        structure.species,
        structure.frac_coords,
        coords_are_cartesian=False,
    )


def tasks(row, split, direction, variant=0):
    gen, read, modify = TEMPLATES[split][variant % len(TEMPLATES[split])]
    condition = row["condition"]
    scale = edit_factor(direction)
    prefix = "User: " + gen.format(**condition) + "\nAssistant:"
    trigger = " I will generate the requested input-cell structure.\n" + START
    history = prefix + trigger + row["c1"] + END + "\nUser: " + read + "\nAssistant:"
    response1 = " " + score_answer(row["plans"]["source"]) + EOS
    editing = history + response1 + "\nUser: " + modify.format(scale=scale) + "\nAssistant:"
    edit_trigger = " I will apply only the requested scale edit.\n" + START
    reading = editing + edit_trigger + row[direction] + END + "\nUser: " + read + "\nAssistant:"
    return [
        dict(task="G", prefix=prefix, lead=trigger, crystal=row["c1"]),
        dict(task="Q1", prefix=history, target=response1),
        dict(task="E", prefix=editing, lead=edit_trigger, crystal=row[direction]),
        dict(task="Q2", prefix=reading, target=" " + score_answer(row["plans"][direction]) + EOS),
    ]


def score_answer(text):
    import score
    return score.answer(text)


def encoded_length(example):
    encode = lambda value: TOK.encode(value, add_special_tokens=False)
    if "crystal" in example:
        return len(encode(example["prefix"] + example["lead"])) + K
    return len(encode(example["prefix"] + example["target"]))


def _audit_record(result):
    return {key: result[key] for key in (
        "valid", "failure_stage", "reasons", "decoded_plan_sha256", "structure", "physical", "requested_checks"
    ) if key in result}


def _source_paths(case_id):
    return {
        "selection": ROUNDTRIP_ROOT / "selection.json",
        "plan": INLINE_ROOT / "prepared_final/preflight_only/oracle_plans" / f"{case_id}.json",
        "facts": INLINE_ROOT / "prepared_final/cases" / case_id / "encoder_facts.json",
        "text": RELATIONAL_ROOT / "run100_001/texts" / f"{case_id}.txt",
        "cif": ROUNDTRIP_ROOT / "inputs" / f"{case_id}.cif",
    }


def load_manifest(path=DATA_DIR / "manifest.json"):
    return _read_json(Path(path))


def load_records(splits=None, path=DATA_DIR / "records.jsonl"):
    path = Path(path)
    manifest_path = path.with_name("manifest.json")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if sha256(path.read_bytes()) != manifest["records_sha256"]:
            raise ValueError("Pilot records hash does not match manifest")
    if splits is None:
        return rows
    wanted = {splits} if isinstance(splits, str) else set(splits)
    return [row for row in rows if row["split"] in wanted]


def prepare(out=DATA_DIR):
    """Write exactly the first 24 frozen cases; never scans or expands the cohort."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite frozen pilot data: {out}")

    selection_path = ROUNDTRIP_ROOT / "selection.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    inline_manifest = _read_json(INLINE_ROOT / "prepared_final/manifest.json")
    all_ids = [case["id"] for case in selection["cases"]]
    if selection.get("selected") != 100 or len(all_ids) != 100 or len(set(all_ids)) != 100:
        raise ValueError("Frozen source selection must contain exactly 100 unique ordered IDs")
    ids = all_ids[:24]
    if ids != inline_manifest["case_ids_in_original_order"][:24] or len(ids) != 24:
        raise ValueError("First 24 source IDs differ between frozen selection and prepared facts")

    import score

    rows, token_lengths = [], {"train": [], "heldout": []}
    source_cases = {record["id"]: record for record in inline_manifest["cases"]}
    for index, case_id in enumerate(ids):
        split = "train" if index < 8 else "heldout"
        paths = _source_paths(case_id)
        source_plan_bytes = paths["plan"].read_bytes()
        source_facts_bytes = paths["facts"].read_bytes()
        source_text_bytes = paths["text"].read_bytes()
        source_cif_bytes = paths["cif"].read_bytes()
        plan = json.loads(source_plan_bytes)
        facts = json.loads(source_facts_bytes)
        if sha256(source_plan_bytes) != source_cases[case_id]["oracle_plan_sha256"]:
            raise ValueError(f"Frozen plan hash mismatch for {case_id}")
        if sha256(source_facts_bytes) != source_cases[case_id]["facts_sha256"]:
            raise ValueError(f"Frozen facts hash mismatch for {case_id}")
        if facts["input_atoms"] != plan["input_atom_count"] or facts["space_group_number"] != plan["space_group_number"]:
            raise ValueError(f"Frozen prepared facts disagree with source plan for {case_id}")

        source_text = source_text_bytes.decode("utf-8")
        if _render_plan(plan) != source_text:
            raise ValueError(f"Run100 CNL text differs from facts-rendered source plan for {case_id}")
        if relational_codec.normalize(source_text, table=TABLE) != source_text:
            raise ValueError(f"Source CNL is not canonical for {case_id}")
        if relational_codec.decode(source_text, table=TABLE) != plan:
            raise ValueError(f"Source CNL does not decode to exact frozen plan for {case_id}")

        source_cif = CifParser(str(paths["cif"]), frac_tolerance=0).parse_structures(primitive=False)[0]
        source_exec = _structure(plan)
        source_geom = relational_codec.geometry.compare_structures(source_cif, source_exec)
        source_audit = score.audit(source_text, plan)

        target_plans, target_texts, target_audits, edit_geometry = {}, {}, {}, {}
        target_hashes = {}
        lengths = {}
        for direction, factor in (("plus", "1.05"), ("minus", "0.95")):
            target_plan = _scaled_plan(plan, factor)
            target_text = _render_plan(target_plan)
            decoded = relational_codec.decode(target_text, table=TABLE)
            if decoded != target_plan:
                raise ValueError(f"{case_id} {direction} exact target plan mismatch")
            if relational_codec.normalize(target_text, table=TABLE) != target_text:
                raise ValueError(f"{case_id} {direction} edited CNL is not canonical")
            target_structure = _structure(decoded)
            comparison = relational_codec.geometry.compare_structures(
                _scaled_reference(source_cif, factor), target_structure
            )
            target_audit = score.audit(target_text, target_plan)
            target_plans[direction] = target_plan
            target_texts[direction] = target_text
            target_audits[direction] = _audit_record(target_audit)
            edit_geometry[direction] = {
                "lattice_component_error_A": comparison[1],
                "coordinate_error_A": comparison[0],
            }
            target_hashes[direction] = {
                "plan_sha256": sha256(json.dumps(target_plan, sort_keys=True, separators=(",", ":")).encode()),
                "text_sha256": sha256(target_text.encode()),
            }
            lengths[direction] = len(TOK.encode(target_text, add_special_tokens=False)) + 1
            if lengths[direction] > K:
                raise ValueError(f"{case_id} {direction} requires {lengths[direction]} tokens; K={K}, never truncate")

        lengths["c1"] = len(TOK.encode(source_text, add_special_tokens=False)) + 1
        if lengths["c1"] > K:
            raise ValueError(f"{case_id} source requires {lengths['c1']} tokens; K={K}, never truncate")
        row = {
            "id": case_id,
            "split": split,
            "c1": source_text,
            "plus": target_texts["plus"],
            "minus": target_texts["minus"],
            "condition": _condition(plan),
            "plans": {"source": plan, "plus": target_plans["plus"], "minus": target_plans["minus"]},
            "source_hashes": {
                "selection_sha256": sha256(selection_bytes),
                "source_plan_sha256": sha256(source_plan_bytes),
                "source_facts_sha256": sha256(source_facts_bytes),
                "source_text_sha256": sha256(source_text_bytes),
                "source_cif_sha256": sha256(source_cif_bytes),
            },
            "source_files": {key: path.relative_to(PROJECT).as_posix() for key, path in paths.items()},
            "target_hashes": target_hashes,
            "token_lengths": lengths,
            "baseline_audit": {
                "source": _audit_record(source_audit),
                "plus": target_audits["plus"],
                "minus": target_audits["minus"],
                "source_cif_geometry": {
                    "lattice_component_error_A": source_geom[1],
                    "coordinate_error_A": source_geom[0],
                },
                "edit_geometry": edit_geometry,
            },
        }
        for direction in ("plus", "minus"):
            for example in tasks(row, split, direction):
                total = encoded_length(example)
                if total > MAXLEN:
                    raise ValueError(f"{case_id} {direction} {example['task']} trajectory is {total}; MAXLEN={MAXLEN}, never truncate")
                lengths["max_trajectory"] = max(lengths.get("max_trajectory", 0), total)
        token_lengths[split].append(lengths)
        rows.append(row)

    records_path = out / "records.jsonl"
    out.mkdir(parents=True, exist_ok=False)
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    provenance_paths = {
        "selection": selection_path,
        "prepared_manifest": INLINE_ROOT / "prepared_final/manifest.json",
        "prepared_source_hashes": INLINE_ROOT / "prepared_final/source_hashes.json",
        "relational_codec": RELATIONAL_ROOT / "relational_codec.py",
        "geometry": INLINE_ROOT / "geometry.py",
        "source_facts_helper": INLINE_ROOT / "codec.py",
        "public_wyckoff_fixture": PROJECT / "experiments/explicit_full_mp20_20260923/codec/fixtures/wyckoff_list.csv",
        "preflight": ROOT / "preflight_001/relational_preflight.json",
        "adapter_prepare": ROOT / "prepare.py",
        "adapter_score": ROOT / "score.py",
    }
    runtime_packages = ("numpy", "scipy", "pymatgen", "spglib", "torch", "transformers")
    provenance = {
        "files": {name: sha256(path.read_bytes()) for name, path in provenance_paths.items() if path.exists()},
        "public_table_sha256": relational_codec.geometry.FROZEN_TABLE_SHA256,
        "tokenizer_path": TOKENIZER_PATH.relative_to(PROJECT).as_posix(),
        "tokenizer_files": {path.name: sha256(path.read_bytes()) for path in sorted(TOKENIZER_PATH.iterdir()) if path.is_file()},
        "runtime_versions": {name: importlib.metadata.version(name) for name in runtime_packages},
        "python_version": sys.version.split()[0],
        "model": MODEL,
        "revision": REVISION,
        "source_facts_helper": "relational_codec.oldcodec.source_facts_from_plan(plan)",
    }
    manifest = {
        "status": "frozen",
        "codec": "relational_cnl_v1",
        "cohort": {
            "selection_path": selection_path.relative_to(PROJECT).as_posix(),
            "selection_sha256": sha256(selection_bytes),
            "selected_total": len(all_ids),
            "included": 24,
            "train_ids": ids[:8],
            "heldout_ids": ids[8:24],
            "excluded_from_this_pilot": all_ids[24:],
        },
        "capacity": {"K": K, "MAXLEN": MAXLEN},
        "diffusion": {"STEPS": STEPS, "tokenizer_path": provenance["tokenizer_path"]},
        "token_lengths": {
            split: {
                "source_max": max(row["c1"] for row in lengths),
                "source_p50": statistics.median(row["c1"] for row in lengths),
                "edited_max": max(max(row["plus"], row["minus"]) for row in lengths),
                "trajectory_max": max(row["max_trajectory"] for row in lengths),
            }
            for split, lengths in token_lengths.items()
        },
        "length_distribution_note": "Fixed first-eight train and next-sixteen heldout split is unchanged; the heldout source texts are longer on this cohort. No case was filtered or truncated.",
        "provenance": provenance,
        "records_sha256": sha256(records_path.read_bytes()),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "physical_reference_audit": {
            "source_cases": sum(bool(row["baseline_audit"]["source"]["valid"]) for row in rows),
            "plus_cases": sum(bool(row["baseline_audit"]["plus"]["valid"]) for row in rows),
            "minus_cases": sum(bool(row["baseline_audit"]["minus"]["valid"]) for row in rows),
            "required_denominator_each": 24,
            "no_truncation": True,
        },
        "reference_failure_report": {
            key: {
                "denominator": 24,
                "failed_ids": [row["id"] for row in rows if not row["baseline_audit"][key]["valid"]],
                "classes": dict(Counter(
                    reason
                    for row in rows
                    for reason in row["baseline_audit"][key]["reasons"]
                )),
            }
            for key in ("source", "plus", "minus")
        },
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), ensure_ascii=False, indent=2))
