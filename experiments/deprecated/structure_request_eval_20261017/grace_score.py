"""Score fixed structure-request receipts with a pinned GRACE model and MP hull."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/structure_request_eval_20261017"
ASSETS = EXPERIMENT / "assets"
MODEL_NAME = "GRACE-3L-OMAT-large-ft-AM"
MODEL_MANIFEST = ASSETS / "grace-weight-manifest.json"
HULL_PATH = ASSETS / "2023-02-07-ppd-mp.pkl.gz"
SOURCE_VAL = ASSETS / "source-val.jsonl.gz"
TRAIN_TRIPLES = ROOT / "research/crystal_training_preflight_20261013/data_001/train-triples.jsonl.gz"
EVALUATION_IDS = ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015/evaluation_001/evaluation-ids.json"
CALIBRATION_OUT = EXPERIMENT / "calibration_001"
THRESHOLDS = (0.0, 0.08, 0.1)
SEED = 42
EXPECTED_TRAIN_RECORDS = 7272
HULL_MD5 = "60d19d691fa1d338aa496a40a9641bef"
FMAX = 0.05
MAX_STEPS = 300
MIN_DISTANCE = 0.5
SYMPREC = 0.01
ANGLE_TOLERANCE = 5


def _threshold_key(value):
    return format(float(value), "g")


def proxy_ehull(total_energy_eV, nsites, hull_energy_eV_per_atom):
    if int(nsites) <= 0:
        raise ValueError("nsites must be positive")
    return float(total_energy_eV) / int(nsites) - float(hull_energy_eV_per_atom)


def select_calibration_records(records, entries_by_id, excluded_ids, count=12, seed=SEED):
    eligible = []
    excluded_ids = set(excluded_ids)
    for record in records:
        material_id = record["material_id"]
        if material_id in excluded_ids or material_id not in entries_by_id:
            continue
        entry = entries_by_id[material_id]
        entry_formula = entry["formula"] if isinstance(entry, dict) else entry.composition.reduced_formula
        record_formula = record.get("reduced_formula", record.get("formula"))
        if record_formula == entry_formula:
            eligible.append(record)
    eligible.sort(key=lambda row: row["material_id"])
    if len(eligible) < count:
        raise ValueError(f"need {count} matched calibration structures; found {len(eligible)}")
    random.Random(seed).shuffle(eligible)
    return eligible[:count]


def summarize_scores(plan, results):
    by_slot = {(row["request_id"], row["attempt_id"]): row for row in results}
    completed = [by_slot[(item["request_id"], item["attempt_id"])]
                 for item in plan if (item["request_id"], item["attempt_id"]) in by_slot]
    summary = {
        "status": "complete" if len(completed) == len(plan) else "incomplete",
        "scheduled_attempts": len(plan),
        "completed_attempts": len(completed),
        "pending_attempts": len(plan) - len(completed),
        "raw_attempt_denominator": len(plan),
        "structural_valid_attempts": sum(row.get("structural_valid") is True for row in completed),
        "pre_relax_condition_matches": sum(row.get("requested_sg_match") is True for row in completed),
        "post_relax_valid_attempts": sum(row.get("post_relax_valid") is True for row in completed),
        "post_relax_condition_matches": sum(row.get("post_relax_sg_match") is True for row in completed),
        "converged_relaxations": sum(row.get("converged") is True for row in completed),
        "scored_attempts": sum(row.get("score_status") == "scored" for row in completed),
        "threshold_counts": {},
        "per_request": {},
    }
    requests = sorted({item["request_id"] for item in plan})
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        scored = [row for row in completed if row.get("score_status") == "scored"
                  and row.get("ehull_proxy_corrected_eV_per_atom") is not None
                  and row["ehull_proxy_corrected_eV_per_atom"] <= threshold]
        converged = [row for row in scored if row.get("converged") is True]
        valid_converged = [row for row in converged if row.get("structural_valid") is True
                           and row.get("post_relax_valid") is True]
        conditioned_converged = [row for row in valid_converged
                                 if row.get("post_relax_sg_match") is True]
        unique_novel = [row for row in completed
                        if threshold in row.get("unique_novel_thresholds_eV_per_atom", [])]
        global_unique_novel = [row for row in completed
                               if threshold in row.get("global_unique_novel_thresholds_eV_per_atom", [])]
        summary["threshold_counts"][key] = {
            "endpoint_proxy_leq_diagnostic": len(scored),
            "converged_proxy_leq": len(converged),
            "valid_converged_proxy_leq": len(valid_converged),
            "valid_conditioned_converged_proxy_leq": len(conditioned_converged),
            "unique_novel_conditioned_converged_leq": len(unique_novel),
            "global_unique_novel_conditioned_converged_leq": len(global_unique_novel),
            "raw_attempt_denominator": len(plan),
        }
        for request_id in requests:
            rows = [row for row in completed if row.get("request_id") == request_id]
            per_request = summary["per_request"].setdefault(request_id, {
                "scheduled_attempts": sum(item["request_id"] == request_id for item in plan),
                "completed_attempts": len(rows),
                "strict_valid": sum(row.get("structural_valid") is True for row in rows),
                "post_relax_valid": sum(row.get("post_relax_valid") is True for row in rows),
                "post_relax_condition_matches": sum(row.get("post_relax_sg_match") is True for row in rows),
                "thresholds": {},
            })
            request_unique = [row for row in unique_novel if row.get("request_id") == request_id]
            per_request["thresholds"][key] = {
                "endpoint_proxy_leq_diagnostic": sum(row in scored for row in rows),
                "unique_novel_conditioned_converged_leq": len(request_unique),
                "raw_attempt_denominator": per_request["scheduled_attempts"],
            }
    return summary


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5_file(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _ensure_manifest(path, manifest, resume):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != manifest:
            raise ValueError(f"existing scoring manifest differs: {path}")
        if not resume:
            raise FileExistsError(f"{path.parent} already exists; pass --resume")
        return
    if resume:
        raise FileNotFoundError(f"cannot resume without {path}")
    _write_json(path, manifest)


def _load_hull(path):
    if _md5_file(path) != HULL_MD5:
        raise ValueError("MP phase-diagram asset does not match the pinned 2023-02-07 reference")
    with gzip.open(path, "rb") as stream:
        hull = pickle.load(stream)
    return hull


def _entries_by_material_id(hull):
    entries = defaultdict(list)
    for entry in hull.all_entries:
        material_id = entry.data.get("material_id")
        if material_id:
            entries[str(material_id)].append(entry)
    return entries


def _select_entry(entries, material_id, reduced_formula):
    matches = [entry for entry in entries.get(material_id, [])
               if entry.composition.reduced_formula == reduced_formula]
    if not matches:
        return None
    return min(matches, key=lambda entry: entry.energy_per_atom)


def _read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_previous_ids(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("evaluation_ids", payload) if isinstance(payload, dict) else payload
    return {str(item) for item in ids}


def _calibration_rows(source_path, entries, excluded_ids):
    from pymatgen.core import Structure

    rows = []
    for row in _read_jsonl(source_path):
        structure = Structure.from_dict(row["structure"])
        formula = structure.composition.reduced_formula
        material_id = str(row["material_id"])
        entry = _select_entry(entries, material_id, formula)
        if material_id in excluded_ids or entry is None:
            continue
        rows.append({**row, "reduced_formula": formula, "_structure": structure,
                     "_reference_entry": entry})
    return rows


def _entry_energy_metrics(pairs):
    if not pairs:
        return {"n": 0, "mean_error_eV_per_atom": None, "mae_eV_per_atom": None,
                "rmse_eV_per_atom": None, "spearman_rho": None}
    errors = [predicted - reference for predicted, reference in pairs]
    result = {
        "n": len(pairs),
        "mean_error_eV_per_atom": sum(errors) / len(errors),
        "mae_eV_per_atom": sum(abs(error) for error in errors) / len(errors),
        "rmse_eV_per_atom": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "spearman_rho": None,
    }
    if len(pairs) > 1:
        from scipy.stats import spearmanr
        result["spearman_rho"] = float(spearmanr(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs]).statistic)
    return result


def _calibration_summary(results):
    rows = [row for row in results if row.get("reference_ehull_eV_per_atom") is not None]
    converged_rows = [row for row in rows if row.get("converged") is True
                      and row.get("post_relax_valid") is True]
    corrected_rows = [row for row in converged_rows
                      if row.get("ehull_proxy_corrected_eV_per_atom") is not None]
    output = {
        "n_reference_structures": len(rows),
        "n_converged_valid_structures": len(converged_rows),
        "reference_source": "fixed Matbench Discovery 2023-02-07 MP phase diagram entries",
        "source_2025_energy_above_hull_used": False,
        "raw_grace_proxy": _entry_energy_metrics([
            (row["ehull_proxy_raw_eV_per_atom"], row["reference_ehull_eV_per_atom"])
            for row in converged_rows if row.get("ehull_proxy_raw_eV_per_atom") is not None]),
        "mp2020_corrected_grace_proxy": _entry_energy_metrics([
            (row["ehull_proxy_corrected_eV_per_atom"], row["reference_ehull_eV_per_atom"])
            for row in rows if row.get("ehull_proxy_corrected_eV_per_atom") is not None]),
        "threshold_classification": {},
    }
    for threshold in THRESHOLDS:
        key = _threshold_key(threshold)
        truth = [row["reference_ehull_eV_per_atom"] <= threshold for row in corrected_rows]
        predicted = [row["ehull_proxy_corrected_eV_per_atom"] <= threshold for row in corrected_rows]
        output["threshold_classification"][key] = {
            "n": len(corrected_rows),
            "true_positive": sum(t and p for t, p in zip(truth, predicted)),
            "false_positive": sum(not t and p for t, p in zip(truth, predicted)),
            "false_negative": sum(t and not p for t, p in zip(truth, predicted)),
            "true_negative": sum(not t and not p for t, p in zip(truth, predicted)),
        }
    return output


def _mp2020_metadata(structure):
    from pymatgen.io.vasp.sets import MPRelaxSet

    preset = MPRelaxSet(structure)
    incar = preset.incar
    symbols = preset.poscar.site_symbols
    hubbards = dict(zip(symbols, map(float, incar.get("LDAUU", []))))
    is_hubbard = bool(incar.get("LDAU", True) and sum(hubbards.values()) > 0)
    return {"hubbards": hubbards, "is_hubbard": is_hubbard,
            "run_type": "GGA+U" if is_hubbard else "GGA"}


def _mp2020_corrected_energy(structure, energy_eV):
    from pymatgen.entries.computed_entries import ComputedStructureEntry
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    entry = ComputedStructureEntry(
        structure, float(energy_eV), correction=0.0,
        parameters=_mp2020_metadata(structure))
    corrected = MaterialsProject2020Compatibility(check_potcar=False).process_entry(
        entry, clean=True)
    if corrected is None:
        raise ValueError("MaterialsProject2020Compatibility rejected the GRACE entry")
    return float(corrected.energy), float(corrected.correction)


def _minimum_periodic_distance(structure, cutoff=MIN_DISTANCE):
    # Every site has a periodic image within the shortest lattice vector length.
    search_radius = min(map(float, structure.lattice.abc)) + 1e-8
    distances = structure.get_neighbor_list(search_radius, exclude_self=True)[3]
    return min(map(float, distances), default=None)


def _post_relax_structure_check(structure, initial_formula):
    parameters = list(structure.lattice.abc) + list(structure.lattice.angles)
    finite_lattice = all(math.isfinite(float(value)) for value in parameters)
    volume = float(structure.volume)
    min_distance = _minimum_periodic_distance(structure)
    reasons = []
    if not finite_lattice or not math.isfinite(volume) or volume <= 0:
        reasons.append("invalid_lattice_or_volume")
    if structure.composition.reduced_formula != initial_formula:
        reasons.append("composition_changed_during_relaxation")
    if min_distance is not None and min_distance < MIN_DISTANCE - 1e-8:
        reasons.append("periodic_distance_below_0.5A")
    return not reasons, reasons, min_distance


def _model_calculator(model_dir):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to one explicitly selected GPU")
    import tensorflow as tf
    if len(tf.config.list_physical_devices("GPU")) != 1:
        raise RuntimeError("the GRACE scorer requires exactly one visible GPU")
    from tensorpotential.calculator import TPCalculator
    model_dir = Path(model_dir).resolve(strict=True)
    if not (model_dir / "saved_model.pb").is_file():
        raise FileNotFoundError(f"no TensorFlow SavedModel at {model_dir}")
    return TPCalculator(str(model_dir), min_dist=MIN_DISTANCE)


def _score_structure(structure, calculator, hull):
    import numpy as np
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    total_started = time.monotonic()
    formula = structure.composition.reduced_formula
    pre_valid, pre_errors, pre_min_distance = _post_relax_structure_check(structure, formula)
    if not pre_valid:
        return {"score_status": "invalid_pre_relax_geometry",
                "pre_relax_valid": False,
                "pre_relax_validation_errors": pre_errors,
                "minimum_periodic_distance_pre_relax_A": pre_min_distance,
                "converged": False,
                "post_relax_valid": False,
                "relaxation_seconds": 0.0,
                "timing_seconds": {"total": time.monotonic() - total_started}}
    hull_per_atom = float(hull.get_hull_energy_per_atom(structure.composition))
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    initial_energy_started = time.monotonic()
    initial_energy = float(atoms.get_potential_energy())
    initial_energy_seconds = time.monotonic() - initial_energy_started
    relax_started = time.monotonic()
    optimizer = FIRE(FrechetCellFilter(atoms), logfile=None)
    converged = bool(optimizer.run(fmax=FMAX, steps=MAX_STEPS))
    relaxation_seconds = time.monotonic() - relax_started
    final_singlepoint_started = time.monotonic()
    relaxed_energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    final_singlepoint_seconds = time.monotonic() - final_singlepoint_started
    max_force = float(np.linalg.norm(forces, axis=1).max())
    if not math.isfinite(initial_energy) or not math.isfinite(relaxed_energy) or not np.isfinite(forces).all():
        raise ValueError("GRACE returned a non-finite energy or force")
    final_structure = AseAtomsAdaptor.get_structure(atoms)
    post_valid, post_errors, min_distance = _post_relax_structure_check(final_structure, formula)
    final_sg = int(SpacegroupAnalyzer(
        final_structure, symprec=SYMPREC, angle_tolerance=ANGLE_TOLERANCE).get_space_group_number())
    try:
        corrected_energy, mp2020_correction = _mp2020_corrected_energy(final_structure, relaxed_energy)
        correction_error = None
    except Exception as error:
        corrected_energy = None
        mp2020_correction = None
        correction_error = f"{type(error).__name__}: {error}"
    return {
        "score_status": "scored",
        "score_label": "GRACE energy minus fixed MP 2023 hull proxy; not DFT ehull",
        "energy_units": {"total_energy": "eV", "force": "eV/angstrom", "ehull": "eV/atom"},
        "initial_grace_energy_eV": initial_energy,
        "relaxed_grace_energy_eV": relaxed_energy,
        "relaxed_mp2020_corrected_grace_energy_eV": corrected_energy,
        "mp2020_correction_eV": mp2020_correction,
        "mp2020_correction_error": correction_error,
        "reference_hull_energy_eV_per_atom": hull_per_atom,
        "ehull_proxy_raw_eV_per_atom": proxy_ehull(
            relaxed_energy, len(final_structure), hull_per_atom),
        "ehull_proxy_corrected_eV_per_atom": (
            proxy_ehull(corrected_energy, len(final_structure), hull_per_atom)
            if corrected_energy is not None else None),
        "converged": converged,
        "relax_steps": int(optimizer.get_number_of_steps()),
        "max_atomic_force_eV_per_A": max_force,
        "actual_space_group_number_post_relax": final_sg,
        "post_relax_valid": post_valid,
        "post_relax_validation_errors": post_errors,
        "pre_relax_valid": pre_valid,
        "pre_relax_validation_errors": pre_errors,
        "minimum_periodic_distance_pre_relax_A": pre_min_distance,
        "minimum_periodic_distance_A": min_distance,
        "relaxation_seconds": relaxation_seconds,
        "timing_seconds": {"initial_singlepoint": initial_energy_seconds,
                           "relaxation": relaxation_seconds,
                           "final_singlepoint_and_force": final_singlepoint_seconds,
                           "total": time.monotonic() - total_started},
        "relaxed_structure": final_structure.as_dict(),
    }


def _runtime_software():
    names = ("tensorpotential", "tensorflow", "ase", "pymatgen", "pyxtal", "scipy")
    return {name: importlib.metadata.version(name) for name in names}


def _runtime_manifest(model_dir, hull_path, purpose):
    model_meta = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    return {
        "schema": 1,
        "purpose": purpose,
        "scorer_sha256": _sha256_file(__file__),
        "model": MODEL_NAME,
        "model_manifest": model_meta,
        "model_manifest_sha256": _sha256_file(MODEL_MANIFEST),
        "model_dir": str(Path(model_dir).resolve()),
        "hull_path": str(Path(hull_path).resolve()),
        "hull_sha256": _sha256_file(hull_path),
        "hull_snapshot": "Matbench Discovery 2023-02-07 MP PPD",
        "relaxation": {"optimizer": "ASE FIRE + FrechetCellFilter", "fmax_eV_per_A": FMAX,
                       "max_steps": MAX_STEPS, "min_distance_A": MIN_DISTANCE,
                       "symprec_A": SYMPREC, "angle_tolerance_deg": ANGLE_TOLERANCE},
        "thresholds_eV_per_atom": list(THRESHOLDS),
        "software": _runtime_software(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _import_structure_parser():
    score_path = ROOT / "experiments/interleaved_ar_diffusion_20261002/score.py"
    sys.path.insert(0, str(ROOT / "work/PLaID"))
    spec = importlib.util.spec_from_file_location("grace_score_plaid_parser", score_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_fn_wyckoff


def _load_training_references(data_path, formulas, parse_structure):
    references = defaultdict(list)
    failures = defaultdict(int)
    record_count = 0
    for row in _read_jsonl(data_path):
        record_count += 1
        facts = row.get("core_facts", {})
        formula = facts.get("reduced_formula")
        if formula not in formulas:
            continue
        try:
            structure = parse_structure(row["plaid_text"], to_cif=False)
            references[formula].append(structure)
        except Exception:
            failures[formula] += 1
    if record_count != EXPECTED_TRAIN_RECORDS:
        raise ValueError(f"expected {EXPECTED_TRAIN_RECORDS} training references; found {record_count}")
    return references, failures


def _is_novel(structure, formula, references, failures, matcher):
    if any(matcher.fit(structure, reference) for reference in references.get(formula, [])):
        return False, "matched_training_structure"
    if failures.get(formula, 0):
        return None, "training_reference_parse_incomplete"
    return True, "checked_against_formula_matched_training_structures"


def _matches_any(structure, formula, representatives, matcher):
    for prior_formula, prior_structure, prior_key in representatives:
        if formula == prior_formula and matcher.fit(structure, prior_structure):
            return prior_key
    return None


def _mark_unique_novel(records, matcher):
    for record in records:
        record["unique_novel_thresholds_eV_per_atom"] = []
        record["global_unique_novel_thresholds_eV_per_atom"] = []
    for threshold in THRESHOLDS:
        per_request = defaultdict(list)
        global_representatives = []
        for record in sorted(records, key=lambda row: (row["request_id"], row["attempt_id"])):
            if (record.get("score_status") != "scored" or record.get("structural_valid") is not True
                    or record.get("post_relax_valid") is not True
                    or record.get("converged") is not True
                    or record.get("post_relax_sg_match") is not True
                    or record.get("novel_vs_own_train") is not True
                    or record.get("ehull_proxy_corrected_eV_per_atom") is None
                    or record["ehull_proxy_corrected_eV_per_atom"] > threshold
                    or record.get("_structure") is None):
                continue
            formula = record["reduced_formula"]
            key = (record["request_id"], record["attempt_id"])
            if _matches_any(record["_structure"], formula, per_request[record["request_id"]], matcher) is None:
                record["unique_novel_thresholds_eV_per_atom"].append(threshold)
                per_request[record["request_id"]].append((formula, record["_structure"], key))
            if _matches_any(record["_structure"], formula, global_representatives, matcher) is None:
                record["global_unique_novel_thresholds_eV_per_atom"].append(threshold)
                global_representatives.append((formula, record["_structure"], key))


def _structure_from_result(result):
    if result.get("_structure") is not None:
        return result["_structure"]
    if result.get("relaxed_structure"):
        from pymatgen.core import Structure
        return Structure.from_dict(result["relaxed_structure"])
    return None


def _result_path(out, request_id, attempt_id):
    return out / "attempts" / request_id / f"attempt-{attempt_id:03d}.json"


def _read_generation_run(run_dir):
    run_dir = Path(run_dir).resolve(strict=True)
    if not run_dir.is_relative_to(ROOT.resolve()):
        raise ValueError("run-dir must be under the project root")
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest["attempt_plan"]
    receipts = {}
    for item in plan:
        if not re.fullmatch(r"sg-\d{3}", item["request_id"]):
            raise ValueError("request_id cannot contain path components")
        if not isinstance(item["attempt_id"], int) or item["attempt_id"] < 0:
            raise ValueError("attempt_id must be a nonnegative integer")
        paths = list(run_dir.glob(
            f"shard-*/attempts/{item['request_id']}/attempt-{item['attempt_id']:03d}.json"))
        if len(paths) > 1:
            raise ValueError(f"duplicate generation receipts for {item['request_id']} attempt {item['attempt_id']}")
        if paths:
            receipt = json.loads(paths[0].read_text(encoding="utf-8"))
            if (receipt.get("request_id"), receipt.get("attempt_id"), receipt.get("seed")) != (
                    item["request_id"], item["attempt_id"], item["seed"]):
                raise ValueError("generation receipt does not match the fixed attempt plan")
            receipts[(item["request_id"], item["attempt_id"])] = (receipt, paths[0])
    return run_dir, manifest, plan, receipts


def _score_generation(args):
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Structure

    run_dir, generation_manifest, plan, generation_receipts = _read_generation_run(args.run_dir)
    out = run_dir / "grace_score"
    identity = _runtime_manifest(args.model_dir, args.hull, "score")
    identity.update({"run_manifest_sha256": _sha256_file(run_dir / "run_manifest.json"),
                     "training_triples_sha256": _sha256_file(args.train_data),
                     "calibration_summary_sha256": (
                         _sha256_file(CALIBRATION_OUT / "summary.json")
                         if (CALIBRATION_OUT / "summary.json").is_file() else None)})
    identity["generation_attempt_plan_sha256"] = generation_manifest["attempt_plan_sha256"]
    _ensure_manifest(out / "scoring_manifest.json", identity, args.resume)
    hull = _load_hull(args.hull)
    calculator = _model_calculator(args.model_dir)
    parser = _import_structure_parser()
    matcher = StructureMatcher(ltol=.3, stol=.5, angle_tol=10, primitive_cell=True,
                               scale=True, attempt_supercell=False)
    results = []
    for item in plan:
        request_id, attempt_id = item["request_id"], item["attempt_id"]
        slot = (request_id, attempt_id)
        if slot not in generation_receipts:
            continue
        generation_receipt, receipt_path = generation_receipts[slot]
        result_path = _result_path(out, request_id, attempt_id)
        if result_path.exists():
            if not args.resume:
                raise FileExistsError(f"{result_path} exists; pass --resume")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if (result.get("request_id"), result.get("attempt_id")) != slot:
                raise ValueError(f"existing score receipt does not match {slot}")
        else:
            result = {
                "request_id": request_id,
                "attempt_id": attempt_id,
                "seed": item["seed"],
                "space_group_number_requested": item["space_group_number"],
                "structural_valid": generation_receipt.get("structural_valid") is True,
                "requested_sg_match": generation_receipt.get("requested_sg_match") is True,
                "actual_space_group_number_pre_relax": generation_receipt.get("actual_space_group_number"),
                "generation_receipt": {"path": str(receipt_path), "sha256": _sha256_file(receipt_path)},
                "score_status": "not_scored",
                "ehull_proxy_raw_eV_per_atom": None,
                "ehull_proxy_corrected_eV_per_atom": None,
                "novel_vs_own_train": None,
            }
            if result["structural_valid"]:
                score_started = time.monotonic()
                try:
                    cif_meta = generation_receipt["cif"]
                    cif_path = Path(cif_meta["path"])
                    if not cif_path.is_absolute():
                        cif_path = run_dir / cif_path
                    cif_path = cif_path.resolve(strict=True)
                    if not cif_path.is_relative_to(ROOT.resolve()):
                        raise ValueError("generation CIF must be under the project root")
                    result["source_cif"] = {
                        "path": str(cif_path),
                        "sha256": _sha256_file(cif_path),
                    }
                    structure = Structure.from_file(cif_path)
                    result["reduced_formula"] = structure.composition.reduced_formula
                    result.update(_score_structure(structure, calculator, hull))
                    result["post_relax_sg_match"] = (
                        result.get("actual_space_group_number_post_relax") == item["space_group_number"])
                    result["converged"] = bool(result.get("converged"))
                    relaxed_structure = _structure_from_result(result)
                    if relaxed_structure is not None:
                        result["relaxed_cif"] = str(
                            out / "relaxed" / request_id / f"attempt-{attempt_id:03d}.cif")
                        from pymatgen.io.cif import CifWriter
                        relaxed_path = Path(result["relaxed_cif"])
                        relaxed_path.parent.mkdir(parents=True, exist_ok=True)
                        CifWriter(relaxed_structure).write_file(relaxed_path)
                        result["relaxed_cif_sha256"] = _sha256_file(relaxed_path)
                except Exception as error:
                    result["score_status"] = "failed"
                    result["score_error"] = f"{type(error).__name__}: {error}"
                    result.setdefault("timing_seconds", {})["total"] = time.monotonic() - score_started
            else:
                result["score_status"] = "skipped_invalid_structure"
            _write_json(result_path, result)
        structure = _structure_from_result(result)
        if structure is not None:
            result["_structure"] = structure
        results.append(result)
        _write_json(out / "summary.json", summarize_scores(plan, results))
    formulas = {row["reduced_formula"] for row in results
                if row.get("reduced_formula") and row.get("_structure") is not None}
    training_references, training_parse_failures = _load_training_references(
        args.train_data, formulas, parser)
    for result in results:
        structure = result.get("_structure")
        if result.get("post_relax_valid") is True and structure is not None:
            novel, novelty_status = _is_novel(
                structure, result["reduced_formula"], training_references,
                training_parse_failures, matcher)
            result["novel_vs_own_train"] = novel
            result["novelty_status"] = novelty_status
        else:
            result["novel_vs_own_train"] = None
            result["novelty_status"] = "not_checked_invalid_or_unscored"
    _mark_unique_novel(results, matcher)
    for result in results:
        result.pop("_structure", None)
        _write_json(_result_path(out, result["request_id"], result["attempt_id"]), result)
    summary = summarize_scores(plan, results)
    summary.update({"score_label": "MP2020-corrected GRACE energy minus fixed MP 2023 hull proxy; not DFT ehull",
                    "property_evaluation_status": "proxy_scored",
                    "novelty_reference": {"dataset": str(args.train_data),
                                          "sha256": _sha256_file(args.train_data),
                                          "training_records": EXPECTED_TRAIN_RECORDS,
                                          "formula_matched_reference_counts": {
                                              formula: len(training_references.get(formula, []))
                                              for formula in sorted(formulas)},
                                          "formula_matched_parse_failures": {
                                              formula: training_parse_failures.get(formula, 0)
                                              for formula in sorted(formulas)}},
                    "calibration_summary": (
                        json.loads((CALIBRATION_OUT / "summary.json").read_text(encoding="utf-8"))
                        if (CALIBRATION_OUT / "summary.json").is_file() else None),
                    "scoring_manifest_sha256": _json_sha256(identity)})
    _write_json(out / "summary.json", summary)
    return summary


def _calibrate(args):
    from pymatgen.core import Structure

    identity = _runtime_manifest(args.model_dir, args.hull, "calibrate")
    identity.update({"source_val_sha256": _sha256_file(args.source_val),
                     "excluded_evaluation_ids_sha256": _sha256_file(EVALUATION_IDS),
                     "source_val_snapshot": "local MP 2025.06.09 validation structures; labels ignored",
                     "reference_hull_snapshot": "Matbench Discovery MP 2023-02-07",
                     "calibration_count": 12,
                     "selection_seed": SEED})
    out = args.out.resolve()
    if not out.is_relative_to(ROOT.resolve()):
        raise ValueError("calibration output must be under the project root")
    hull = _load_hull(args.hull)
    entries = _entries_by_material_id(hull)
    source_records = _calibration_rows(args.source_val, entries, _load_previous_ids(EVALUATION_IDS))
    entries_by_id = {}
    for row in source_records:
        entries_by_id[row["material_id"]] = {"formula": row["reduced_formula"]}
    selected = select_calibration_records(
        source_records, entries_by_id, excluded_ids=(), count=12, seed=SEED)
    selected_ids = [row["material_id"] for row in selected]
    identity["selected_material_ids"] = selected_ids
    _ensure_manifest(out / "calibration_manifest.json", identity, args.resume)
    calculator = _model_calculator(args.model_dir)
    results = []
    for index, row in enumerate(selected):
        material_id = row["material_id"]
        if material_id in {".", ".."} or Path(material_id).name != material_id:
            raise ValueError(f"invalid material_id for calibration output: {material_id}")
        entry = row["_reference_entry"]
        result_path = out / "structures" / f"{material_id}.json"
        if result_path.exists():
            if not args.resume:
                raise FileExistsError(f"{result_path} exists; pass --resume")
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {"material_id": material_id, "attempt_id": index,
                      "reduced_formula": row["reduced_formula"],
                      "reference_entry_id": str(entry.entry_id),
                      "reference_entry_energy_eV": float(entry.energy),
                      "reference_ehull_eV_per_atom": float(hull.get_e_above_hull(entry)),
                      "source_energy_above_hull_used": False,
                      "structure_identity": "material_id_and_composition_match_only; entry_has_no_CIF",
                      "score_status": "pending"}
            score_started = time.monotonic()
            try:
                result.update(_score_structure(row["_structure"], calculator, hull))
            except Exception as error:
                result["score_status"] = "failed"
                result["score_error"] = f"{type(error).__name__}: {error}"
                result.setdefault("timing_seconds", {})["total"] = time.monotonic() - score_started
            relaxed_structure = _structure_from_result(result)
            if relaxed_structure is not None:
                relaxed_path = out / "relaxed" / f"{material_id}.cif"
                relaxed_path.parent.mkdir(parents=True, exist_ok=True)
                from pymatgen.io.cif import CifWriter
                CifWriter(relaxed_structure).write_file(relaxed_path)
                result["relaxed_cif"] = str(relaxed_path)
                result["relaxed_cif_sha256"] = _sha256_file(relaxed_path)
            _write_json(result_path, result)
        results.append(result)
        _write_json(out / "summary.json", _calibration_summary(results))
    summary = _calibration_summary(results)
    summary.update({"status": "complete" if len(results) == len(selected) else "incomplete",
                    "source_val_snapshot": "local MP 2025.06.09 validation structures; original labels were not used",
                    "reference_hull_snapshot": "Matbench Discovery 2023-02-07 MP PPD",
                    "n_selected": len(selected),
                    "selected_material_ids": selected_ids,
                    "result_interpretation": "12-structure calibration check, not a fitted correction model",
                    "scoring_manifest_sha256": _json_sha256(identity)})
    _write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "score"), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model-dir", type=Path,
                        default=ASSETS / "grace-model" / MODEL_NAME)
    parser.add_argument("--hull", type=Path, default=HULL_PATH)
    parser.add_argument("--source-val", type=Path, default=SOURCE_VAL)
    parser.add_argument("--train-data", type=Path, default=TRAIN_TRIPLES)
    parser.add_argument("--out", type=Path, default=CALIBRATION_OUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.model_dir, args.hull):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(ROOT.resolve()):
            raise ValueError(f"asset must be under project root: {resolved}")
    if args.mode == "calibrate":
        for path in (args.source_val, EVALUATION_IDS):
            if not path.resolve(strict=True).is_relative_to(ROOT.resolve()):
                raise ValueError(f"calibration input must be under project root: {path}")
        result = _calibrate(args)
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required for --mode score")
        if not args.train_data.resolve(strict=True).is_relative_to(ROOT.resolve()):
            raise ValueError("train-data must be under project root")
        result = _score_generation(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
