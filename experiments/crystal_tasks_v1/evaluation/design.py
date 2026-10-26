"""Shared structure and property evaluator for bounded design attempts."""
import math
import statistics
from collections import defaultdict

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .legacy import extract_crystal, load_legacy_audit


def _rate(k, n):
    return {"k": k, "n": n, "rate": k / n if n else 0.0}


def _composition_buckets(structures):
    buckets = defaultdict(list)
    for index, structure in enumerate(structures):
        buckets[structure.composition.reduced_formula].append((index, structure))
    return buckets


def _stable_classes(indices, receipts, relaxed, training_buckets, matcher):
    representatives = []
    representatives_by_composition = defaultdict(list)
    classes = {}
    for index in indices:
        receipt = receipts[index]
        if not receipt["stable"]:
            continue
        structure = relaxed[index]
        composition = structure.composition.reduced_formula
        representative = next((
            rep for rep in representatives_by_composition[composition]
            if matcher.fit(structure, relaxed[rep])
        ), None)
        if representative is None:
            representative = index
            representatives.append(index)
            representatives_by_composition[composition].append(index)
            novel = not any(
                matcher.fit(structure, reference)
                for _, reference in training_buckets.get(composition, ())
            )
            classes[index] = (len(representatives) - 1, True, novel)
        else:
            class_id, _, novel = classes[representative]
            classes[index] = (class_id, False, novel)
    return representatives, classes


def _summarize(indices, receipts, relaxed, training_buckets, cov_matches,
               n_test_references, matcher):
    n_attempts = len(indices)
    strict = [index for index in indices if receipts[index]["strict_valid"]]
    stable = [index for index in indices if receipts[index]["stable"]]
    representatives, classes = _stable_classes(
        indices, receipts, relaxed, training_buckets, matcher)
    novel_classes = sum(classes[index][2] for index in representatives)
    matched_outputs = sum(bool(cov_matches.get(index)) for index in strict)
    matched_references = {
        reference for index in strict for reference in cov_matches.get(index, ())
    }
    rates = {
        "parseable": _rate(sum(receipts[i]["parseable"] for i in indices), n_attempts),
        "strict_valid": _rate(len(strict), n_attempts),
        "requested_sg_match": _rate(sum(receipts[i]["requested_sg_match"] for i in indices), n_attempts),
        "property_threshold_pass": _rate(sum(receipts[i]["property_threshold_pass"] for i in indices), n_attempts),
        "converged": _rate(sum(receipts[i]["converged"] for i in indices), n_attempts),
        "post_relax_valid": _rate(sum(receipts[i]["post_relax_valid"] for i in indices), n_attempts),
        "post_relax_sg_match": _rate(sum(receipts[i]["post_relax_sg_match"] for i in indices), n_attempts),
        "stable": _rate(len(stable), n_attempts),
        "unique_stable": _rate(len(representatives), len(stable)),
        "unique_stable_attempt_rate": _rate(len(representatives), n_attempts),
        "novel_unique_stable": _rate(novel_classes, len(representatives)),
        "sun": _rate(novel_classes, n_attempts),
        "cov_precision": _rate(matched_outputs, len(strict)),
        "cov_recall": _rate(len(matched_references), n_test_references),
    }
    supplied = sum(receipts[i]['property_supplied'] for i in indices)
    available = sum(receipts[i]['property_available'] for i in indices)
    missing_valid = sum(not receipts[i]['property_available'] for i in strict)
    if missing_valid:
        for name in ('property_threshold_pass', 'stable', 'unique_stable',
                     'unique_stable_attempt_rate', 'novel_unique_stable', 'sun',
                     'converged', 'post_relax_valid', 'post_relax_sg_match'):
            rates[name]['rate'] = None
    status = 'not_run' if not supplied else 'failed' if not available else 'partial' if missing_valid else 'complete'
    return {"n_attempts": n_attempts, "rates": rates, 'property_evaluation': {
        'status': status, 'receipts_supplied': supplied, 'receipts_available': available,
        'missing_valid_receipts': missing_valid,
    }}, classes


def evaluate_design(attempts, training_references, test_references):
    """Score raw attempts; S.U.N. uses all attempts, including parse failures."""
    legacy = load_legacy_audit()
    matcher = StructureMatcher(
        ltol=0.3, stol=0.5, angle_tol=10, primitive_cell=True,
        scale=True, attempt_supercell=False,
    )
    training_buckets = _composition_buckets(training_references)
    test_buckets = _composition_buckets(test_references)
    receipts, decoded, relaxed, cov_matches = [], {}, {}, {}

    for index, attempt in enumerate(attempts):
        output = attempt["output"]
        try:
            crystal = extract_crystal(output)
            markers_valid = True
        except Exception as error:
            crystal = output
            markers_valid = False
            structure = None
            parseable = False
            parse_error = f"{type(error).__name__}: {error}"
        if markers_valid:
            try:
                structure = legacy.parse_fn_wyckoff(crystal, to_cif=False)
                parseable, parse_error = True, None
            except Exception as error:
                structure = None
                parseable = False
                parse_error = f"{type(error).__name__}: {error}"

        audit = legacy.audit(crystal)
        strict_valid = bool(markers_valid and audit.get("valid"))
        actual_sg = audit.get("detected_sg")
        decoded_sg_error = None
        if structure is not None and actual_sg is None:
            try:
                actual_sg = int(SpacegroupAnalyzer(
                    structure, symprec=0.01, angle_tolerance=5
                ).get_space_group_number())
            except Exception as error:
                decoded_sg_error = f"{type(error).__name__}: {error}"
        requested_sg = attempt["request"]["space_group"]
        requested_sg_match = actual_sg == requested_sg
        property_receipt = attempt.get("property_receipt") or {}
        property_available = attempt.get('property_receipt') is not None and property_receipt.get('score_status') not in ('setup_error', 'error', 'audit_error')
        ehull = property_receipt.get("ehull_proxy_corrected_eV_per_atom")
        ehull = float(ehull) if ehull is not None else None
        ehull_upper = float(attempt["request"]["ehull_upper"])
        property_pass = ehull is not None and math.isfinite(ehull) and ehull <= ehull_upper
        converged = property_receipt.get("converged") is True
        post_relax_valid = property_receipt.get("post_relax_valid") is True
        post_relax_sg = property_receipt.get("actual_space_group_number_post_relax")
        post_relax_sg_match = post_relax_sg == requested_sg
        stable = bool(
            property_available and strict_valid and requested_sg_match and converged and
            post_relax_valid and post_relax_sg_match and property_pass
        )
        if parseable:
            decoded[index] = structure
        if stable:
            relaxed[index] = Structure.from_dict(property_receipt["relaxed_structure"])
        receipts.append({
            "attempt_id": attempt["attempt_id"],
            "seed": attempt["seed"],
            "request": attempt["request"],
            "parseable": parseable,
            "parse_error": parse_error,
            "strict_valid": strict_valid,
            "strict_audit": audit,
            "actual_space_group_number_pre_relax": actual_sg,
            "decoded_sg_error": decoded_sg_error,
            "requested_sg_match": requested_sg_match,
            "property_threshold_pass": property_pass,
            "corrected_ehull_eV_per_atom": ehull,
            "property_supplied": attempt.get('property_receipt') is not None,
            "property_available": property_available,
            "converged": converged,
            "post_relax_valid": post_relax_valid,
            "actual_space_group_number_post_relax": post_relax_sg,
            "post_relax_sg_match": post_relax_sg_match,
            "stable": stable,
            "stable_class_id": None,
            "unique_stable_representative": False,
            "stable_class_novel": None,
            "cov_test_match": None,
        })

    for index, structure in decoded.items():
        if receipts[index]["strict_valid"]:
            cov_matches[index] = [
                ref_index for ref_index, reference in test_buckets.get(
                    structure.composition.reduced_formula, ()
                ) if matcher.fit(structure, reference)
            ]
            receipts[index]["cov_test_match"] = bool(cov_matches[index])

    all_indices = list(range(len(attempts)))
    summary, overall_classes = _summarize(
        all_indices, receipts, relaxed, training_buckets, cov_matches,
        len(test_references), matcher,
    )
    for index, (class_id, is_representative, novel) in overall_classes.items():
        receipts[index]["stable_class_id"] = class_id
        receipts[index]["unique_stable_representative"] = is_representative
        receipts[index]["stable_class_novel"] = novel

    per_seed = {}
    for seed in sorted({attempt["seed"] for attempt in attempts}):
        indices = [i for i, attempt in enumerate(attempts) if attempt["seed"] == seed]
        per_seed[seed], _ = _summarize(
            indices, receipts, relaxed, training_buckets, cov_matches,
            len(test_references), matcher,
        )
    multiseed_rates = {}
    for name in summary["rates"]:
        values = [per_seed[seed]["rates"][name]["rate"] for seed in per_seed]
        multiseed_rates[name] = {
            "mean": statistics.fmean(values) if values and all(v is not None for v in values) else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 and all(v is not None for v in values) else None,
        }
    return {
        "summary": summary,
        "receipts": receipts,
        "per_seed": per_seed,
        "multiseed": {"variance_scope": "generation_seeds", "rates": multiseed_rates},
    }
