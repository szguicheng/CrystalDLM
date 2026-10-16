import hashlib
import math


_BIN_LABELS = ("0", "(0,0.05]", "(0.05,0.1]", "(0.1,0.2]", "(0.2,0.3]")
_SPLITS = ("train", "val", "test")


def _bin_index(energy):
    if energy == 0:
        return 0
    if energy < 0:
        return None
    if 0 < energy <= 0.05:
        return 1
    if energy <= 0.1:
        return 2
    if energy <= 0.2:
        return 3
    if energy <= 0.3:
        return 4
    return None


def _digest(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _record_rank(row, seed):
    material_id = row["material_id"]
    return _digest(f"{seed}|{material_id}"), str(material_id)


def _allocate_quotas(counts, budget):
    systems = sorted(counts)
    quotas = {system: 0 for system in systems}
    remaining = min(budget, sum(counts.values()))
    active = systems[:]

    while active and remaining:
        weight_total = sum(math.sqrt(counts[system]) for system in active)
        capped = [
            system
            for system in active
            if remaining * math.sqrt(counts[system]) / weight_total >= counts[system] - quotas[system]
        ]
        if not capped:
            break
        for system in capped:
            capacity = counts[system] - quotas[system]
            quotas[system] += capacity
            remaining -= capacity
            active.remove(system)

    if active and remaining:
        weight_total = sum(math.sqrt(counts[system]) for system in active)
        for system in active:
            share = remaining * math.sqrt(counts[system]) / weight_total
            quota = min(counts[system] - quotas[system], math.floor(share))
            quotas[system] += quota
        remaining = min(budget, sum(counts.values())) - sum(quotas.values())

    while remaining:
        progressed = False
        for system in systems:
            if quotas[system] < counts[system]:
                quotas[system] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            break
    return quotas


def select_records(rows, per_bin=10000, seed=17):
    """Deduplicate structures and sample each hull-energy bin deterministically."""
    structure_groups = {}
    input_counts = [0] * len(_BIN_LABELS)
    out_of_range_ids = []

    for row in rows:
        index = _bin_index(row["energy_above_hull"])
        if index is None:
            out_of_range_ids.append(row["material_id"])
        else:
            input_counts[index] += 1
        structure_groups.setdefault(row["structure_key"], []).append(row)

    duplicate_groups = []
    by_structure = {}
    deduplicated_ids = []
    for key, group in structure_groups.items():
        ranked = sorted(group, key=lambda row: _record_rank(row, seed))
        by_structure[key] = ranked[0]
        if len(group) > 1:
            dropped = sorted((row["material_id"] for row in ranked[1:]), key=str)
            deduplicated_ids.extend(dropped)
            duplicate_groups.append({
                "structure_key": key,
                "kept_material_id": by_structure[key]["material_id"],
                "deduplicated_ids": dropped,
            })

    candidates = [[] for _ in _BIN_LABELS]
    for row in by_structure.values():
        index = _bin_index(row["energy_above_hull"])
        if index is not None:
            candidates[index].append(row)

    selected = []
    bin_reports = {}
    for index, label in enumerate(_BIN_LABELS):
        by_system = {}
        for row in candidates[index]:
            by_system.setdefault(row["chemsys"], []).append(row)
        counts = {system: len(group) for system, group in by_system.items()}
        quotas = _allocate_quotas(counts, per_bin)
        system_report = {}
        selected_count = 0
        for system in sorted(by_system):
            chosen = sorted(by_system[system], key=lambda row: _record_rank(row, seed))[:quotas[system]]
            selected.extend(chosen)
            selected_count += len(chosen)
            system_report[system] = {
                "candidate_count": len(by_system[system]),
                "quota": quotas[system],
                "selected_count": len(chosen),
            }
        bin_reports[label] = {
            "input_count": input_counts[index],
            "candidate_count": len(candidates[index]),
            "selected_count": selected_count,
            "chemical_systems": system_report,
        }

    selected.sort(key=lambda row: (_bin_index(row["energy_above_hull"]), str(row["chemsys"]), _record_rank(row, seed)))
    duplicate_groups.sort(key=lambda item: str(item["structure_key"]))
    return selected, {
        "seed": seed,
        "per_bin_limit": per_bin,
        "selection_rule": "ascending SHA-256(f'{seed}|{material_id}'); material_id ascending breaks ties",
        "deduplicated_ids": sorted(set(deduplicated_ids), key=str),
        "duplicate_groups": duplicate_groups,
        "out_of_range_ids": sorted(out_of_range_ids, key=str),
        "bins": bin_reports,
    }


def split_records(selected_rows, seed=17):
    """Assign each unique structure to one deterministic 80/10/10 split."""
    by_structure = {}
    for row in selected_rows:
        by_structure.setdefault(row["structure_key"], []).append(row)

    space = 1 << 256
    train_cutoff = 4 * space // 5
    val_cutoff = 9 * space // 10
    split_by_structure = {}
    hash_by_structure = {}
    for key in by_structure:
        digest = _digest(f"{seed}|split|{key}")
        value = int(digest, 16)
        split_by_structure[key] = "train" if value < train_cutoff else "val" if value < val_cutoff else "test"
        hash_by_structure[key] = digest

    result = []
    row_counts = {split: 0 for split in _SPLITS}
    structure_counts = {split: 0 for split in _SPLITS}
    for key in sorted(by_structure, key=lambda item: (hash_by_structure[item], str(item))):
        split = split_by_structure[key]
        structure_counts[split] += 1
        for row in sorted(by_structure[key], key=lambda item: (str(item["material_id"]), _record_rank(item, seed))):
            result.append({**row, "split": split})
            row_counts[split] += 1

    return result, {
        "seed": seed,
        "rule": "SHA-256(f'{seed}|split|{structure_key}') as an integer: below 80% train, below 90% val, otherwise test",
        "target_ratio": {"train": 0.8, "val": 0.1, "test": 0.1},
        "row_counts": row_counts,
        "structure_counts": structure_counts,
    }
