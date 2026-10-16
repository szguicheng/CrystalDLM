"""Prepare the frozen cohort for inline binding and offline-only preflight."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments/naturecry_llm_100_20261002"))
sys.path.insert(0, str(HERE))

import cohort
import codec
import geometry

FROZEN = PROJECT / "experiments/naturecry_roundtrip_20261001/run_001"
SELECTION = FROZEN / "selection.json"
INVENTORY_SCHEMA = {
    "codec_version": codec.CODEC_VERSION,
    "parameter_tag": {"element": "param", "required_attributes": ["name", "type", "unit", "frame"],
                      "body": "compact JSON scalar/vector; float64 values use shortest-roundtrip decimal repr"},
    "types": ["string", "int", "null|int", "float64", "float64[n]", "float64[3,3]", "composition_count"],
    "required_groups": ["cell.a/b/c/alpha/beta/gamma", "space_group.number", "space_group.hall", "input.atom_count",
                        "composition.<element>", "orbit.<id>.id/element/wyckoff/parameters when nonempty",
                        "frame.matrix", "frame.origin_shift", "round_digits", "correction.<id>.site_id/element/delta"],
    "frames": {"cell": "input", "orbit.parameters": "standard", "frame.matrix": "standard_from_input",
               "frame.origin_shift": "standard", "correction.delta": "input_delta"},
}
GEOMETRY_RULES = {
    "version": geometry.GEOMETRY_RULES_VERSION,
    "free_parameter_order": "exact axes order in the frozen public Wyckoff definition; no representative fitting",
    "orbit_order": "stable orbit id ascending",
    "operation_order": "frozen public definition order",
    "frame_relation": "q_standard = P q_input + s",
    "frame_images": "low=floor(sum(min(P,0), axis=1)+s-1), high=ceil(sum(max(P,0), axis=1)+s+1), inclusive integer ranges in itertools.product order with the last axis changing fastest",
    "boundary_selection": "retain transformed candidates with every component in [-1e-9,1+1e-9], then modulo 1 and snap values within 1e-12 of 1 to 0",
    "deduplication": "first-retained, same-element periodic max-component distance <1e-8; no deduplication after correction",
    "stable_site_id": "<orbit_id>_<1-based retained-site ordinal within orbit>",
    "rounding": "numpy.round to round_digits fractional decimals (ties to even), then modulo 1",
    "execution_order": ["expand standard public orbits", "enumerate input-frame images", "select boundary images", "modulo, snap boundary one to zero, and species-aware first-retained deduplicate", "round coordinates if round_digits is stated", "modulo rounded coordinates", "resolve correction site_ids", "add input-frame fractional delta", "modulo corrected coordinates", "check composition and input atom count", "construct Structure"],
    "thresholds": {"coordinate_A_strictly_less_than": 1e-6, "lattice_component_A_strictly_less_than": 1e-8},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def update_progress(step, status, detail):
    path = HERE / "progress.json"
    state = json.loads(path.read_text())
    state["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    state["current_step"] = step
    state["steps"][step] = status
    state.setdefault("notes", []).append(detail)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


def _clean_list(value):
    return np.asarray(value, dtype=float).tolist()


def prepare_case(source, table):
    old_facts, _ = cohort.prepare_case(source)
    full, _, audit = cohort.wyckoff.encode_cif(source, symprec=1e-3)
    if audit["table_sha256"] != geometry.FROZEN_TABLE_SHA256:
        raise ValueError("Source encoder used an unexpected public table")
    sg = int(old_facts["space_group_number"])
    group = table[sg]
    axis_name = ("x", "y", "z")
    orbits, orbit_context = [], []
    for index, old_orbit in enumerate(old_facts["independent_orbits"]):
        label = old_orbit["wyckoff"]
        definition = group["definitions"][label]
        parameters = [float(old_orbit["independent_parameters"][axis_name[int(axis)]]) for axis in definition["axes"]]
        orbit_id = f"o{index:03d}"
        if len(parameters) != len(definition["axes"]):
            raise ValueError(f"Audit free-parameter order mismatch for {orbit_id}")
        orbits.append({"id": orbit_id, "element": old_orbit["element"], "wyckoff": label, "parameters": parameters})
        orbit_context.append({"id": orbit_id, "multiplicity": int(definition["multiplicity"]),
                              "coordinate_relation": old_orbit["coordinate_relation"],
                              "free_axes": [axis_name[int(axis)] for axis in definition["axes"]]})

    rows = full.splitlines()
    residual_header = next(index for index, row in enumerate(rows) if row.startswith("RESIDUALS "))
    residual_count = int(rows[residual_header].split()[1])
    residual_rows = rows[residual_header + 1:residual_header + 1 + residual_count]
    old_corrections = old_facts["precision_corrections"]
    if len(residual_rows) != residual_count or len(old_corrections) != residual_count:
        raise ValueError("Legacy residual rows and prepared correction facts disagree")

    raw_frame = old_facts["frame_transform"]
    facts = {
        "input_lattice_parameters": list(map(float, old_facts["input_lattice_parameters"])),
        "input_composition": dict(sorted(old_facts["input_composition"].items())),
        "input_atoms": int(old_facts["input_atoms"]),
        "space_group_number": sg,
        "space_group_symbol": old_facts["space_group_symbol"],
        "hall_symbol": old_facts["hall_symbol"],
        "frame_transform": {"matrix": _clean_list(raw_frame["matrix"]), "origin_shift": _clean_list(raw_frame["origin_shift"])},
        "round_digits": old_facts["round_digits"],
        "orbits": orbits,
        "public_orbit_context": orbit_context,
        "precision_corrections": [],
    }

    std_sites = geometry.expand_orbit_sites(sg, orbits, table=table)
    frame = facts["frame_transform"]
    positions, species, site_ids, _ = geometry.to_input_frame_stable(
        std_sites, np.asarray(frame["matrix"]), np.asarray(frame["origin_shift"])
    )
    if facts["round_digits"] is not None:
        positions = np.round(positions, facts["round_digits"]) % 1
    original = cohort.parse(source)
    residual_map = []
    mapped = []
    for row, correction in zip(residual_rows, old_corrections):
        tokens = row.split()
        legacy_index, legacy_delta = int(tokens[0]), list(map(float, tokens[1:]))
        if legacy_delta != list(map(float, correction["delta"])):
            raise ValueError(f"Legacy residual vector mismatch at atom index {legacy_index}")
        target = np.asarray(correction["at"], dtype=float)
        displacement = positions - target
        displacement -= np.rint(displacement)
        errors = np.linalg.norm(displacement @ original.lattice.matrix, axis=1)
        matches = [index for index, (z, error) in enumerate(zip(species, errors))
                   if int(z) == int(cohort.Element(correction["element"]).Z) and error < 1e-8]
        if len(matches) != 1:
            raise ValueError(f"Legacy residual index {legacy_index} maps to {len(matches)} public sites")
        stable_id = site_ids[matches[0]]
        mapped.append({"site_id": stable_id, "element": correction["element"], "delta": legacy_delta})
        residual_map.append({"legacy_expanded_site_index": legacy_index, "stable_site_id": stable_id,
                             "element": correction["element"]})
    if len({record["legacy_expanded_site_index"] for record in residual_map}) != len(residual_map):
        raise ValueError("Repeated legacy residual index")
    if len({record["stable_site_id"] for record in residual_map}) != len(residual_map):
        raise ValueError("Repeated stable residual target")
    facts["precision_corrections"] = sorted(mapped, key=lambda record: record["site_id"])
    plan = codec.plan_from_facts(facts)
    codec.validate_plan(plan, table)
    return facts, plan, residual_map, audit


def prepare(output=HERE / "prepared_final"):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared output: {output}")
    selection_bytes = SELECTION.read_bytes()
    selection = json.loads(selection_bytes)
    if selection.get("selected") != 100 or len(selection.get("cases", [])) != 100:
        raise ValueError("Frozen selection must contain exactly 100 cases")
    ids = [item["id"] for item in selection["cases"]]
    if len(set(ids)) != 100:
        raise ValueError("Frozen selection contains duplicate IDs")

    output.mkdir(parents=True)
    update_progress("final_preparation", "in_progress", "Preparing a final hash-pinned version from the unchanged frozen 100-case selection.")
    table = geometry.verify_public_table()
    records, input_hashes, frame_count, correction_cases, correction_atoms = [], {}, 0, 0, 0
    for item in selection["cases"]:
        source = FROZEN / "inputs" / f"{item['id']}.cif"
        actual_hash = digest(source.read_bytes())
        input_hashes[item["id"]] = actual_hash
        record = {"id": item["id"], "input_sha256_expected": item["input_cif_sha256"], "input_sha256_actual": actual_hash}
        try:
            if actual_hash != item["input_cif_sha256"]:
                raise ValueError("Frozen input hash mismatch")
            facts, plan, residual_map, audit = prepare_case(source, table)
            directory = output / "cases" / item["id"]
            write_json(directory / "encoder_facts.json", facts)
            inventory_path = directory / "inventory.json"
            write_json(inventory_path, {"codec_version": codec.CODEC_VERSION,
                       "fields": [{"name": name, **spec} for name, spec in codec.make_inventory(facts).items()]})
            oracle_path = output / "preflight_only" / "oracle_plans" / f"{item['id']}.json"
            map_path = output / "preflight_only" / "legacy_residual_maps" / f"{item['id']}.json"
            write_json(oracle_path, plan)
            write_json(map_path, residual_map)
            frame = facts["frame_transform"]
            identity = np.allclose(frame["matrix"], np.eye(3), atol=1e-12, rtol=0) and np.allclose(frame["origin_shift"], 0, atol=1e-12, rtol=0)
            frame_count += not identity
            correction_cases += bool(residual_map)
            correction_atoms += len(residual_map)
            record.update(status="prepared", atoms=facts["input_atoms"], space_group=plan["space_group_number"],
                          nonidentity_frame=not identity, correction_atoms=len(residual_map),
                          facts_sha256=digest((directory / "encoder_facts.json").read_bytes()),
                          inventory_sha256=digest(inventory_path.read_bytes()),
                          oracle_plan_sha256=digest(oracle_path.read_bytes()),
                          legacy_residual_map_sha256=digest(map_path.read_bytes()))
        except Exception as error:
            record.update(status="source_preparation_failed", error=f"{type(error).__name__}: {error}")
        records.append(record)

    write_json(HERE / "inventory_schema.json", INVENTORY_SCHEMA)
    write_json(HERE / "decoder_schema.json", codec.DECODER_SCHEMA)
    write_json(HERE / "rules.json", GEOMETRY_RULES)
    write_json(output / "prompts.json", {"encoder": codec.ENCODER_PROMPT, "decoder": codec.DECODER_PROMPT})
    hashes = {
        "selection_json": digest(selection_bytes),
        "selected_dataset": selection["dataset_sha256"],
        "input_cifs": input_hashes,
        "cohort_py": digest((PROJECT / "experiments/naturecry_llm_100_20261002/cohort.py").read_bytes()),
        "public_wyckoff_module": digest((PROJECT / "experiments/explicit_full_mp20_20260923/codec/wyckoff_baseline.py").read_bytes()),
        "public_wyckoff_fixture": digest((PROJECT / "experiments/explicit_full_mp20_20260923/codec/fixtures/wyckoff_list.csv").read_bytes()),
        "geometry_py": digest((HERE / "geometry.py").read_bytes()),
        "codec_py": digest((HERE / "codec.py").read_bytes()),
        "prepare_py": digest((HERE / "prepare.py").read_bytes()),
        "preflight_py": digest((HERE / "preflight.py").read_bytes()),
        "api_runner_py": digest((HERE / "api_runner.py").read_bytes()),
        "inventory_schema_json": digest((HERE / "inventory_schema.json").read_bytes()),
        "decoder_schema_json": digest((HERE / "decoder_schema.json").read_bytes()),
        "rules_json": digest((HERE / "rules.json").read_bytes()),
        "protocol_md": digest((HERE / "PROTOCOL.md").read_bytes()),
        "prompts_json": digest((output / "prompts.json").read_bytes()),
        "encoder_prompt": digest(codec.ENCODER_PROMPT.encode()),
        "decoder_prompt": digest(codec.DECODER_PROMPT.encode()),
        "decoder_schema": digest(json.dumps(codec.DECODER_SCHEMA, sort_keys=True, separators=(",", ":")).encode()),
    }
    manifest = {
        "codec_version": codec.CODEC_VERSION,
        "selection_sha256": hashes["selection_json"],
        "selected_dataset_sha256": selection["dataset_sha256"],
        "selected": 100,
        "case_ids_in_original_order": ids,
        "cases": records,
        "support_summary": {"nonidentity_frames": frame_count, "correction_cases": correction_cases,
                            "residual_atoms": correction_atoms},
    }
    write_json(output / "selection.json", selection)
    write_json(output / "manifest.json", manifest)
    write_json(output / "source_hashes.json", hashes)
    failures = sum(record["status"] != "prepared" for record in records)
    update_progress("final_preparation", "completed", f"Prepared {100 - failures}/100 cases; failed source preparations remain in the denominator.")
    print(json.dumps({"prepared": 100 - failures, "failed": failures, "support_summary": manifest["support_summary"],
                      "selection_sha256": hashes["selection_json"]}, indent=2))


if __name__ == "__main__":
    prepare(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "prepared_final")
