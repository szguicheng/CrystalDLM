"""Public geometry engine for NatureCry inline codec.

Follows frozen PyXtal/spglib Wyckoff public reference definitions.
Implements canonical orbit expansion, stable site ID assignment,
immutable ID lookup for precision corrections, and exact structure reconstruction.
"""
from collections import Counter
import hashlib
import itertools
from pathlib import Path
import sys

import numpy as np
from pymatgen.core import Element, Lattice, Structure
from scipy.optimize import linear_sum_assignment
import spglib

# Baseline references (read-only)
BACKEND = Path(__file__).resolve().parents[2] / 'experiments/explicit_full_mp20_20260923/codec'
sys.path.insert(0, str(BACKEND))
import wyckoff_baseline as wyckoff

FROZEN_TABLE_SHA256 = 'ca56602252b7896121253c910739ef2fccf0ba947f2940358feb941f30c5e13d'
GEOMETRY_RULES_VERSION = 'naturecry-public-expand-frame-select-round-correct-v1'


def verify_public_table():
    fixture = BACKEND / 'fixtures/wyckoff_list.csv'
    actual_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if actual_sha != FROZEN_TABLE_SHA256:
        raise ValueError(f"Public table hash mismatch: expected {FROZEN_TABLE_SHA256}, got {actual_sha}")
    return wyckoff.reference_table()


def compare_structures(a: Structure, b: Structure):
    """Rigorous comparison matching cohort.py / benchmark.py standards."""
    if len(a) != len(b):
        raise ValueError(f"Atom count mismatch: {len(a)} vs {len(b)}")
    if a.composition != b.composition:
        raise ValueError(f"Composition mismatch: {a.composition} vs {b.composition}")
    lattice_error = float(np.max(np.abs(a.lattice.matrix - b.lattice.matrix)))
    if not lattice_error < 1e-8:
        raise ValueError(f"Lattice component error {lattice_error:.2e} >= 1e-8 A")
    same = np.array(a.atomic_numbers)[:, None] == np.array(b.atomic_numbers)[None, :]
    distances = a.lattice.get_all_distances(a.frac_coords, b.frac_coords)
    i, j = linear_sum_assignment(np.where(same, distances, np.inf))
    residual = a.frac_coords[i] - b.frac_coords[j]
    residual -= np.rint(residual)
    coord_error = float(np.linalg.norm(residual @ a.lattice.matrix, axis=1).max())
    if not coord_error < 1e-6:
        raise ValueError(f"Coordinate error {coord_error:.2e} >= 1e-6 A")
    return coord_error, lattice_error


def expand_orbit_sites(sg: int, orbits: list, table=None):
    """Canonically expand orbits using exact audit node free parameters.
    
    No pseudoinverse or representative fitting is performed.
    """
    if table is None:
        table = verify_public_table()
    definitions = table[sg]['definitions']
    
    # Orbits are ordered canonically by ID
    sorted_orbits = sorted(orbits, key=lambda o: o['id'])
    
    std_sites = []
    for orbit in sorted_orbits:
        oid = orbit['id']
        z = Element(orbit['element']).Z
        label = orbit['wyckoff']
        if label not in definitions:
            raise ValueError(f"Unknown Wyckoff label {label} for space group {sg}")
        d = definitions[label]
        params = np.array(orbit.get('parameters', []), dtype=float)
        if len(params) != len(d['axes']):
            raise ValueError(
                f"Orbit {oid} ({label}) requires {len(d['axes'])} free parameters, got {len(params)}"
            )
        full_u = np.zeros(3, dtype=float)
        full_u[d['axes']] = params
        for op in d['operations']:
            a, b = op
            q = (a @ full_u + b) % 1
            std_sites.append((q, z, oid))
    return std_sites


def to_input_frame_stable(std_sites: list, matrix: np.ndarray, shift: np.ndarray):
    """Transform standard sites into the original input frame with stable site IDs.
    
    Assigns stable site IDs: f"{orbit_id}_{site_counter_within_orbit}".
    """
    p = np.array(matrix, dtype=float)
    s = np.array(shift, dtype=float)
    if p.shape != (3, 3) or s.shape != (3,):
        raise ValueError("Invalid frame transform dimensions")
    if abs(np.linalg.det(p)) < 1e-12:
        raise ValueError("Degenerate frame matrix")
    
    low = np.floor(np.minimum(p, 0).sum(1) + s - 1).astype(int)
    high = np.ceil(np.maximum(p, 0).sum(1) + s + 1).astype(int)
    images = np.array(list(itertools.product(*[range(a, b + 1) for a, b in zip(low, high)])))
    inv_p = np.linalg.inv(p)
    
    retained_positions = []
    retained_species = []
    retained_site_ids = []
    orbit_site_counter = Counter()

    for q_std, z, oid in std_sites:
        candidates = (q_std + images - s) @ inv_p.T
        candidates = candidates[np.all((candidates >= -1e-9) & (candidates <= 1 + 1e-9), axis=1)]
        for pt in candidates:
            pt = pt % 1
            pt[np.abs(pt - 1) < 1e-12] = 0.0
            # Species-aware first-retained deduplication
            if any(z == zz and np.max(np.abs(pt - old_pt - np.rint(pt - old_pt))) < 1e-8
                   for old_pt, zz in zip(retained_positions, retained_species)):
                continue
            retained_positions.append(pt)
            retained_species.append(z)
            orbit_site_counter[oid] += 1
            site_id = f"{oid}_{orbit_site_counter[oid]}"
            retained_site_ids.append(site_id)
            
    site_id_to_index = {sid: idx for idx, sid in enumerate(retained_site_ids)}
    return (
        np.array(retained_positions, dtype=float),
        retained_species,
        retained_site_ids,
        site_id_to_index
    )


def execute_geometry_plan(plan: dict, table=None) -> Structure:
    """Execute a validated decoder construction plan into a Structure.
    
    Preserves sequence:
      1. Standard orbit expansion (exact free parameters)
      2. Input-frame image selection, mod, first-retained dedup, stable ID assignment
      3. Rounding (if round_digits is specified)
      4. Precision corrections via immutable site_id -> index lookup
      5. Composition check and Structure construction
    """
    if table is None:
        table = verify_public_table()
    sg = int(plan['space_group_number'])
    if sg not in table:
        raise ValueError(f"Unknown space group number: {sg}")
    
    # 1. Expand standard sites
    std_sites = expand_orbit_sites(sg, plan['orbits'], table=table)
    
    # 2. Map to input frame & assign stable IDs
    frame = plan['frame_transform']
    matrix = np.array(frame['matrix'], dtype=float)
    shift = np.array(frame['origin_shift'], dtype=float)
    positions, species, site_ids, site_id_to_index = to_input_frame_stable(std_sites, matrix, shift)
    
    # 3. Rounding before corrections
    digits = plan.get('round_digits')
    if digits is not None:
        if not isinstance(digits, int) or isinstance(digits, bool) or not (0 <= digits <= 15):
            raise ValueError(f"Invalid round_digits: {digits}")
        positions = np.round(positions, digits) % 1
        
    # 4. Precision corrections (immutable lookup, no spatial coordinate search)
    corrected_indices = set()
    for corr in plan.get('precision_corrections', []):
        target_id = corr['site_id']
        if target_id not in site_id_to_index:
            raise ValueError(f"Correction target site_id {target_id} not found in expanded sites")
        idx = site_id_to_index[target_id]
        if idx in corrected_indices:
            raise ValueError(f"Duplicate precision correction target: {target_id}")
        corrected_indices.add(idx)
        
        elem_expected = Element.from_Z(species[idx]).symbol
        if corr['element'] != elem_expected:
            raise ValueError(
                f"Correction target {target_id} element mismatch: expected {elem_expected}, got {corr['element']}"
            )
            
        delta = np.array(corr['delta'], dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError(f"Invalid correction delta for {target_id}: {delta}")
        positions[idx] = (positions[idx] + delta) % 1
        
    # 5. Composition check
    actual_counts = {Element.from_Z(z).symbol: n for z, n in Counter(species).items()}
    if actual_counts != plan['composition']:
        raise ValueError(
            f"Expanded composition mismatch: expected {plan['composition']}, got {actual_counts}"
        )
    input_atom_count = plan.get('input_atom_count', sum(plan['composition'].values()))
    if sum(actual_counts.values()) != input_atom_count:
        raise ValueError("Expanded atom count does not match input_atom_count")
        
    # 6. Structure creation
    lattice_params = plan['lattice']
    if len(lattice_params) != 6 or not np.isfinite(lattice_params).all():
        raise ValueError(f"Invalid lattice parameters: {lattice_params}")
    lattice = Lattice.from_parameters(*lattice_params)
    structure = Structure(lattice, species, positions)
    if len(structure) != input_atom_count:
        raise ValueError("Constructed atom count does not match input_atom_count")
    return structure
