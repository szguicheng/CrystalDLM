"""Periodic convex-hull sharing facts for decoded pymatgen structures."""

from collections import Counter, defaultdict
import itertools
import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


SHARING_TYPES = ("corner", "edge", "face")
NAMED_POLYHEDRA = {"tetrahedral", "square pyramidal", "trigonal bipyramidal", "octahedral"}


def _key(site_index, image):
    rounded = np.rint(np.asarray(image, dtype=float))
    if rounded.shape != (3,) or not np.allclose(rounded, image, rtol=0, atol=1e-8):
        raise ValueError(f"periodic image must be an integer 3-vector: {image}")
    return int(site_index), tuple(int(value) for value in rounded)


def _shift_key(vertex, image):
    return vertex[0], tuple(vertex[1][axis] + image[axis] for axis in range(3))


def _polygon_edges(facet, coordinates, normal):
    ids = sorted(facet)
    points = np.array([coordinates[key] for key in ids])
    origin = points[0]
    tangent = points[1] - origin
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    projected = np.column_stack(((points - origin) @ tangent, (points - origin) @ bitangent))
    order = ConvexHull(projected).vertices
    return {
        frozenset((ids[order[index]], ids[order[(index + 1) % len(order)]]))
        for index in range(len(order))
    }


def build_polyhedron(center_xyz, vertices, tolerance_angstrom):
    """Build a strict 3D hull from periodic neighbor rows in Cartesian coordinates."""
    center = np.asarray(center_xyz, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center coordinate must be a finite Cartesian 3-vector")
    points = {}
    elements = {}
    for vertex in vertices:
        vertex_id = _key(vertex["site_index"], vertex["image"])
        coordinate = np.asarray(vertex["coords"], dtype=float)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all():
            raise ValueError(f"vertex {vertex_id} has invalid Cartesian coordinates")
        if vertex_id in points:
            return {"status": "ambiguous", "reason": "duplicate_periodic_vertex"}
        points[vertex_id] = coordinate - center
        elements[vertex_id] = vertex["element"]

    if len(points) < 4:
        return {"status": "not_applicable", "reason": "fewer_than_four_neighbors"}
    ids = list(points)
    coordinates = np.array([points[key] for key in ids])
    for left, right in itertools.combinations(range(len(coordinates)), 2):
        if np.linalg.norm(coordinates[left] - coordinates[right]) <= tolerance_angstrom:
            return {"status": "ambiguous", "reason": "duplicate_vertex_coordinates"}

    try:
        hull = ConvexHull(coordinates)
    except QhullError:
        return {"status": "ambiguous", "reason": "convex_hull_degenerate"}

    if set(map(int, hull.vertices)) != set(range(len(ids))):
        return {"status": "ambiguous", "reason": "neighbor_not_hull_vertex"}
    if float(hull.volume) <= tolerance_angstrom**3:
        return {"status": "ambiguous", "reason": "convex_hull_volume_too_small"}
    if float(np.max(hull.equations[:, -1])) >= -tolerance_angstrom:
        return {"status": "ambiguous", "reason": "center_not_strictly_inside_hull"}

    facets = set()
    edges = set()
    for equation in hull.equations:
        normal, offset = equation[:3], equation[3]
        facet = frozenset(
            ids[index]
            for index, coordinate in enumerate(coordinates)
            if abs(float(np.dot(normal, coordinate) + offset)) <= tolerance_angstrom
        )
        if len(facet) < 3 or facet in facets:
            continue
        facets.add(facet)
        edges.update(_polygon_edges(facet, points, normal))

    return {
        "status": "valid",
        "center_xyz": center,
        "points": points,
        "elements": elements,
        "edges": edges,
        "facets": facets,
        "neighbor_count": len(points),
    }


def candidate_translations(polyhedron_a, polyhedron_b):
    """Return center-image shifts that make at least one periodic vertex coincide."""
    a_by_site = defaultdict(list)
    b_by_site = defaultdict(list)
    for site_index, image in polyhedron_a["points"]:
        a_by_site[site_index].append(image)
    for site_index, image in polyhedron_b["points"]:
        b_by_site[site_index].append(image)
    shifts = {
        tuple(a_image[axis] - b_image[axis] for axis in range(3))
        for site_index in a_by_site.keys() & b_by_site.keys()
        for a_image in a_by_site[site_index]
        for b_image in b_by_site[site_index]
    }
    return sorted(shifts)


def _candidate_polyhedron_pairs(polyhedra):
    incidence = defaultdict(list)
    for polyhedron_index, polyhedron in enumerate(polyhedra):
        for site_index, image in polyhedron["points"]:
            incidence[site_index].append((polyhedron_index, image))

    shifts_by_pair = defaultdict(set)
    for entries in incidence.values():
        for entry_a, entry_b in itertools.combinations(entries, 2):
            index_a, image_a = entry_a
            index_b, image_b = entry_b
            shift = tuple(image_a[axis] - image_b[axis] for axis in range(3))
            if index_a == index_b:
                if shift == (0, 0, 0):
                    continue
                if shift < (0, 0, 0):
                    shift = tuple(-value for value in shift)
                pair = (index_a, index_b)
            elif index_a < index_b:
                pair = (index_a, index_b)
            else:
                pair = (index_b, index_a)
                shift = tuple(-value for value in shift)
            shifts_by_pair[pair].add(shift)
    return [
        (index_a, index_b, shift)
        for (index_a, index_b), shifts in sorted(shifts_by_pair.items())
        for shift in sorted(shifts)
    ]


def _separated(poly_a_points, poly_b_points, shared_keys, shared_points, tolerance):
    origin = shared_points[0]
    shared_count = len(shared_points)
    scale = max(
        1.0,
        *(float(np.linalg.norm(point - origin)) for point in (*poly_a_points.values(), *poly_b_points.values())),
    )
    side_a = [point - origin for key, point in poly_a_points.items() if key not in shared_keys]
    side_b = [point - origin for key, point in poly_b_points.items() if key not in shared_keys]
    if not side_a or not side_b:
        return False

    equality = None
    if shared_count == 2:
        edge = (shared_points[1] - shared_points[0]) / scale
        equality = [[edge[0], edge[1], edge[2], 0.0]]

    threshold = tolerance / scale
    rows = [[*(point / scale), 1.0] for point in side_a]
    rows.extend([[*(-point / scale), 1.0] for point in side_b])
    result = linprog(
        [0.0, 0.0, 0.0, -1.0],
        A_ub=rows,
        b_ub=np.zeros(len(rows)),
        A_eq=equality,
        b_eq=[0.0] if equality else None,
        bounds=[(-1, 1), (-1, 1), (-1, 1), (0, None)],
        method="highs",
    )
    return result.success and result.x[3] > threshold


def classify_pair(polyhedron_a, polyhedron_b, center_b_image, lattice_matrix, tolerance_angstrom):
    """Classify a pure hull contact, rejecting overlaps and unverified features."""
    shift = tuple(int(value) for value in center_b_image)
    if len(shift) != 3 or any(int(value) != value for value in center_b_image):
        raise ValueError("center image must have three integer components")
    lattice_shift = np.asarray(shift, dtype=float) @ np.asarray(lattice_matrix, dtype=float)
    center_a = polyhedron_a["center_xyz"]
    center_b = polyhedron_b["center_xyz"] + lattice_shift
    points_a = {key: center_a + point for key, point in polyhedron_a["points"].items()}
    points_b = {
        _shift_key(key, shift): center_b + point
        for key, point in polyhedron_b["points"].items()
    }
    common = set(points_a) & set(points_b)
    if not common:
        return None
    shared_points = [points_a[key] for key in sorted(common)]
    shifted_edges_b = {
        frozenset(_shift_key(key, shift) for key in edge)
        for edge in polyhedron_b["edges"]
    }
    shifted_facets_b = {
        frozenset(_shift_key(key, shift) for key in facet)
        for facet in polyhedron_b["facets"]
    }

    sharing_type = None
    reason = None
    if len(common) >= 3:
        common_facet = frozenset(common)
        if common_facet in polyhedron_a["facets"] and common_facet in shifted_facets_b:
            normal = np.cross(shared_points[1] - shared_points[0], shared_points[2] - shared_points[0])
            norm = float(np.linalg.norm(normal))
            if norm > tolerance_angstrom:
                normal /= norm
                side_a = float(np.dot(normal, center_a - shared_points[0]))
                side_b = float(np.dot(normal, center_b - shared_points[0]))
                if side_a < -tolerance_angstrom and side_b > tolerance_angstrom or side_b < -tolerance_angstrom and side_a > tolerance_angstrom:
                    sharing_type = "face"
                else:
                    reason = "face_centers_not_on_opposite_sides"
            else:
                reason = "degenerate_common_face"
        else:
            reason = "common_vertices_do_not_form_same_hull_facet"
    elif len(common) == 2:
        edge = frozenset(common)
        if edge not in polyhedron_a["edges"] or edge not in shifted_edges_b:
            reason = "common_vertices_do_not_form_hull_edge"
        elif _separated(points_a, points_b, common, shared_points, tolerance_angstrom):
            sharing_type = "edge"
        else:
            reason = "polyhedra_overlap_beyond_common_edge"
    else:
        if _separated(points_a, points_b, common, shared_points, tolerance_angstrom):
            sharing_type = "corner"
        else:
            reason = "polyhedra_overlap_beyond_common_corner"

    shared_vertices = [
        {"site_index": key[0], "image": list(key[1]), "element": polyhedron_a["elements"][key]}
        for key in sorted(common)
    ]
    if sharing_type is None:
        return {
            "status": "ambiguous",
            "reason": reason,
            "center_b_image": list(shift),
            "common_vertex_count": len(common),
        }
    return {
        "status": "shared",
        "sharing_type": sharing_type,
        "center_b_image": list(shift),
        "shared_vertices": shared_vertices,
    }


def _center_descriptor(polyhedron):
    metadata = polyhedron["center_metadata"]
    recognized = metadata["recognized_motif"]
    return {
        "center_site_index": polyhedron["center_site_index"],
        "center_element": metadata["center_element"],
        "orbit_index": metadata["orbit_index"],
        "motif_classification": metadata["motif_classification"],
        "recognized_motif": recognized,
        "polyhedron_label": recognized if recognized in NAMED_POLYHEDRA else None,
        "motif_metadata_validation": metadata["motif_metadata_validation"],
        "coordination_number": metadata["coordination_number"],
    }


def _validated_motif_metadata(environment, center_element, coordination_number, neighbor_element_counts):
    matches = (
        environment["center_element"] == center_element
        and int(environment["coordination_number"]) == coordination_number
        and environment["neighbor_element_counts"] == neighbor_element_counts
    )
    if not matches:
        return {
            "motif_classification": "unsupported_by_recomputed_neighbors",
            "recognized_motif": None,
            "motif_metadata_validation": "mismatch_with_stored_orbit",
        }
    return {
        "motif_classification": environment["motif_classification"],
        "recognized_motif": environment["recognized_motif"],
        "motif_metadata_validation": "matches_stored_orbit",
    }


def _orbit_descriptor(polyhedron):
    descriptor = _center_descriptor(polyhedron)
    descriptor.pop("center_site_index")
    return descriptor


def extract_polyhedra_sharing(
    structure,
    local_structure_facts,
    neighbor_settings,
    symprec_angstrom,
    angle_tolerance_degrees,
    tolerance_angstrom,
    method_id,
    method_sha256,
    source_method_sha256,
):
    """Extract validated periodic hull contacts from the decoded structure."""
    analyzer = SpacegroupAnalyzer(
        structure,
        symprec=symprec_angstrom,
        angle_tolerance=angle_tolerance_degrees,
    )
    symmetrized = analyzer.get_symmetrized_structure()
    environments = local_structure_facts["coordination_environments"]
    by_orbit = {int(row["orbit_index"]): row for row in environments}
    if len(by_orbit) != len(symmetrized.equivalent_indices):
        raise ValueError("stored coordination environments do not match decoded symmetry orbits")
    environment_by_site = {}
    for orbit_index, indices in enumerate(symmetrized.equivalent_indices):
        row = by_orbit[orbit_index]
        if int(row["representative_site_index"]) != int(indices[0]):
            raise ValueError(f"stored representative does not match decoded orbit {orbit_index}")
        for site_index in indices:
            environment_by_site[int(site_index)] = row

    nn = CrystalNN(**neighbor_settings)
    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=float)
    polyhedra = []
    center_statuses = Counter()
    center_groups = Counter()
    for center_index, center_site in enumerate(structure):
        environment = environment_by_site[center_index]
        neighbors = nn.get_nn_info(structure, center_index)
        center_xyz = np.asarray(center_site.coords, dtype=float)
        vertex_rows = []
        for neighbor in neighbors:
            site_index = int(neighbor["site_index"])
            image = _key(site_index, neighbor["image"])[1]
            coordinate = structure[site_index].coords + np.asarray(image, dtype=float) @ lattice_matrix
            vertex_rows.append({
                "site_index": site_index,
                "image": image,
                "element": structure[site_index].specie.symbol,
                "coords": coordinate,
            })
        metadata = {
            "center_element": center_site.specie.symbol,
            "orbit_index": int(environment["orbit_index"]),
            "coordination_number": len(neighbors),
            "neighbor_element_counts": dict(sorted(Counter(row["element"] for row in vertex_rows).items())),
        }
        metadata.update(_validated_motif_metadata(
            environment,
            metadata["center_element"],
            metadata["coordination_number"],
            metadata["neighbor_element_counts"],
        ))
        if len(neighbors) < 4:
            center_statuses["not_applicable"] += 1
            continue
        center_statuses["eligible"] += 1
        hull = build_polyhedron(center_xyz, vertex_rows, tolerance_angstrom)
        center_statuses[hull["status"]] += 1
        if hull["status"] != "valid":
            center_groups[hull["reason"]] += 1
            continue
        hull.update({
            "center_site_index": center_index,
            "center_metadata": metadata,
        })
        polyhedra.append(hull)

    relation_counts = Counter({kind: 0 for kind in SHARING_TYPES})
    orbit_pairs = {}
    ambiguous_pairs = 0
    ambiguous_examples = []
    for index_a, index_b, shift in _candidate_polyhedron_pairs(polyhedra):
        poly_a, poly_b = polyhedra[index_a], polyhedra[index_b]
        relation = classify_pair(poly_a, poly_b, shift, lattice_matrix, tolerance_angstrom)
        if relation is None:
            continue
        center_a = _center_descriptor(poly_a)
        center_b = _center_descriptor(poly_b)
        orbit_key = tuple(sorted((center_a["orbit_index"], center_b["orbit_index"])))
        if relation["status"] == "ambiguous":
            ambiguous_pairs += 1
            if len(ambiguous_examples) < 3:
                ambiguous_examples.append({"from": center_a, "to": center_b, **relation})
            continue

        sharing_type = relation["sharing_type"]
        relation_counts[sharing_type] += 1
        group = orbit_pairs.setdefault(orbit_key, {
            "orbits": list(orbit_key),
            "centers": [],
            "relation_count_per_cell": 0,
            "sharing_counts": {kind: 0 for kind in SHARING_TYPES},
            "examples": {kind: [] for kind in SHARING_TYPES},
        })
        descriptors = {center_a["orbit_index"]: _orbit_descriptor(poly_a),
                       center_b["orbit_index"]: _orbit_descriptor(poly_b)}
        group["centers"] = [descriptors[index] for index in orbit_key]
        group["relation_count_per_cell"] += 1
        group["sharing_counts"][sharing_type] += 1
        if len(group["examples"][sharing_type]) < 2:
            group["examples"][sharing_type].append({"from": center_a, "to": center_b, **relation})

    valid_polyhedra = len(polyhedra)
    if center_statuses["eligible"] == 0:
        status = "not_applicable"
        reason = "no_center_has_four_or_more_geometry_only_neighbors"
    elif valid_polyhedra == 0:
        status = "ambiguous"
        reason = "no_center_has_a_strict_valid_convex_coordination_hull"
    else:
        status = "computed"
        reason = None

    polyhedron_groups = Counter()
    for polyhedron in polyhedra:
        descriptor = _center_descriptor(polyhedron)
        group_key = (
            descriptor["center_element"],
            descriptor["motif_classification"],
            descriptor["recognized_motif"],
            descriptor["motif_metadata_validation"],
            descriptor["coordination_number"],
        )
        polyhedron_groups[group_key] += 1
    center_types = [
        {
            "center_element": key[0],
            "motif_classification": key[1],
            "recognized_motif": key[2],
            "polyhedron_label": key[2] if key[2] in NAMED_POLYHEDRA else None,
            "motif_metadata_validation": key[3],
            "coordination_number": key[4],
            "count": count,
        }
        for key, count in sorted(polyhedron_groups.items())
    ]
    orbit_rows = [orbit_pairs[key] for key in sorted(orbit_pairs)]
    relation_total = sum(relation_counts.values())
    if status == "not_applicable":
        llm_summary = "No center has CN >= 4 under the frozen geometry-only CrystalNN neighbor method, so convex polyhedron sharing is not applicable."
    elif status == "ambiguous":
        llm_summary = f"Sharing is ambiguous: all {center_statuses['eligible']} CN >= 4 coordination neighborhoods failed strict convex-hull validation."
    elif relation_total == 0:
        llm_summary = f"No validated corner, edge, or face contact was observed among {valid_polyhedra} valid periodic convex coordination hulls."
    else:
        llm_summary = (
            f"Observed {relation_total} validated periodic polyhedron pair contacts among {valid_polyhedra} valid hulls: "
            f"{relation_counts['corner']} corner, {relation_counts['edge']} edge, and {relation_counts['face']} face."
        )
    if ambiguous_pairs:
        llm_summary += f" {ambiguous_pairs} candidate pair(s) remain ambiguous and are excluded from those counts."

    return {
        "status": status,
        "method_id": method_id,
        "method_sha256": method_sha256,
        "source_feature_method_sha256": source_method_sha256,
        "geometry_tolerance_angstrom": tolerance_angstrom,
        "periodic_vertex_identity": ["site_index", "image"],
        "reason": reason,
        "summary": {
            "center_site_count": len(structure),
            "eligible_center_count": center_statuses["eligible"],
            "not_applicable_center_count": center_statuses["not_applicable"],
            "valid_convex_polyhedron_count": valid_polyhedra,
            "ambiguous_center_count": center_statuses["ambiguous"],
            "geometry_issue_counts": dict(sorted(center_groups.items())),
            "polyhedra_by_center": center_types,
            "ambiguous_pair_count": ambiguous_pairs,
            "relation_count_per_cell": relation_total,
            "sharing_type_counts": dict(relation_counts),
            "orbit_pair_count": len(orbit_rows),
            "llm_summary": llm_summary,
        },
        "connectivity": {
            "interpretation": "validated sharing edges between center symmetry orbits, with periodic pair counts",
            "orbit_pair_edges": [
                {
                    "orbits": row["orbits"],
                    "relation_count_per_cell": row["relation_count_per_cell"],
                    "sharing_counts": row["sharing_counts"],
                }
                for row in orbit_rows
            ],
        },
        "orbit_pair_relations": orbit_rows,
        "ambiguous_pair_examples": ambiguous_examples,
    }
