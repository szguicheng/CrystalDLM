import copy
from collect import metadata_rejection, structure_key, structure_rejection


def check():
    structure = {"lattice": {"matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]}, "sites": [
        {"species": [{"element": "Si", "occu": 1}], "abc": [0, 0, 0]},
        {"species": [{"element": "O", "occu": 1}], "abc": [0.25, 0.25, 0.25]},
    ]}
    row = {"material_id": "mp-test", "energy_above_hull": 0.3, "nsites": 2, "nelements": 2,
           "elements": ["Si", "O"], "deprecated": False, "structure": structure}
    assert metadata_rejection(row) is None and structure_rejection(row) is None
    shifted = copy.deepcopy(structure)
    shifted["sites"].reverse()
    for site in shifted["sites"]:
        site["abc"] = [(v+0.125) % 1 for v in site["abc"]]
    assert structure_key(shifted) == structure_key(structure)
    altered = copy.deepcopy(structure)
    altered["sites"][1]["abc"][0] += 0.01
    assert structure_key(altered) != structure_key(structure)
    row["structure"] = copy.deepcopy(structure)
    row["structure"]["sites"][0]["species"][0]["occu"] = 0.5
    assert structure_rejection(row) == "disorder_or_partial_occupancy"
    row["energy_above_hull"] = float("nan")
    assert metadata_rejection(row) == "missing_or_nonfinite_ehull"
    print("collection checks passed")


if __name__ == "__main__":
    check()
