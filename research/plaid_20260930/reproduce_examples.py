"""Run the unmodified PLaID codec on two released generations and one train CIF."""
import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1] / "work" / "PLaID"
sys.path.insert(0, str(REPO))
from utils import get_crystal_string_wyckoff_pyx, parse_fn_wyckoff
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

source = REPO / "results/qwen_7b_dpo_wyckoff_sg_combined_t2_novelty_v2_dt_v2_it7_temp1.3_esen_ehull_results_combined_sun.csv"
with source.open() as f:
    reader = csv.DictReader(f)
    cases = [("released_" + str(i), source, i, next(reader)) for i in range(2)]
train = REPO / "data/basic/train.csv"
with train.open() as f:
    for i, row in enumerate(csv.DictReader(f)):
        if row["material_id"] == "mp-2657":
            cases.append(("train_mp-2657", train, i, row))
            break
assert len(cases) == 3

matcher = StructureMatcher(scale=False)
strict = StructureMatcher(ltol=1e-4, stol=1e-4, angle_tol=1e-3, scale=False)
receipt = {
    "repository_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version,
    "packages": {p: importlib.metadata.version(p) for p in ["pyxtal", "pymatgen", "spglib", "numpy"]},
    "note": "No model inference, relaxation, or stability evaluation; unmodified upstream codec only.",
    "cases": [],
}
for name, path, index, row in cases:
    text = row["gen_str"].split("\n", 1)[1] if "gen_str" in row else get_crystal_string_wyckoff_pyx(row["cif"])
    original = Structure.from_str(row["cif"], fmt="cif")
    decoded = parse_fn_wyckoff(text, to_cif=False)
    assert matcher.fit(original, decoded), name
    assert Structure.from_str(parse_fn_wyckoff(text), fmt="cif").composition == decoded.composition
    (OUT / (name + ".txt")).write_text(text + "\n")
    (OUT / (name + "_source.cif")).write_text(row["cif"])
    decoded.to(filename=str(OUT / (name + "_decoded.cif")), fmt="cif")
    decoded.to(filename=str(OUT / (name + ".vasp")), fmt="poscar")
    if "gen_str" in row:
        (OUT / (name + "_full_prompt.txt")).write_text(row["gen_str"] + "\n")
    result = dict(name=name, source=str(path.relative_to(REPO)), csv_data_row_zero_based=index,
                  material_id=row.get("material_id"), formula=decoded.formula, atoms=len(decoded),
                  orbit_records=len(text.splitlines()) - 5, structure_match=True,
                  strict_structure_match=bool(strict.fit(original, decoded)),
                  source_lattice=list(original.lattice.parameters), decoded_lattice=list(decoded.lattice.parameters))
    receipt["cases"].append(result)
    print(name + "\n" + text + "\n" + json.dumps(result) + "\n")

text = (OUT / "released_0.txt").read_text()
lines = text.splitlines()
lines[0], lines[4] = "H999", "Sites (999)"
receipt["header_corruption_ignored"] = bool(strict.fit(parse_fn_wyckoff(text, False), parse_fn_wyckoff("\n".join(lines), False)))
assert receipt["header_corruption_ignored"]
(OUT / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("PASS: 3 examples match their source CIFs; formula and Sites headers are not checked by this decoder.")
