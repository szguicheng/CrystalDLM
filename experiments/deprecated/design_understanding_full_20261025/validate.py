"""Check the exported files against source crystals, split membership and answers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from prepare import DATA, STAGE, SOURCE, EXPECTED, NAMES, START, END, rows, sha, write_json


def composition(text):
    compact = text.replace(" ", "")
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", compact)
    assert "".join(element + number for element, number in parts) == compact
    result = Counter()
    for element, number in parts:
        result[element] += int(number or "1")
    return result


def verify_crystal(body, facts, consistency):
    lines = body.splitlines()
    orbit_counts = Counter()
    for line in lines[5:]:
        element, x, y, z, wyckoff = line.split()
        multiplicity = int(re.fullmatch(r"(\d+)[A-Za-z]", wyckoff).group(1))
        orbit_counts[element] += multiplicity
    assert composition(lines[0]) == orbit_counts == composition(facts["cell_formula"])
    assert int(re.fullmatch(r"Sites \((\d+)\)", lines[4]).group(1)) == sum(orbit_counts.values()) == facts["cell_atom_count"]
    assert len(lines[5:]) == facts["independent_orbit_count"]
    # PLaID drops screw-axis underscores, e.g. I41/a versus I4_1/a.
    assert lines[1] == "Spacegroup: " + consistency["encoded_space_group_symbol"]
    assert consistency["encoded_space_group_number"] == consistency["decoded_space_group_number"] == facts["space_group_number"]
    assert consistency["encoded_space_group_canonical_symbol"] == consistency["decoded_space_group_canonical_symbol"]


def main(understanding_only=False):
    registry = json.loads((DATA / "dataset_info.json").read_text())
    tasks = ["understanding"] if understanding_only else ["design", "understanding"]
    assert len(registry) == 3 * len(tasks)
    report, all_ids, all_hashes = {}, set(), set()
    for split, expected in EXPECTED.items():
        name = NAMES[split]
        source_path = SOURCE / f"{split}-facts.jsonl.gz"
        count = 0
        for index, (row, meta) in enumerate(zip(rows(source_path), rows(DATA / "metadata" / f"{name}.jsonl"), strict=True)):
            assert row["material_id"] == meta["material_id"] and meta["row_index"] == index
            assert row["split"] == meta["source_split"] == split
            assert meta["material_id"] not in all_ids
            assert meta["source_structure_sha256"] not in all_hashes
            all_ids.add(meta["material_id"])
            all_hashes.add(meta["source_structure_sha256"])
            assert meta["core_facts"] == row["core_facts"]
            assert meta["property_labels"] == row["property_labels"]
            assert meta["source_structure_sha256"] == row["source"]["structure_sha256"]
            verify_crystal(row["plaid"]["encoded_text"], row["core_facts"], row["plaid"]["consistency"])
            count += 1
        assert count == expected
        for task in tasks:
            key = f"crystal_{task}_{name}"
            path = DATA / registry[key]["file_name"]
            assert registry[key]["columns"] == {"prompt": "instruction", "query": "input", "response": "output"}
            count = 0
            for sample, source in zip(rows(path), rows(source_path), strict=True):
                assert set(sample) == {"instruction", "input", "output"}
                assert all(isinstance(v, str) for v in sample.values())
                assert sample["instruction"] and sample["output"]
                assert not any(marker in "\n".join(sample.values()) for marker in ("<|im_start|>", "<|im_end|>", "User:", "Assistant:"))
                body = source["plaid"]["encoded_text"]
                facts = source["core_facts"]
                if task == "understanding":
                    assert sample["input"] == START + body + END
                    match = re.fullmatch(r"The formula is (.+)\. The space group number is (\d+)\. The unit cell contains (\d+) atoms and (\d+) independent orbits\.", sample["output"])
                    assert match is not None
                    formula, sg, atoms, orbits = match.groups()
                    assert composition(formula) == composition(body.splitlines()[0])
                    assert (int(sg), int(atoms), int(orbits)) == (facts["space_group_number"], facts["cell_atom_count"], facts["independent_orbit_count"])
                    assert sample["instruction"] == "Read the current crystal and report its formula, space group number, cell atom count, and independent orbit count."
                else:
                    assert sample["input"] == ""
                    caption, encoded = sample["output"].split("\n" + START)
                    assert caption.strip() and encoded == body + END
                    assert not re.search(r"[\u4e00-\u9fff]", caption)
                    assert source["material_id"] not in caption
                    condition = re.fullmatch(r"Generate a crystal with space group number (\d+) and energy above hull no greater than ([\d.]+) eV/atom\.", sample["instruction"])
                    assert condition and int(condition[1]) == facts["space_group_number"]
                    assert 0 <= source["property_labels"]["energy_above_hull"] <= float(condition[2])
                count += 1
            assert count == expected
            report[key] = {"rows": count, "sha256": sha(path)}
    result = {"status": "passed", "scope": tasks, "files": report, "unique_materials": len(all_ids),
              "unique_source_structure_hashes": len(all_hashes), "no_split_overlap": True,
              "understanding_context_is_plaid_only": True, "four_answers_match_structure_facts": True,
              "all_cells_have_matching_formula_multiplicities_atom_and_orbit_counts": True}
    destination = DATA / ("understanding-validation-report.json" if understanding_only else "validation-report.json")
    write_json(destination, result)
    if not understanding_only:
        manifest = json.loads((DATA / "manifest.json").read_text())
        manifest.update(status="both_tasks_ready", validation_report_sha256=sha(destination))
        write_json(DATA / "manifest.json", manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--understanding-only", action="store_true")
    main(parser.parse_args().understanding_only)
