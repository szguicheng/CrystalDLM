"""Four-fact readback metrics for Crystal Tasks v1 Understanding."""

import re


VERSION = "four-facts-v3-20261025"
FIELDS = ("formula", "space_group", "cell_atom_count", "independent_orbit_count")


def grade_four_facts(response, formula, space_group, atom_count, orbit_count):
    from pymatgen.core import Composition

    formulas = re.findall(
        r"(?i:\b(?:composition|formula)\s*(?:is\s*|[:=]\s*))(.+?)(?=\.(?:\s|$)|[;\n]|<\||$)",
        response,
    )
    expected = Composition(formula).reduced_composition
    try:
        formula_correct = bool(formulas) and all(
            Composition(value.strip()).reduced_composition == expected for value in formulas
        )
    except (TypeError, ValueError):
        formula_correct = False

    integer = r"(?<![\w.+-])([+-]?\d+(?:\.\d+)?)(?!\w)(?!\.\d)"
    patterns = {
        "space_group": (
            r"(?:space\s*group)\s*(?:number\s*)?(?:is\s*|[:=]\s*)?" + integer,
            space_group,
        ),
        "cell_atom_count": (integer + r"\s+atoms\b", atom_count),
        "independent_orbit_count": (
            integer + r"\s+(?:independent\s+)?orbits\b",
            orbit_count,
        ),
    }
    fields = {"formula": formula_correct}
    for key, (pattern, target) in patterns.items():
        values = re.findall(pattern, response, re.I)
        fields[key] = bool(values) and all(
            value.isdecimal() and int(value) == int(target) for value in values
        )
    return {
        "fields": fields,
        "all_four_correct": all(fields.values()),
        "evaluator_version": VERSION,
    }


def summarize_understanding(rows):
    grades = [
        grade_four_facts(
            row["response"],
            row["reference"]["formula"],
            row["reference"]["space_group"],
            row["reference"]["cell_atom_count"],
            row["reference"]["independent_orbit_count"],
        )
        for row in rows
    ]
    n = len(grades)

    def metric(correct):
        k = sum(correct)
        return {"k": k, "n": n, "rate": k / n if n else 0.0}

    return {
        "fields": {
            field: metric(grade["fields"][field] for grade in grades) for field in FIELDS
        },
        "all_four": metric(grade["all_four_correct"] for grade in grades),
        "grades": grades,
    }
