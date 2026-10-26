"""Canonical four-fact readback grading for saved or newly generated responses."""
import re

VERSION='four-facts-v2-20261020'


def grade_four_facts(response, formula, space_group, atom_count, orbit_count):
    from pymatgen.core import Composition
    formulas=re.findall(
        r'(?i:\b(?:composition|formula)\s*(?:is\s*|[:=]\s*))(.+?)(?=\.(?:\s|$)|[;\n]|<\||$)',response)
    target=Composition(formula).get_el_amt_dict()
    try:
        formula_correct=bool(formulas) and all(
            Composition(value.strip()).get_el_amt_dict()==target for value in formulas)
    except (ValueError,TypeError):
        formula_correct=False
    patterns={
        'space_group':(r'(?:space\s*group)\s*(?:number\s*)?(?:is\s*|[:=]\s*)?(\d+)',space_group),
        'cell_atom_count':(r'(\d+)\s+atoms\b',atom_count),
        'independent_orbit_count':(r'(\d+)\s+(?:independent\s+)?orbits\b',orbit_count),
    }
    fields={'formula':formula_correct}
    for key,(pattern,target_value) in patterns.items():
        values=re.findall(pattern,response,re.I)
        fields[key]=bool(values) and all(int(value)==int(target_value) for value in values)
    return dict(fields=fields,all_four_correct=all(fields.values()),evaluator_version=VERSION)
