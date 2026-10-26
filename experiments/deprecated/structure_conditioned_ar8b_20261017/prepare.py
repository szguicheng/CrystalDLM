"""Space-group and truthful ehull-upper-bound conditions over frozen targets."""
from decimal import Decimal
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = spec_from_file_location(
    "structural_ar_original_prepare",
    ROOT / "experiments/ehull_pure_ar_qwen3_8b_20261015/prepare.py")
SOURCE = module_from_spec(_spec)
_spec.loader.exec_module(SOURCE)
for _name in ("END", "EOS", "K", "MASK", "MAXLEN", "MODEL", "PAD", "READ_QUESTION",
              "REVISION", "START", "STEPS"):
    globals()[_name] = getattr(SOURCE, _name)

EHULL_UPPER_BOUNDS = (0.08, 0.1, 0.2, 0.3)


def ehull_upper_bound(value):
    energy = Decimal(str(value))
    if not energy.is_finite() or energy < 0:
        raise ValueError("source ehull must be finite and nonnegative")
    for bound in EHULL_UPPER_BOUNDS:
        if energy <= Decimal(str(bound)):
            return bound
    raise ValueError("source ehull exceeds the frozen 0.3 eV/atom data range")


def tasks(row):
    if row["property_labels"]["unit"] != "eV/atom":
        raise ValueError("ehull labels must use eV/atom")
    group = int(row["core_facts"]["space_group_number"])
    bound = ehull_upper_bound(row["property_labels"]["energy_above_hull"])
    prefix = (f"User: Generate a crystal with space group number {group} and "
              f"energy above hull no greater than {bound:g} eV/atom.\nAssistant:")
    examples = SOURCE.tasks(row)
    old_prefix = examples[0]["prefix"]
    for example in examples:
        assert example["prefix"].startswith(old_prefix)
        example["prefix"] = prefix + example["prefix"][len(old_prefix):]
    return examples
