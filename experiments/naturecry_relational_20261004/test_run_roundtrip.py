from pathlib import Path
import json
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_roundtrip import CONTRACTS, exact_plan_equal, require_pilot_for_limit, select_prefix, validate_pilot


def test_100_case_run_requires_a_pilot_summary():
    with pytest.raises(ValueError, match="--pilot"):
        require_pilot_for_limit(100, None)


def test_roster_is_the_unfiltered_selection_prefix():
    selected = [f"case-{index:03d}" for index in range(100)]
    assert select_prefix(selected, 30) == selected[:30]
    assert select_prefix(selected, 100) == selected


def test_exact_plan_comparison_preserves_numeric_spelling():
    assert not exact_plan_equal({"x": -0.0}, {"x": 0.0})
    assert not exact_plan_equal({"x": 1}, {"x": 1.0})


@pytest.mark.parametrize(
    ("pilot_hash", "stable", "after_hash"),
    [("old-hash", True, "old-hash"), ("new-hash", False, "new-hash"),
     ("new-hash", True, "changed-after-run")],
)
def test_pilot_with_changed_or_unstable_runtime_is_rejected(tmp_path, pilot_hash, stable, after_hash):
    summary = {
        "limit": 30,
        "selected_denominator": 30,
        "completed": 30,
        "passed": 30,
        "failed": 0,
        "all_contracts_passed": True,
        "contract_pass_counts": {name: 30 for name in CONTRACTS},
        "selection_sha256": "selection-hash",
        "runtime_hashes": {"relational_codec.py": pilot_hash},
        "runtime_hashes_stable": stable,
        "runtime_hashes_after": {"relational_codec.py": after_hash},
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="runtime hashes"):
        validate_pilot(path, "selection-hash", {"relational_codec.py": "new-hash"})
