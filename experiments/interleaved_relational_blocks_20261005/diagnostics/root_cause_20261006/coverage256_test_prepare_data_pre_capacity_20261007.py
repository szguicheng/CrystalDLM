from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from prepare_data import (
    _coverage, _prepare_case_with_unified_sg_policy, _runtime_policy,
    append_jsonl, select_candidates,
)


def row(case_id, sg, elements=("A",), digest=None, source_row=0):
    return {
        "id": case_id,
        "cif_sha256": digest or f"cif-{case_id}",
        "space_group_number": sg,
        "elements": list(elements),
        "source_row_zero_based": source_row,
    }


def test_exclusions_and_round_robin_are_deterministic():
    rows = [
        row("old-id", 1),
        row("old-cif", 2, digest="old-hash"),
        row("held-chemistry", 1, elements=("A", "B")),
        row("a1", 1), row("a2", 1),
        row("b1", 2), row("b2", 2),
        row("c1", 3), row("c2", 3), row("duplicate-cif", 3, digest="cif-c1"),
    ]
    kwargs = dict(
        excluded_ids={"old-id"},
        excluded_cif_hashes={"old-hash"},
        heldout_element_sets={frozenset(("A", "B"))},
        limit=6,
    )
    selected, counts = select_candidates(rows, **kwargs)
    repeated, _ = select_candidates(rows, **kwargs)

    assert [item["id"] for item in selected] == [item["id"] for item in repeated]
    assert [item["space_group_number"] for item in selected] == [1, 2, 3, 1, 2, 3]
    assert {item["id"] for item in selected}.isdisjoint({"old-id", "old-cif", "held-chemistry"})
    assert counts["excluded_by_reason"] == {
        "old_selection_id": 1,
        "old_selection_cif_sha256": 1,
        "heldout_element_set": 1,
        "duplicate_candidate_id": 0,
        "duplicate_candidate_cif_sha256": 1,
    }


def test_selects_224_unique_candidates_across_169_space_groups():
    rows = [row(f"mp-{sg}-{copy}", sg, digest=f"sha-{sg}-{copy}", source_row=sg * 2 + copy)
            for sg in range(1, 170) for copy in range(2 if sg <= 55 else 1)]
    selected, counts = select_candidates(rows)

    assert len(selected) == 224
    assert len({item["id"] for item in selected}) == 224
    assert len({item["cif_sha256"] for item in selected}) == 224
    assert {item["space_group_number"] for item in selected} == set(range(1, 170))
    assert counts["selected"] == 224


def test_appended_jsonl_keeps_old_prefix_bytes():
    prefix = b'{"id":"old"}\n'
    output = append_jsonl(prefix, [{"id": "new"}])

    assert output.startswith(prefix)
    assert output == prefix + b'{"id":"new"}\n'


def test_coverage_splits_and_heldout_unseen_are_train_only():
    def record(case_id, split, sg, labels):
        return {
            "id": case_id,
            "split": split,
            "plans": {"source": {"space_group_number": sg,
                                  "orbits": [{"wyckoff": label} for label in labels]}},
        }

    report = _coverage([
        record("train-a", "train", 1, ["a"]),
        record("train-b", "train", 1, ["b"]),
        record("held-seen", "heldout", 1, ["a"]),
        record("held-new-wyckoff", "heldout", 1, ["c"]),
        record("held-new-sg", "heldout", 2, ["a"]),
    ])

    assert report["by_split"]["train"]["encoded_space_group_counts"] == {"1": 2}
    assert report["by_split"]["heldout"]["encoded_space_group_counts"] == {"1": 2, "2": 1}
    assert report["by_split"]["all"]["encoded_space_group_counts"] == {"1": 4, "2": 1}
    unseen = report["heldout_generalization_from_train"]
    assert unseen["heldout_ids_with_unseen_space_group"] == ["held-new-sg"]
    assert unseen["heldout_ids_with_unseen_sg_wyckoff_pair"] == ["held-new-wyckoff", "held-new-sg"]
    assert unseen["unseen_sg_wyckoff_pairs_by_heldout_id"] == {
        "held-new-wyckoff": ["1:c"], "held-new-sg": ["2:a"]
    }


def test_runtime_policy_allows_torch_and_python_patch_only():
    baseline = {"numpy": "1.26.4", "scipy": "1.15.3", "pymatgen": "2025.10.7",
                "spglib": "2.7.0", "torch": "2.8.0", "transformers": "4.57.6"}
    actual = {**baseline, "torch": "2.2.0+cu121"}
    policy = _runtime_policy(baseline, actual, "3.10.19", "3.10.18")
    assert policy["accepted"]
    assert policy["baseline_python"] == "3.10.19" and policy["actual_python"] == "3.10.18"
    assert policy["baseline_torch"] == "2.8.0" and policy["actual_torch"] == "2.2.0+cu121"
    assert policy["arithmetic_scope"] == "data preparation performs no Torch/model arithmetic"
    assert not _runtime_policy(baseline, actual, "3.10.19", "3.11.0")["accepted"]
    for package in ("numpy", "scipy", "pymatgen", "spglib", "transformers"):
        changed = {**actual, package: actual[package] + ".different"}
        assert not _runtime_policy(baseline, changed, "3.10.19", "3.10.18")["accepted"]


def test_unified_sg_policy_scopes_both_encoder_calls_and_preserves_inputs():
    cif_bytes = b"data_fake\n_cell_length_a 3.141592653589793\n"
    coordinate = 0.12345678901234566
    with TemporaryDirectory() as directory:
        cif = Path(directory) / "source.cif"
        cif.write_bytes(cif_bytes)
        classifications = []

        def get_dataset(cell, **kwargs):
            classifications.append((cell, dict(kwargs)))
            return {"call": len(classifications)}

        spglib = SimpleNamespace(get_symmetry_dataset=get_dataset)

        def encode_cif(path, symprec=1e-3):
            cell = cif.read_bytes()
            return (
                spglib.get_symmetry_dataset(cell, symprec=symprec, angle_tolerance=-1),
                spglib.get_symmetry_dataset(cell, hall_number=7, symprec=symprec, angle_tolerance=-1),
            )

        wyckoff = SimpleNamespace(spglib=spglib, encode_cif=encode_cif)

        def cohort_prepare_case(path):
            return {"encoded": wyckoff.encode_cif(path, symprec=1e-3)}

        cohort = SimpleNamespace(wyckoff=wyckoff, prepare_case=cohort_prepare_case)

        def inline_prepare_case(path, table):
            first = cohort.prepare_case(path)
            second = wyckoff.encode_cif(path, symprec=1e-3)
            return {"coordinate": coordinate}, {"coordinate": coordinate}, first, second

        inline = SimpleNamespace(cohort=cohort, prepare_case=inline_prepare_case)
        original_encode = wyckoff.encode_cif
        original_get_dataset = spglib.get_symmetry_dataset

        facts, plan, _, _ = _prepare_case_with_unified_sg_policy(inline, str(cif), {})

        assert len(classifications) == 4
        assert all(kwargs["symprec"] == 0.01 and kwargs["angle_tolerance"] == 5
                   for _, kwargs in classifications)
        assert [kwargs.get("hall_number") for _, kwargs in classifications] == [None, 7, None, 7]
        assert wyckoff.encode_cif is original_encode
        assert spglib.get_symmetry_dataset is original_get_dataset
        assert cif.read_bytes() == cif_bytes
        assert facts["coordinate"].hex() == coordinate.hex()
        assert plan["coordinate"].hex() == coordinate.hex()


if __name__ == "__main__":
    test_exclusions_and_round_robin_are_deterministic()
    test_selects_224_unique_candidates_across_169_space_groups()
    test_appended_jsonl_keeps_old_prefix_bytes()
    test_coverage_splits_and_heldout_unseen_are_train_only()
    test_runtime_policy_allows_torch_and_python_patch_only()
    test_unified_sg_policy_scopes_both_encoder_calls_and_preserves_inputs()
    print("prepare_data checks passed")
