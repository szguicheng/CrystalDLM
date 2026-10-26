# Parent versus repair-trained crystal generation

The comparison uses the same 32 validation repair prompts, with eight examples per actual corruption type. Both arms receive the corrupted input in the repair prefix; the reference lead and target crystal are excluded from model context. START is sampled naturally. Success requires strict structural validity and the requested space group. ehull was not evaluated; factual readback was not run.

| Arm | Strict valid | Requested SG match | Valid and SG match | Natural START | Whitespace-normalized input-body match | Whitespace-normalized target-body match | Mean seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent | 8/32 | 22/32 | 8/32 | 32/32 | 12/32 | 5/32 | 39.49 |
| new | 27/32 | 31/32 | 27/32 | 32/32 | 0/32 | 27/32 | 28.15 |

Paired structural-and-SG outcomes: both 7, parent only 1, new only 20, neither 4

Failure categories are counted from each saved attempt receipt:

- parent: `{"ValueError: Wyckoff multiplicity does not match label": 8, "ValueError: not enough values to unpack (expected 5, got 4)": 1, "formula_header_mismatch": 7, "sites_header_mismatch": 3, "strict_invalid": 24, "wyckoff_coordinate_constraint": 8}`
- new: `{"formula_header_mismatch": 1, "strict_invalid": 5, "wyckoff_coordinate_constraint": 4}`

Per-corruption-type strict-valid and valid-plus-SG counts:

| Corruption type | Parent valid | Parent valid+SG | New valid | New valid+SG |
|---|---:|---:|---:|---:|
| formula_count | 0/8 | 0/8 | 7/8 | 7/8 |
| sites_count | 6/8 | 6/8 | 8/8 | 8/8 |
| wyckoff_multiplicity | 1/8 | 1/8 | 8/8 | 8/8 |
| coordinate_constraint | 1/8 | 1/8 | 4/8 | 4/8 |
