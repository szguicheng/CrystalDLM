# Repair feedback full-loop evaluation

Status: complete; 7 scheduled groups per policy.

Generation success requires strict structural validity and the originally requested space group.
GRACE relaxed-space-group and proxy-ehull results are reported separately; the primary final count also requires generation success.

| Policy | Generated strict successes | GRACE post-relax valid | GRACE post-relax requested-SG matches | Final valid, matching, converged and ≤0.08 eV/atom | Unique and novel among final successes |
|---|---:|---:|---:|---:|---:|
| independent | 2/7 | 2/7 | 2/7 | 1/7 | 1/7 |
| feedback | 2/7 | 2/7 | 2/7 | 1/7 | 1/7 |

Paired loop outcomes: both 2; independent only 0; feedback only 0; neither 5.

The final main count requires the loop-selected output to have passed strict structure and requested-space-group checks before GRACE relaxation, then remain valid, preserve the requested space group, converge, and score ≤0.08 eV/atom.

Full per-arm GRACE reports, including supplementary thresholds, uniqueness, novelty, and failures, are in `scoring-summary.json`. GRACE proxy scores were post-hoc only and were not returned to the feedback loop.
