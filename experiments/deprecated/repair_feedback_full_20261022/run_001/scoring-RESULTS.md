# Repair feedback full-loop evaluation

Status: complete; 700 scheduled groups per policy.

Generation success requires strict structural validity and the originally requested space group.
GRACE relaxed-space-group and proxy-ehull results are reported separately; the primary final count also requires generation success.

| Policy | Generated strict successes | GRACE post-relax valid | GRACE post-relax requested-SG matches | Final valid, matching, converged and ≤0.08 eV/atom | Unique and novel among final successes |
|---|---:|---:|---:|---:|---:|
| independent | 337/700 | 337/700 | 313/700 | 55/700 | 49/700 |
| feedback | 224/700 | 225/700 | 201/700 | 28/700 | 26/700 |

Paired loop outcomes: both 201; independent only 136; feedback only 23; neither 340.

The final main count requires the loop-selected output to have passed strict structure and requested-space-group checks before GRACE relaxation, then remain valid, preserve the requested space group, converge, and score ≤0.08 eV/atom.

Full per-arm GRACE reports, including supplementary thresholds, uniqueness, novelty, and failures, are in `scoring-summary.json`. GRACE proxy scores were post-hoc only and were not returned to the feedback loop.
