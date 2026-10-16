"""Exact MAP for cross-element fixed-Wyckoff exclusion, given profile scores.

Only labelled zero-DOF WPs are exclusive resources. Free-WP geometry and
collisions are outside this decoder. This is not a crystal stability check.
"""


def exact_fixed_site_map(element_candidates):
    """Input per element: iterable of (profile_id, log_score, fixed_site_bitmask)."""
    states = {0: (0.0, ())}
    for candidates in element_candidates:
        best_per_mask = {}
        for profile_id, score, mask in candidates:
            if mask not in best_per_mask or score > best_per_mask[mask][0]:
                best_per_mask[mask] = (float(score), int(profile_id))
        updated = {}
        for used, (old_score, chosen) in states.items():
            for mask, (score, profile_id) in best_per_mask.items():
                if used & mask:
                    continue
                union = used | mask
                total = old_score + score
                if union not in updated or total > updated[union][0]:
                    updated[union] = (total, chosen + (profile_id,))
        states = updated
    if not states:
        return None
    return max(states.values(), key=lambda x: x[0])
