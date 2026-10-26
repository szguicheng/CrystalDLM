# Local visual revision — 2026-10-21

User feedback: the figure should compare a few representative families and depict their tokenization visually; restore the earlier checkmark table instead of putting full module descriptions in every cell.

## Changes

- Preserved the five-factor prose structure from 2026-10-18.
- Restored twelve method rows with value type, neural structural latent encoding, explicit complete geometry, and explicit crystallographic symmetry as the central comparison. Training and reconstruction are short labels; detailed algorithms remain in prose.
- Replaced seven text-flowchart rows with four illustrated panels sharing a schematic CsCl-type periodic input: explicit geometry, symmetry orbits, periodic connectivity, and learned latent encoding.
- The illustration shows record-to-symbol conversion, orbit representatives, periodic edges carrying cell-shift labels, and neural compression into a continuous vector. The orbit example uses SG 221 and sites 1a/1b. The graph sketches and chips are conceptual illustrations, not literal outputs of the cited encoders.
- Kept the figure as editable TikZ in the existing section, with no external image dependency or manuscript file reorganization.

## Verification

Tectonic builds the five-page chapter preview and 23-page local manuscript. All thirteen references resolve. The new section has no overfull/underfull boxes or bibliography warnings. Rendered pages were inspected and long labels and overlaps corrected. Full-document warnings about the existing overview PDF version/tags remain as before.

Build and render directories: `build/visual_20261021/` and `rendered/visual_20261021/`. The build manifest records refreshed output hashes. No upload or Git push was performed.
