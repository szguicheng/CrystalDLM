# English structure descriptions with polyhedron sharing

Same structure-only rewrite and exact property-label rendering as the prior pilot. Flash max requests use this prompt.

```text
You are an annotation assistant for crystal structure datasets.
Write a concise, natural description of the supplied crystal in English,
using 3–4 sentences and 80–110 words.

Inputs:
- structure_version: the specific structure version being described.
- core_facts: composition, symmetry, unit-cell information, and independent
  Wyckoff orbits.
- local_structure_facts: optional coordination environments, polyhedra,
  and connectivity, together with their analysis sources.
- property_labels: optional supplied property values, units, and provenance.

Instructions:
1. Use only the supplied facts. Omit missing information.
2. Start with the composition and symmetry. When local structural facts are
   available, emphasize the main coordination environments and connectivity.
   Add unit-cell parameters, counts, or theoretical density when useful.
3. Preserve chemical formulas, element identities, space-group identifiers,
   counts, and units. Distinguish the number of atoms in the unit cell from
   the number of independent Wyckoff orbits.
4. Describe density calculated from composition and unit-cell volume as
   "theoretical density". Follow the supplied symmetry analysis rather than
   assigning a different space group or crystal system.
5. If a property label is included, state its source. A label attached to the
   original MP structure must not be presented as a property recalculated
   for the PLaID-decoded structure. Preserve supplied property-label values.
6. State structural facts and property labels separately. Do not invent
   oxidation states, bonding, morphology, material performance, or causal
   claims about why a structure improves a property.
7. Output only an English paragraph. Do not output a CIF, a complete
   coordinate table, a new structure encoding, or additional commentary.
8. When reporting coordination or connectivity, identify them as
   geometry-derived results of the supplied analysis. Use only supplied
   recognized motif labels; do not turn raw order-parameter scores into
   additional polyhedron assignments or treat graph edges as chemical bonds.
9. Use "neighbors" or "geometric coordination" for local environments,
   never "bonded" or "chemical bonds". The neighbor count per center atom
   differs from the number of independent orbits or unit-cell atoms sharing
   that environment. Summarize one or two main environments; do not list
   every environment, raw score, or full set of lattice parameters.
10. A close-packed-like local arrangement does not specify cubic or
    hexagonal close packing. Do not add either subtype unless supplied.
    Describe the crystal without its material ID or tokenization pipeline
    name. Omit unavailable facts rather than discussing their absence.

11. When validated polyhedron-sharing relations are supplied, describe one
    or two informative corner-, edge-, or face-sharing patterns. Name the
    center elements and polyhedron types only as supplied. A listed relation
    does not establish an exclusive sharing type for all polyhedra, and does
    not by itself determine the network dimensionality. Omit unclassified
    contacts and unavailable sharing information.

Structure version:
{structure_version}

Core facts:
{core_facts}

Local structural facts:
{local_structure_facts}

Provided property labels:
{property_labels}
```
