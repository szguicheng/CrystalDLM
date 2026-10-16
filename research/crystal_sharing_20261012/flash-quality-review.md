# Flash caption QA: raw versus final pilot30

Read-only review of the 30 raw Flash responses, supplied selected relation and graph records, and `pilot30-flash-final.jsonl`.

**Raw scientific status: 29/30. Final scientific status: 30/30.** Raw line 16 called the unclassified W CN6 environment “octahedral-like.” The final caption says W has six O neighbors and no longer assigns a motif. The remaining composition, space-group, cell/orbit-count, local coordination, sharing, periodic-graph, and archived-label claims match the supplied records.

All 30 final paragraphs have 4–5 sentences, a geometry-basis qualifier, at least one valid sharing pattern, and the exact MP label suffix. All corner/edge/face types checked out; graph dimension claims match the supplied single-component 3D graph; no material IDs remain. The correction log changes were verified at lines 2, 5, 6, 16, 19, 25, 26, and 27.

## Remaining notes

- **Line 6 punctuation — resolved:** The stray comma was removed; the final caption now reads “The structure of SmCdAg2 adopts ...” (`pilot30-flash-final.jsonl:6`).
- **Vertex-count brevity:** Lines 2, 6–9, 13–14, and 23–25 still use generic element-only labels for some unnamed sharing polyhedra. The sharing relations are supported and the captions do not invent a sharing type; explicit vertex counts would be more precise. These non-false omissions were retained for brevity, so they remain a literal style-precision exception to the supplied preference for vertex-count wording. The captions and selected relation records are on matching lines in `pilot30-flash-final.jsonl` and `pilot30-flash-annotations.jsonl`.

## Corrections verified

- Line 2 replaces “higher coordination” with F CN5 and H CN4.
- Line 16 removes the unsupported octahedral-like assignment; the base fact remains W CN6 with `recognized_motif: null` and `motif_classification: unrecognized`.
- Lines 6, 19, and 25 remove the material IDs.
- Lines 5, 16, 19, 26, and 27 explicitly attribute 3D connectivity to the independent periodic neighbor graph.

## Scope limit

I did not recompute polyhedra or sharing contacts. This QA uses the supplied validated sharing and graph records as the authority.
