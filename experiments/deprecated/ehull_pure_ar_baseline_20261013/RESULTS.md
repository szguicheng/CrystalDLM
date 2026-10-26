# Pure-AR one-epoch baseline results

Job **627890** completed on Burgundy `gpu-a100-03`: **COMPLETED, exit0:0**, elapsed **26:10**. The fixed7,272 training materials contributed one generation/reading pair each, with909 optimizer steps. Five compute-node tests passed. Training losses and gradient norms remained finite. Peak allocated GPU memory was **32.75 GiB**.

The optimizer loop took20:16; evaluation took4:05. The median training objective was0.9348 for the first50 updates and0.4503 for the last50, using different materials across updates. These are training statistics, not a held-out quality or property score.

## Unrepaired generation outcomes

| Check | Result |
|---|---:|
| AR lead reached CRYSTAL_START | 32/32 |
| Complete crystal block and END/PAD boundaries | 32/32 |
| Strict PLaID structural audit | **7/32 (21.875%)** |
| Text continuation returned for valid crystals | 7/7 |
| Four basic continuation facts matched the generated crystal | 7/7 |
| Generated-crystal property evaluation | Not run |

Strict validity uses the pinned existing parser: formula/site-count agreement, Wyckoff labels and coordinate constraints, symmetry-compatible lattice, decoding, overlap and detected-space-group checks. No output repairs were applied. A failure before decoding does not establish that later geometry checks would pass.

## First failures encountered

Reason counts overlap:19 formula-header mismatches,19 site-count-header mismatches,1 Wyckoff multiplicity mismatch and1 coordinate-constraint failure. Exclusive combinations account for all25 invalid candidates:

- 13 formula and site-count mismatches;
- 5 site-count mismatch only;
- 5 formula mismatch only;
- 1 Wyckoff multiplicity mismatch;
- 1 both header mismatches plus a coordinate-constraint failure.

For example, the candidate conditioned on sourceID `mp-1079881` declares `Mg4Ti4O10` (18 atoms) and `Sites (16)`, while its actual site records imply `Mg4Ti4O6` (14 atoms). The headers and Wyckoff-body counts disagree. Evaluation source IDs identify prompts; they are not material IDs assigned to generated candidates.

## Concentrated output families

The32 prompts contain26 distinct target ehull values, from0 to0.2633054 eV/atom. All32 raw crystal texts differ, but only two composition/space-group headers appear:

| Input target | Requests | Generated header | Strict-valid outputs |
|---|---:|---|---:|
| ehull=0 |7 | Cs8Tl4Au4Br24 / Fm-3m |7 |
| ehull>0 |25 | Mg4Ti4O10 / C2/m |0 |

There are four lead paragraphs overall. The seven valid candidates share the same lead, tail, composition and space group, while lattice and Br coordinates vary. This is observed concentration under greedy lead decoding. The pattern distinguishes zero and positive requests; it is not evidence that the model ignores every condition or that generated structures achieve the requested property.

The valid reduced composition Cs2TlAuBr6 occurs0 times in the frozen train/validation set. Mg2Ti2O5 occurs once in training, in an18-atom Pbam/No.55 record whose caption and crystal header both correctly state18 atoms. The generated C2/m/16-atom combination differs from that source. The observed output concentration therefore cannot be explained as either exact family dominating this dataset.

## What continuation agreement establishes

For the seven valid candidates, continuation formula, space-group number225, atom count40 and orbit count4 match their actual generated structures. Because these candidates share the same plan and family, this does not demonstrate that the reader depends on the crystal rather than the earlier plan. Swapped-structure controls were not run. Richer coordination, sharing and connectivity claims in the lead were not recomputed on generated structures.

## Current conclusion

The pure-AR route learned the modality boundaries and can produce some strictly valid crystals. This first baseline has unresolved cross-field consistency and output-diversity limitations. Target-property satisfaction and causal structure feedback remain unverified. Before interpreting a joint-training comparison, preserve this raw baseline and investigate its failures; do not count source or target ehull as a generated-crystal score.

Evidence: `run_001/summary.json`, `run_001/evaluation.jsonl`, `run_001/updates.jsonl`, fixed IDs/order and initialization records, `result-audit.json`, compute-node logs. The full18.5GB checkpoint remains on Burgundy.
