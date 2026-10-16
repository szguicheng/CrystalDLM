# Matched B16 diffusion control

Relative to the completed `interleaved_relational_blocks_20261005/pilot_001/diffusion_b32` run, the only changed factor is crystal block size, B32 to B16. Both runs use clean-crystal AR weight `0.0`, with no auxiliary AR group.

Keep `relational_cnl_v1`, the frozen 8-seen/16-heldout cohort and source records, native tokenizer, Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, fresh seed 17, 512 updates, effective batch 4, alternating Q1/Q2 schedule, `K=2560`, `STEPS=32`, `MAXLEN=8192`, prediction offset 0, training sampler, and strict scorer unchanged. Preserve the seen8 generation gate; do not evaluate heldout when seen8 generation is zero.
