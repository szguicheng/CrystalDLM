# Feedback modality switch diagnosis and fix

Status: complete
Authorized: 2026-10-20

User requested causal diagnosis and repair of retry AR-to-diffusion switching. Fixed14-case probes reproduced missingSTART onbothmatchedARandFLARE (0/14);cap768recoveredonly1/14FLARE;freshsingle-usergenerationrequestwithfailedcrystalasdatarecovered14/14naturally.

Fix changes only context serialization, preserving originaltarget/failedstructure/checkerfeedback. No forcedSTART,training,orGRACE. Full70groupvalidation onGPU0/1 started23:00:35Asia/Shanghai; originalfirst/directretryreceipts reusedverbatimwithpinnedSHA;diagnostic14andheldout56separatelyreported.

See [diagnosis](../../../experiments/structure_feedback_loop_20261020/switch_fix/DIAGNOSIS.md) and [protocol](../../../experiments/structure_feedback_loop_20261020/switch_fix/PROTOCOL.md). Completed2026-10-20 23:45:43;70/70groups,allprocessesexit0,GPUsreleased. NaturalSTART122/122,strictvalidnew0,feedback9/70vsdirect19/70;heldout56naturalSTART94/94,newvalid0. 67/122newcrystaltokensexactlyrepeatprevious;105/122retainanauditreason. Code/contextswitchfixverified;structuralrepairnotimproved. No furtherexperimentstarted. See [fullresults](../../../experiments/structure_feedback_loop_20261020/switch_fix/RESULTS.md).
