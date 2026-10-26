# Structure-feedback loop pilot

Status: complete
Authorized: 2026-10-20

Reuse the completed unified FLARE checkpoint, compare three-attempt independent resampling with structure-feedback retries. Seven space groups × ten paired seed groups. No training; no GRACE feedback or property scoring in this pilot. Keep the original requested space group and ehull target fixed.

Protocol and implementation plan: [PROTOCOL.md](../../../experiments/structure_feedback_loop_20261020/PROTOCOL.md).

Completed: formal70pairedgroups finished2026-10-20 20:27:18 Asia/Shanghai, bothworker/supervisor exits0, GPUsreleased. Independent19/70 vsfeedback9/70; feedback122retries,117noSTART(82tokenlimit,35EOS),5generatedcrystalsallstrictinvalid. Allplannedgroups,pairedseeds,sharedinitial,frozenstateidentities andsummarycountsverified. NoGRACEortraining. See [results](../../../experiments/structure_feedback_loop_20261020/RESULTS.md). Further diagnosis not started.
