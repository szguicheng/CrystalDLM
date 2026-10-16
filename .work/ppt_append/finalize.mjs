import fs from 'node:fs/promises';
import { finalizePresentation } from '/Volumes/Extention/Users/guicheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM';
const skill='/Volumes/Extention/Users/guicheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const r=await finalizePresentation({
 workspaceDir:root,
 candidatePath:root+'/.work/ppt_append/candidate.pptx',
 finalPath:root+'/outputs/group_meeting_with_model_comparison_final.pptx',
 pythonExecutable:'/Volumes/Extention/Users/guicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','11'],
 explicitTotalSlideCount:11,
 requiredNativeTableOwnerSlides:[11],
 verifyArtifactToolImport:true,
 receiptPath:root+'/.work/ppt_append/validation_final.json',
});
console.log(JSON.stringify(r));
