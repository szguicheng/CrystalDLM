import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM';
const skill='/Volumes/Extention/Users/guicheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const work=root+'/work/edge_orbit_update';
const runtime='/Volumes/Extention/Users/guicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3';
process.env.RUNTIME_NODE_MODULES='/Volumes/Extention/Users/guicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE='/Volumes/Extention/Users/guicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node';
process.env.RUNTIME_PYTHON=runtime;
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const preserved=JSON.parse(await fs.readFile(work+'/preservation.json','utf8'));
const finalPath=root+'/outputs/group_meeting_edge_orbit_candidate_20260909.pptx';
const result=await finalizePresentation({workspaceDir:root,candidatePath:work+'/candidate.pptx',finalPath,
 pythonExecutable:runtime,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','11'],
 explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[11],verifyArtifactToolImport:true,
 fontPolicy:{basis:'reference',families:preserved.font_families,referencePath:work+'/reference.pptx',referenceSha256:preserved.reference_sha256},
 receiptPath:work+'/validation.json'});
console.log(JSON.stringify(result));
const p=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<11;i++){
 const blob=await p.slides.getItem(i).export({format:'png',scale:1});
 await fs.writeFile(work+`/final_slide${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
}
const montage=await p.export({format:'webp',montage:true,scale:.3});
await fs.writeFile(work+'/montage.webp',new Uint8Array(await montage.arrayBuffer()));
