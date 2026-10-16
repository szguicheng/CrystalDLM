import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM';
const work=root+'/work/meeting_followup_concise_20260914';
const skill='/Volumes/Extention/Users/guicheng/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations';
const rt='/Volumes/Extention/Users/guicheng/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=rt+'/node/node_modules';process.env.RUNTIME_NODE=rt+'/node/bin/node';process.env.RUNTIME_PYTHON=rt+'/python/bin/python3';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const m=JSON.parse(await fs.readFile(work+'/manifest.json','utf8'));
const finalPath=work+'/validated/Crystal DLM 2 final.pptx';await fs.mkdir(work+'/validated',{recursive:true});
const receipt=await finalizePresentation({workspaceDir:root,candidatePath:work+'/candidate.pptx',finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...m.tableOwners.flatMap(i=>['--require-native-table-slide',String(i)])],
 explicitTotalSlideCount:m.total,requiredNativeTableOwnerSlides:m.tableOwners,verifyArtifactToolImport:true,
 receiptPath:work+'/validation-final.json'});
console.log(JSON.stringify(receipt));
const p=await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(let i=0;i<p.slides.items.length;i++){
const b=await p.slides.getItem(i).export({format:'png',scale:1});await fs.writeFile(work+`/final-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));
}
const b=await p.export({format:'webp',montage:true,scale:.22});await fs.writeFile(work+'/montage.webp',new Uint8Array(await b.arrayBuffer()));
