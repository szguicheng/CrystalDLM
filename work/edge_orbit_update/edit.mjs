import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM';
const dir=root+'/work/edge_orbit_update';
await fs.copyFile(root+'/outputs/group_meeting_with_model_comparison_final.pptx',dir+'/reference.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'/reference.pptx'));
const scan=await p.inspect({kind:'slide,table,textbox',maxChars:50000});
const records=scan.ndjson.trim().split('\n').map(x=>JSON.parse(x));
const tableRecord=records.find(x=>x.kind==='table'&&x.slide===11);
const slideRecord=records.find(x=>x.kind==='slide'&&x.slide===11);
const table=p.resolve(tableRecord.id),slide=p.resolve(slideRecord.id);
const row=['CrystalDLM\n（边轨道候选）','位点约束＋边轨道\n离散原型、几何分离',
  '拟非自回归扩散\n已做并行单级去噪','已测：原型＋指定晶格\n最终应用条件待定','原型编码\n去噪位移'];
for(let i=0;i<row.length;i++) table.cells.set(9,i,row[i]);
const changes={
 definitions:'文献中的原型结构通常不含完整几何。* 条件字段已有，缺少专门基准。本方案完整生成仍拟议。',
 footer:'论文核对：2026-09-07；本方案更新：2026-09-09。原文、直接先例与实验边界见备注。'
};
for(const [name,text] of Object.entries(changes)){
 const record=records.find(x=>x.kind==='textbox'&&x.slide===11&&x.name===name);
 p.resolve(record.id).text.replace(record.text,text);
}
const notes=await fs.readFile(root+'/research/model_comparison.md','utf8')+'\n\n'+
 await fs.readFile(root+'/research/edge_orbit_tokenizer_proposal.md','utf8');
slide.speakerNotes.textFrame.setText(notes);
await (await PresentationFile.exportPptx(p)).save(dir+'/artifact_edit.pptx');
const preview=await slide.export({format:'png',scale:1.5});
await fs.writeFile(dir+'/edited_slide11.png',new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(dir+'/authored_texts.json',JSON.stringify({row,changes,notes},null,2));
await fs.writeFile(dir+'/edited_layout.json',await (await slide.export({format:'layout'})).text());
console.log(JSON.stringify({slide:slideRecord.id,table:tableRecord.id,rows:tableRecord.rows,cols:tableRecord.cols}));
