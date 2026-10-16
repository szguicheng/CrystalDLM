import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const root='/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM';
const src='/Volumes/Extention/Users/guicheng/Code/CodeX-tmp/2026-09-01/bang-w/outputs/group_meeting_draft.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(src));
console.log((await p.inspect({kind:'slide,layout',maxChars:1600})).ndjson);
const s=p.slides.add();
s.background.fill='#F5F8F5';
function text(name,txt,x,y,w,h,size,color='#17372F',bold=false){
 const o=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 o.text=txt;o.text.style={typeface:'PingFang SC',fontSize:size,color,bold,autoFit:'none'};return o;
}
text('number','11',72,34,40,20,12,'#2D6B5B',true);
text('section','模型对比',126,34,220,20,12,'#62736C',true);
text('title','晶体生成模型与我们的方案',72,66,1140,45,35,'#17372F',true);
text('subtitle','比较 tokenizer、生成范式与条件任务；我们的条件类型与输出层级尚待确定',72,113,1150,24,16,'#62736C');
const values=[
 ['模型','表示 / tokenizer','生成范式','条件生成设置','直接输出','与我们方案的关系'],
 ['CrystalFormer','Wyckoff / 元素字段\n连续坐标与晶格','自回归\n逐位点、逐字段','空间群、结构补全\n性质预测器 + MCMC 引导','完整晶体','条件生成已有先例\n拟更换编码与生成机制'],
 ['Matra-Genoa','可逆混合字段序列\n含自由坐标与晶格','自回归\n条件前缀后逐项预测','稳定性、元素系统\n支持空间群 / 位点前缀*','完整晶体','已有结构化条件生成\n拟引入 motif 与并行去噪'],
 ['WyFormer','元素 + 位点对称 + 编号\n复合 orbit 单元，无坐标','自回归级联\n无位置编码 + 集合打乱','空间群条件\n化学计量条件属未来工作','原型结构\n再补几何','集合表示接近\n无序输入仍可自回归'],
 ['WyckoffDiff','Wyckoff 位点 × 元素\n占据 / 计数离散变量','非自回归\nD3PM 离散扩散','按抽取的空间群去噪\n未做性质 / 化学式任务','原型结构\n再补几何','已有 Wyckoff 离散扩散\n新编码需区别于占据矩阵'],
 ['SymmCD','不对称单元、位点对称\n元素、连续坐标与晶格','非自回归\n离散 + 连续扩散','空间群输入，按先验采样\n未做组成 / 性质条件基准','完整晶体','已有对称压缩与扩散\n潜在差异在 motif / 条件'],
 ['DynaCrys','动态空间群、无序占据集\n学习的对称 codebook','非自回归\n符号与几何两阶段扩散','主实验为无条件生成\n几何以自生成原型为条件','完整晶体','已有集合与对称 codebook\n我们的条件任务定位不同'],
 ['MaskGXT','逐原子 / 字段 token\n几何量化 + 连续精度修正','非自回归掩码生成\n按置信度迭代并行解码','给定化学组成的 CSP\n原子类型与数量已知','完整晶体','已有条件 + 非自回归\n组成条件下的直接基线'],
 ['WyckoffDiff-\nAdaptor（补充）','沿用占据 / 计数表示\n去噪网络加入条件适配器','非自回归\n条件离散扩散','空间群 + 元素系统\n+ 凸包上方能量条件','原型结构\n再补几何','已有对称感知条件扩散\n需比较编码与条件机制'],
 ['CrystalDLM\n（我们的方案）','新 Wyckoff tokenizer\n候选：set-level BPE / motif','非自回归扩散 / DLM\n拟双向恢复 masked 单元','以条件生成为主要任务\n具体条件类型待定','待定\n原型或完整','拟验证新集合 / motif 编码\n对条件扩散生成的作用'],
];
const widths=[148,238,187,244,94,233];
const t=s.tables.add({rows:values.length,columns:6,left:68,top:148,width:1144,height:474,columnWidths:widths,values});
t.styleOptions={headerRow:false,bandedRows:false};
t.borders.assign({fill:'#C9D8D0',width:0.6,style:'solid'});
for(let r=0;r<values.length;r++){
 t.rows[r].height=r===0?33:49;
 for(let c=0;c<6;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#2D6B5B':r===9?'#DCECE3':r%2?'#F5F8F5':'#EAF0EB';
  cell.text.style={typeface:'PingFang SC',fontSize:r===0?16:15,color:r===0?'#FFFFFF':'#17372F',bold:r===0||r===9||c===0,autoFit:'none'};
 }
}
t.cells.block({row:0,column:0,rowCount:10,columnCount:6}).assign({margins:{left:8,right:7,top:5,bottom:5},anchor:'center'});
text('definitions','原型结构（protostructure）：空间群与元素占据描述，尚缺完整晶格 / 自由坐标。* 支持字段，缺少专门条件基准。',72,664,1140,22,12,'#62736C');
text('footer','依据 8 篇论文正文及相关附录（2026-09-07 核对）。完整比较与原文出处见本页备注。',72,695,1140,17,10,'#62736C');
s.speakerNotes.textFrame.setText(await fs.readFile(root+'/research/model_comparison.md','utf8'));
await (await PresentationFile.exportPptx(p)).save(root+'/.work/ppt_append/candidate.pptx');
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.items[i];
 const blob=await p.export({slide,format:'png',scale:1.25});
 await fs.writeFile(root+`/.work/ppt_append/slide-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
}
await fs.writeFile(root+'/.work/ppt_append/slide-11.layout.json',await (await s.export({format:'layout'})).text());
console.log('Exported 11 slides');
