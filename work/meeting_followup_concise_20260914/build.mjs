import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const dir=new URL('.',import.meta.url).pathname;
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'reference.pptx'));
if(p.slides.items.length!==15)throw Error('Expected 15 original slides');
const refs=JSON.parse(await fs.readFile(dir+'references.json','utf8'));
const C={red:'#B60000',ink:'#17372F',green:'#2D6B5B',muted:'#56665F'};
const manifest=[],tableOwners=[];
function tx(s,value,x,y,w,h,size=28,color=C.ink,bold=false){const o=s.shapes.add({name:`text-${s.shapes.items.length}`,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});o.text=value;o.text.style={typeface:'PingFang SC',fontSize:size,color,bold,autoFit:'none'};return o;}
function page(title,answer,keys,notes=''){
 const s=p.slides.add();s.background.fill='#FFFFFF';const n=p.slides.items.length;
 tx(s,title,80,58,1120,65,40,C.red,true);
 tx(s,answer,82,143,1110,80,29,C.ink,true);
 tx(s,String(n),1180,675,50,24,17,'#222222');
 s.speakerNotes.textFrame.setText([notes,...keys.map(k=>refs[k])].join('\n\n'));
 manifest.push({slide:n,title,sources:keys});return s;
}
function table(s,values,widths,{top=250,height=320,size=25,rows=null}={}){
 const t=s.tables.add({rows:values.length,columns:widths.length,left:80,top,width:1120,height,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:'#B8C9BE',width:.7,style:'solid'});
 for(let r=0;r<values.length;r++){t.rows[r].height=rows?rows[r]:height/values.length;for(let c=0;c<widths.length;c++){const a=t.getCell(r,c);a.fill=r===0?C.green:r%2?'#F7FAF7':'#EDF3EE';a.text.style={typeface:'PingFang SC',fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0||c===0,autoFit:'none'};}}
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:widths.length}).assign({margins:{left:14,right:12,top:8,bottom:7},anchor:'center'});
 tableOwners.push(p.slides.items.length);
}
function foot(s,value,y=641){tx(s,value,82,y,1090,44,19,C.muted);}

let s=page('① 这种材料基础大模型，有人做过吗？','有。把材料、性质和文字放在一起训练，已有直接相关工作。',['nature','llamat','alm'],
'Targeted search current to 2026-09-14. These three papers are the closest examples selected from the fuller survey. No claim of exhaustive prior-art exclusion. LLaMat has Chat/CIF adaptations; NatureLM-Mat3D is separately adapted; ALM Core/Edit/Gen are staged variants.');
table(s,[['最相关的工作','已经做了什么','和老师设想的关系'],
 ['NatureLM\n2025','材料组成、空间群、性质与文字\n都写成序列，放在一起训练','最接近“都放进 text space”\n完整晶体用 Mat3D 另行训练'],
 ['LLaMat\n2024 / 2026','材料论文和晶体文件参与训练\n再训练问答、提取、生成等任务','没有晶体结构的论文也能用\n但不同任务有不同模型版本'],
 ['ALM（2026）','同一理解模型训练结构任务\n和纯文本任务，再接扩散生成器','最接近“有结构 / 无结构都用”\n但不是全部变成离散文字']
],[250,435,435],{height:316,rows:[46,90,90,90],size:25});
tx(s,'所以，我们需要说明：自己的表示或训练方法，究竟多解决了什么问题。',82,593,1100,72,27,C.green,true);

s=page('① 没有结构、没有性质的数据，能不能用？','能用。每条数据只训练它实际提供的内容。',['nature','alm','multi'],
'This is a proposed CrystalDLM training organization, not an implemented experiment. A naturally absent target has no supervised label. Randomly masked observed fields do have ground truth. Cross-field learning still needs paired data. Distinguish formation energy, total energy and hull energy, units, computation/measurement conditions, and polymorph identity.');
table(s,[['这条数据有什么','可以拿它训练什么'],
 ['组成 + 能量，没有晶体结构','根据组成预测能量'],
 ['晶体结构，没有性质标签','遮住一部分结构，让模型补回来'],
 ['结构 + 部分性质','遮住已有答案的一部分，让模型来预测'],
 ['只有材料文字','学习材料知识、文字补全或问答']
],[480,640],{top:251,height:310,rows:[46,66,66,66,66],size:27});
tx(s,'不知道的值不能当成 0，也不能编一个结构当答案。',82,591,1110,43,28,C.green,true);
foot(s,'同用文字只是统一输入格式。还需要有对应关系的数据，模型才能学会“这种材料对应什么性质”。');

s=page('② 4a 里的“4”和“a”是什么意思？','例子：NaCl，空间群 #225（Fm−3m），使用常规立方晶胞。',['iucr','wp'],
'Standard rock-salt assignment with full occupancy. Conventional fcc cell contains four NaCl formula units; primitive cell contains one. Wyckoff multiplicity is per conventional cell. Letter a is local to the WP table for the specified space group and setting. All listed coordinates are fractional and reduced modulo 1.');
tx(s,'4',94,251,140,100,80,C.green,true);tx(s,'一个代表点按对称规则展开，\n在一个常规晶胞里得到 4 个位置。',245,265,410,92,26);
tx(s,'a',695,251,100,100,80,C.green,true);tx(s,'该空间群位点表里的标签。\n它不是元素，也不是坐标。',820,265,380,92,26);
table(s,[['Na 占据 4a','Cl 占据 4b'],
 ['代表点：(0,0,0)','代表点：(½,½,½)'],
 ['全部位置：\n(0,0,0)   (0,½,½)\n(½,0,½)   (½,½,0)','全部位置：\n(½,½,½)   (½,0,0)\n(0,½,0)   (0,0,½)']
],[560,560],{top:380,height:225,rows:[47,55,123],size:26});
foot(s,'两条位点记录就能表示 4 个 Na 和 4 个 Cl。这里的 4 是位置数，不是 Na 的配位数 6。',641);

s=page('② 一个 Wyckoff 位点，可以在哪些地方？','有的位点固定不动；有的可以改变一个或两个坐标。',['iucr','wp','local'],
'The Pnma numerical example is illustrative, not a measured structure. SG139 4e standard representative (0,0,z) at z=.290407 is equivalent to the stored mp-30009 representative (.5,.5,.790407). WP family parameters are fractional coordinates. Changes must avoid special values that change orbit multiplicity.');
table(s,[['例子','代表位置','还有多少坐标可以选择'],
 ['NaCl：#225 的 4a','(0, 0, 0)','0 个：位置已经固定'],
 ['KrF₂ 中的 F：#139 的 4e','(0, 0, z)','1 个：只需要选 z'],
 ['#62 Pnma 的 4c','(x, ¼, z)','2 个：可以选 x 和 z，y 固定']
],[435,285,400],{top:249,height:276,rows:[48,76,76,76],size:26});
tx(s,'实际例子：KrF₂ 的 F 可以取 z = 0.290407。\n写下 4e，就确定了允许的位置；再给 z，才确定具体坐标。',82,558,1100,86,28);
foot(s,'4e 展开后有 4 个等价位置。4 表示位置数量，1 表示需要选择的自由参数数量。',655);

s=page('③ 不同空间群里的“4a”，是同一个位点吗？','不是。同名 4a 可以有不同的对称规则，甚至不同的可移动范围。',['iucr','wp','wyformer'],
'All examples use standard settings. Site symmetries: SG225 4a m-3m, SG216 4a -43m, SG62 4a -1, SG19 4a 1. SG225 and SG216 4a coordinate sets coincide but their site symmetries differ. SG216 4c fixed (.25,.25,.25) versus SG62 4c (x,.25,z). Shared features may be reused while retaining SG/Hall and WP identity.');
table(s,[['空间群','同样叫 4a，代表位置却可以是','这个点允许的局部对称'],
 ['#225  Fm−3m','(0,0,0)，固定','m−3m'],
 ['#216  F−43m','(0,0,0)，固定','−43m'],
 ['#62  Pnma','(0,0,0)，固定','−1（反演）'],
 ['#19  P2₁2₁2₁','(x,y,z)，三个坐标都可以选','1（只有恒等操作）']
],[335,485,300],{top:245,height:308,rows:[48,65,65,65,65],size:25});
tx(s,'另一个例子：#216 的 4c 是固定点 (¼,¼,¼)；\n#62 的 4c 是 (x,¼,z)，还能选 x 和 z。',82,578,1100,76,27);
foot(s,'因此不能只用“4a”当作通用身份。可以共享对称特征，但还要保留空间群和具体位点。',658);

s=page('④ 边轨道 tokenizer，到底记录什么？','把对称重复的原子和连接各存一份，再用规则展开。',['local','prior'],
'Actual codec fields: global space_group/hall_number/operations; nodes with Z, wyckoff_letter, affine stabilizer (operation,image) and dof; edge_tokens as [i,j,g,nx,ny,nz]; geometry with lattice and free_vectors. Site stabilizers explicitly encode allowed representative coordinates and can be derived from WP/setting/representative. The current codec is not a trained LM vocabulary.');
table(s,[['组件','记录什么','为什么需要'],
 ['全局规则','空间群、坐标设置、对称操作表','告诉模型怎样复制原子和连接'],
 ['位点记录','元素、Wyckoff 位点、位置约束','同一组等价原子，只记一个代表'],
 ['连接记录','从哪个位点，连到哪个对称像\n以及哪个相邻晶胞','指出具体连到谁；等价连接合并存'],
 ['具体几何','晶格大小 / 形状、可变坐标','把抽象的位点和连接变成实际晶体']
],[205,455,460],{top:245,height:330,rows:[46,67,67,83,67],size:25});
tx(s,'编码顺序：全局规则 + 位点记录 + 连接记录，另存具体几何。',82,604,1110,54,27,C.green,true);
foot(s,'当前记录的是 4.5 Å 内的近邻关系，不能把每条连接都叫作化学键。',662);

s=page('④ 一条真实记录，怎样读成一条连接？','KrF₂ 的实际记录：[0, 1, 16, −1, −1, −1]',['local'],
'Recomputed local example in work/meeting_followup_20260914/example_verified.json. g=16 has R=I and t=(.5,.5,.5). Source Kr q0=(0,0,0), target F representative q1=(.5,.5,.790407), lattice diag(4.383300,4.383300,6.668202) Å. Endpoint (0,0,.290407), separation 1.93649 Å. Seven orbits recover 44 undirected periodic edges, or 88 directed entries after explicit reverse expansion. Grouping sizes [4,16,4,8,2,2,8]. 44/7 is not whole-payload token compression. Fixed abstract connectivity need not remain the same cutoff graph if geometry changes. Multi-step generation remains unimplemented.');
table(s,[['0、1','从 Kr 的代表点，连到 F 的代表点的一个复制像'],
 ['16','查操作表：把 F 的坐标加上 (½,½,½)'],
 ['−1、−1、−1','再沿三个晶格方向，各退一个晶胞']
],[265,855],{top:241,height:180,rows:[60,60,60],size:27});
tx(s,'这里选 F 的等价代表点 (½,½,0.790407)\n加上 (½,½,½)，再减去 (1,1,1)\n得到实际邻居 (0,0,0.290407)，距 Kr 约 1.93649 Å。',82,450,1100,124,28);
tx(s,'这个晶体：2 条位点记录展开成 6 个原子，\n7 条连接记录展开成 44 条无向周期边。',82,589,1100,75,28,C.green,true);
foot(s,'已验证的是编码和解码。完整生成模型还没有完成；操作编号 16 只在本份操作表里有意义。',667);

await fs.writeFile(dir+'manifest.json',JSON.stringify({original:15,total:p.slides.items.length,slides:manifest,tableOwners},null,2));
await (await PresentationFile.exportPptx(p)).save(dir+'authored.pptx');
for(let i=15;i<p.slides.items.length;i++){const b=await p.slides.getItem(i).export({format:'png',scale:1});await fs.writeFile(dir+`draft-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(JSON.stringify({total:p.slides.items.length,added:manifest.length}));
