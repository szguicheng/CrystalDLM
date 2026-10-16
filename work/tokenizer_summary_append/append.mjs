import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/Volumes/Extention/Users/guicheng/Code/helloWorld/CrystalDLM';
const dir=root+'/work/tokenizer_summary_append';
await fs.copyFile(root+'/outputs/group_meeting_edge_orbit_candidate_20260909.pptx',dir+'/reference.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'/reference.pptx'));
const n=p.slides.items.length;
if(n!==12) throw new Error(`Source changed: ${n} slides`);
await fs.writeFile(dir+'/source-inspect.ndjson',(await p.inspect({kind:'slide,layout',maxChars:16000})).ndjson);
for(const i of [10,11]){
 const blob=await p.slides.getItem(i).export({format:'png',scale:1});
 await fs.writeFile(dir+`/source-${i+1}.png`,new Uint8Array(await blob.arrayBuffer()));
}
const s=p.slides.add();s.background.fill='#F5F8F5';
function txt(name,value,x,y,w,h,size,color='#17372F',bold=false){
 const o=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 o.text=value;o.text.style={typeface:'PingFang SC',fontSize:size,color,bold,autoFit:'none'};
}
txt('number','13',80,34,40,20,12,'#2D6B5B');
txt('section','探索结果',135,34,180,20,12,'#62736C');
txt('title','Tokenizer 方案探索与实验结果',80,68,1120,48,35,'#17372F',true);
txt('summary','8 个具体方案：6 个做了模型实验，1 个只做词表计算，1 个停在文献筛查',80,119,1120,28,19,'#62736C');
const values=[
 ['方案','分词 / 表示方法','验证阶段','结果与判断'],
 ['1  完整占位配置大 token','同一元素的整套 WP 占位配置\n独立成词','词表计算','扫描 230 个空间群，部分组合数量爆炸\n不采用全覆盖的大词表'],
 ['2  完整占位的组合式编码','将各 WP 占位记录组合\n编码成完整配置表示','模型对照','相同联合解码下，匹配率仅增 0.20 个百分点\n未确认预测收益'],
 ['3  无序 WP 集合合并','将常共同出现的同元素\nWP 轨道合成一个 token','可逆编码＋模型','可以压缩，未证明优于简单计数编码\n随后调整为计数块方案'],
 ['4  WP 计数块合并','先记录 WP 与出现次数\n再合并常见的多条计数记录','三种子模型＋测试','测试编码单元减少约 9.6%\n预测质量无稳定提升，停止扩大实验'],
 ['5  重数组合＋WP 分配','先按对称等价原子组的大小汇总\n再补充具体 WP 身份','模型分解＋消融','层级概率形式未优于共同打分\n未确认按重数聚合的收益'],
 ['6  位点类型计数＋WP 分配','先按位点点群类型汇总占位\n再补充具体 WP 身份','编码检查＋模型','未达到预定的模型收益门槛\n停止扩大该分支'],
 ['7  群–子群 / WP 分裂动作','将子群选择与 WP 分裂\n编码为生成动作','仅文献筛查','已有工作明显覆盖相关思路\n未进入实现和训练'],
 ['8  空间群边轨道 tokenizer','合并对称等价的邻接关系\n离散原型与连续几何分开','编码＋模型＋去噪','保留为候选，指定 ASU 对照下显存少 44.6%\n耗时多 11.7%，完整生成尚待验证'],
];
const t=s.tables.add({rows:9,columns:4,left:80,top:162,width:1120,height:468,columnWidths:[276,290,168,386],values});
t.styleOptions={headerRow:false,bandedRows:false};
t.borders.assign({fill:'#AFC5BB',width:.6,style:'solid'});
for(let r=0;r<9;r++){
 t.rows[r].height=r===0?36:54;
 for(let c=0;c<4;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#2D6B5B':r===8?'#DCECE3':r%2?'#F5F8F5':'#EAF0EB';
  cell.text.style={typeface:'PingFang SC',fontSize:r===0?18:16,color:r===0?'#FFFFFF':'#17372F',bold:r===0||r===8||c===0,autoFit:'none'};
 }
}
t.cells.block({row:0,column:0,rowCount:9,columnCount:4}).assign({margins:{left:9,right:8,top:5,bottom:5},anchor:'center'});
txt('counting','统计包含最初词表检查。③④为同一大方向的两版方案，合并后为 7 大类。训练种子与实现优化不另计。',80,651,1120,25,14,'#62736C');
txt('limits','WP = Wyckoff 位置。各行实验任务与预算不同，不能直接横比准确率。实验记录与比较范围见本页备注。',80,683,1120,20,12,'#62736C');
const sources=[
 'experiments/wyckoff_vocab_gate1/RESULTS.md',
 'experiments/wyckoff_tokenizer_search/PILOT_RESULTS.md',
 'experiments/wyckoff_tokenizer_search/WORD_RESULTS.md',
 'experiments/wyckoff_tokenizer_search/DIRECT_WORD_RESULTS.md',
 'experiments/wyckoff_tokenizer_search/COUNT_BLOCK_ASSESSMENT.md',
 'experiments/wyckoff_tokenizer_search/PACKED_AND_ITERATIVE_RESULTS.md',
 'experiments/wyckoff_tokenizer_search/MULTIPLICITY_RESULTS.md',
 'experiments/wyckoff_tokenizer_search/SITE_MODEL_RESULTS.md',
 'research/notes/subgroup_splitting_token_prior_art.md',
 'experiments/wyckoff_tokenizer_search/ASU_CONTROL_RESULTS.md',
 'experiments/wyckoff_tokenizer_search/PROTOTYPE_REPLICATION_RESULTS.md',
];
let notes='本页总结 2026-09-08 至 09-09 的有记录方案，包含先行词表检查。六条路线进行过不同规模模型实验，一条只计数，一条只查文献。不是八种已验证有效的新 tokenizer。\n\n边轨道资源结果：512 图、同权重径向网络、相同 ASU 节点，比较逐邻居与边轨道路径的前后向计算，不含在线构图。输出与梯度一致，峰值 CUDA allocated memory 减少约 44.6%，耗时增加约 11.7%。条件去噪已有独立复核，但完整离散 token 生成及完整扩散采样尚未完成。\n\n';
for(const f of sources) notes+=`\n来源：${f}\n`+await fs.readFile(root+'/'+f,'utf8')+'\n';
s.speakerNotes.textFrame.setText(notes);
await (await PresentationFile.exportPptx(p)).save(dir+'/authored.pptx');
await fs.writeFile(dir+'/slide13.png',new Uint8Array(await (await s.export({format:'png',scale:1.5})).arrayBuffer()));
await fs.writeFile(dir+'/slide13.layout.json',await (await s.export({format:'layout'})).text());
console.log('Authored 13 slides');
