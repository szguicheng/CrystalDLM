import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const dir=new URL('.',import.meta.url).pathname;
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'reference.pptx'));
if(p.slides.items.length!==15)throw Error('Expected 15 source slides');
const C={red:'#B60000',ink:'#17372F',green:'#2D6B5B',muted:'#56665F',pale:'#EDF3EE'};
const refs={
 nature:'NatureLM team (2025). Nature Language Model: Deciphering the Language of Nature for Scientific Discovery. arXiv:2502.07527v2. https://arxiv.org/html/2502.07527v2 — §2.1, §5.1–5.4, §8.3.',
 llamat:'LLaMat: A family of large language models for materials research with insights into model adaptability in continued pretraining. Nature Machine Intelligence 8, 435–448 (2026). https://doi.org/10.1038/s42256-026-01199-8 . Methods checked against preprint https://arxiv.org/html/2412.09560v1 §4.1.2, §4.2.1–4.2.3. Journal metadata and preprint methods are distinguished.',
 alm:'Atomistic Language Models Understand and Generate Materials (2026), arXiv:2606.21395v1. https://arxiv.org/html/2606.21395v1 — §2.1–2.3, Appendix B.1 Table 15 and B.2 Table 17. Preprint. Core/Edit/Gen are related staged variants, not proof of arbitrary missing-field joint training.',
 matter:'MatterChat: A multimodal large language model for materials science. Nature Machine Intelligence 8, 588–601 (2026). https://doi.org/10.1038/s42256-026-01214-y — Methods, Dataset curation and Discussion.',
 multi:'MultiMat: Multimodal Learning for Materials. https://arxiv.org/html/2312.00111v3 — §IV.2, missing-entry loss masking. Published as Multimodal foundation models for material property prediction and discovery, Newton (2025). Official code https://github.com/vmoro1/multimat . Crystal/text available for every material in the studied dataset; DOS/charge density partially missing.',
 bind:'MatBind: A Shared Embedding Space for Multimodal Materials Characterization (2026). https://arxiv.org/html/2607.08470v1 — §5.1 Table 1 and §5.3. Preprint. Structure and pXRD available for all records; DOS and text are partial. Shared embedding retrieval is not de novo generation.',
 atom:'Choudhary (2024). AtomGPT: Atomistic Generative Pre-trained Transformer for Forward and Inverse Materials Design. https://arxiv.org/html/2405.03680v1 — Methods, separate GPT-2 forward regressor and Mistral inverse model. https://github.com/atomgptlab/atomgpt .',
 pi:'Bone et al. Discovery and recovery of crystalline materials with property-conditioned transformers. https://arxiv.org/html/2511.21299v1 — §II.1–II.2, §IV.1, Appendix V. CrystaLLM-π, arXiv record first posted 2025. Unlabelled CIF pretraining followed by labelled property/XRD conditioning. Continuous condition embeddings, autoregressive CIF output.',
 cloud:'CLOUD: A Scalable and Physics-Informed Foundation Model for Crystal Representation Learning (2026). Nature Communications. https://doi.org/10.1038/s41467-026-70467-3 — Fig.1, Results: SCOPE crystal representation. Coordinate-free symmetry strings, masked pretraining, downstream property prediction.',
 cham:'Exploration of crystal chemical space using text-guided generative artificial intelligence (2025). Nature Communications. https://doi.org/10.1038/s41467-025-59636-y — Fig.1 and Methods: Crystal CLIP plus text-conditioned diffusion, Chameleon.',
 gruver:'Gruver et al. (2024). Fine-Tuned Language Models Generate Stable Inorganic Materials as Text. https://arxiv.org/abs/2402.04379 — unconditional generation, infilling, text-conditioned generation. Included as an early counterexample, not an all-task foundation-model claim.',
 iucr:'IUCr Online Dictionary, Wyckoff position: https://dictionary.iucr.org/Wyckoff_position . IUCr CIF dictionary, Wyckoff multiplicity: https://www.iucr.org/__data/iucr/cifdic_html/3_orig/CORE_DIC/Ispace_group_Wyckoff.multiplicity.html . Multiplicity refers to the conventional cell.',
 wp:'Coordinate lists: PyXtal database https://raw.githubusercontent.com/qzhu2017/PyXtal/master/pyxtal/database/wyckoff_list.csv . Independently evaluated against the project-pinned PyXtal 1.1.4 database: experiments/wyckoff_vocab_gate1/input/wyckoff_list.csv; provenance.json includes wheel and file SHA256. Standard settings throughout; coordinates reduced modulo integers.',
 wyformer:'Kazeev et al. Wyckoff Transformer: Generation of Symmetric Crystals. https://arxiv.org/abs/2503.02407v4 . Local prior analysis research/model_comparison.md. Oriented site-symmetry and enumeration require the space-group context.',
 local:'Local implementation and evidence, rechecked 2026-09-14: work/edge_orbit_release/examples/mp-30009.prototype.json; mp-30009.geometry.json; codec/discrete_graph_prototype.py (split, chart, representatives, expand); codec/symmetry_graph_codec.py (edge_key, transform). Outputs in work/meeting_followup_20260914/example_verified.json. research/edge_orbit_tokenizer_proposal.md. The codec uses 4.5 Å cutoff and canonical UNDIRECTED periodic edges: 44 undirected edges, or 88 after adding reverse directed edges. No new model training was run.',
 prior:'Representation prior art: SLICES https://doi.org/10.1038/s41467-023-42870-7 ; SLICES-PLUS https://arxiv.org/abs/2410.22828 ; coGN https://doi.org/10.1039/D4DD00018H . Previously audited in research/notes/ and research/edge_orbit_tokenizer_proposal.md. This supplement does not re-audit every prior-art implementation.'
};
const manifest=[];const tableOwners=[];
function tx(s,value,x,y,w,h,size=26,color=C.ink,bold=false){const o=s.shapes.add({name:`text-${s.shapes.items.length}`,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});o.text=value;o.text.style={typeface:'PingFang SC',fontSize:size,color,bold,autoFit:'none'};return o;}
function page(title,sub,keys,notes=''){
 const s=p.slides.add();s.background.fill='#FFFFFF';const n=p.slides.items.length;
 tx(s,title,80,57,1120,68,39,C.red,true);if(sub)tx(s,sub,82,132,1110,52,22,C.muted);
 tx(s,String(n),1180,675,50,24,17,'#222222');
 s.speakerNotes.textFrame.setText([notes,...keys.map(k=>refs[k])].filter(Boolean).join('\n\n'));
 manifest.push({slide:n,title,sources:keys});return s;
}
function table(s,values,widths,{top=202,height=370,size=22,rows=null}={}){
 const t=s.tables.add({rows:values.length,columns:widths.length,left:80,top,width:1120,height,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:'#B8C9BE',width:.7,style:'solid'});
 for(let r=0;r<values.length;r++){t.rows[r].height=rows?rows[r]:height/values.length;for(let c=0;c<widths.length;c++){const a=t.getCell(r,c);a.fill=r===0?C.green:r%2?'#F7FAF7':C.pale;a.text.style={typeface:'PingFang SC',fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0||c===0,autoFit:'none'};}}
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:widths.length}).assign({margins:{left:12,right:10,top:8,bottom:7},anchor:'center'});
 tableOwners.push(p.slides.items.length);return t;
}
function foot(s,value,y=630){tx(s,value,82,y,1090,42,17,C.muted);}

let s=page('材料基础模型：已有先例与本项目问题','调研范围：公开论文与官方代码，截至 2026-09-14',['nature','llamat','alm','multi'],
'研究问题：是否已有模型把组成、性质、文本和晶体信息用于一个可复用的学习系统，并利用不完整样本支持预测与生成？本次为针对性原始来源检索，不是穷尽性系统综述。“未证实”只代表所读材料的证据范围。');
tx(s,'已有材料 / 科学基础模型，且与老师设想有直接交集',82,215,1110,55,32,C.ink,true);
tx(s,'NatureLM：科学实体统一成序列，混合纯文本与配对数据\nLLaMat：材料文献与 CIF 参与训练，再适配多种任务\nALM：结构与文本混合任务，语言模型连接晶体扩散生成器',82,298,1110,162,28);
tx(s,'值得验证的具体问题',82,492,1110,45,27,C.green,true);
tx(s,'不完整记录如何产生可靠监督？同一模型如何切换输入 / 输出字段？\n边轨道表示是否改善结构补全、条件生成或资源消耗？',82,544,1110,88,27);
foot(s,'不能把“多条件 + text space + 基础大模型”本身作为尚无人做过的创新点。',658);

s=page('最相关先例：统一程度与缺失数据证据','“统一表示”“统一参数”“不完整样本训练”需要分别判断',['nature','llamat','alm','matter','multi','bind']);
table(s,[['工作','统一了什么','不完整数据的实际证据','与本项目的边界'],
 ['NatureLM (2025)','文本 + 多类科学序列\n材料组成、空间群、性质','纯文本 / 单域 / 跨域混训\n无需每条都有晶体坐标','完整晶体由 Mat3D 专门适配\n未证明任意字段补全'],
 ['LLaMat (2024/26)','材料文献 + CIF\n领域预训练、任务适配','文献不要求配对 CIF\n结构预训练不要求性质标签','Chat / CIF 为适配分支\n并非单一任意任务 checkpoint'],
 ['ALM (2026)','3D soft tokens + 文本\nLLM + diffusion decoder','理解阶段混合结构任务\n与纯文本任务','Core / Edit / Gen 分阶段\n并非纯离散文本 DLM'],
 ['MatterChat (2026)','结构编码器 + LLM\n描述、分类、数值预测','研究数据均有晶体结构\n未证明无结构记录混训','多任务预测\n未展示晶体生成'],
 ['MultiMat (2023/25)','结构、文本、DOS\n与电荷密度的表征对齐','缺 DOS / 电荷密度可训练\n对应缺失损失置零','实验以结构为基础\n主要是预测与检索'],
 ['MatBind (2026)','结构、pXRD、DOS\n文本的表征对齐','DOS / 文本可缺\n结构和 pXRD 全部存在','结构为对齐锚点\n检索不等于生成']
],[200,285,335,300],{height:420,size:20,top:198,rows:[42,63,63,63,63,63,63]});
foot(s,'日期写预印本首发 / 已核实期刊年份。ALM、MatBind 按所读预印本报告，逐篇来源见备注。');

s=page('NatureLM：与“都放进 text space”直接相关','1B / 8B / 46.7B 参数模型；单域与跨域数据混训，材料性质也写成文本',['nature'],
'§2.1 directly supports single-domain and cross-domain training and property-to-text data. §5.4 states that full-coordinate modelling is a separately fine-tuned NatureLM-Mat3D module. The example string below is our own illustrative Li/O sequence under the published syntax, not a copied model output or a newly generated crystal.');
table(s,[['数据 / 阶段','论文实际做法'],['统一科学序列','文本、分子、蛋白、DNA/RNA、材料共享序列建模框架'],['材料基本表示','元素按晶胞计数展开，再接空间群 token'],['建立跨模态联系','材料带隙、凸包上方能量等属性转换为文本，与材料序列配对'],['完整 3D 晶体','Mat3D 另行微调，输入组成 / 空间群，输出晶格和分数坐标']],[250,870],{height:300,top:198,size:24,rows:[42,64,64,66,64]});
tx(s,'形式示例（按论文语法改写）',82,520,1080,35,23,C.green,true);
tx(s,'<material> Li Li Li Li O O O O O O O O <sg225> </material>',82,565,1100,40,25);
foot(s,'意义：混合科学 / 文本语料已有直接先例。边界：主材料序列本身不含完整几何，3D 能力另行适配。');

s=page('ALM：无结构文本也参与同一理解模型训练','结构编码器提供 soft tokens，Qwen3-8B 处理语言，MatterGen 解码晶体',['alm']);
table(s,[['训练桶','输入与目标','采样比例'],['describe','结构 + 问题，输出结构描述 / 应用叙述','40%'],['property_apps','结构 + 性质问题，输出数值 / 文本','40%'],['arxiv','标题与类别，输出摘要','14%'],['camel','化学 / 物理纯文本问答','4%'],['mascqa','纯文本科学选择题','2%']],[260,675,185],{height:341,top:198,size:23,rows:[41,60,60,60,60,60]});
tx(s,'这是“结构任务 + 纯文本任务混训”的明确证据。',82,561,1100,42,28,C.ink,true);
foot(s,'比例来自理解阶段 Appendix B.1 / Table 15。生成另有训练阶段与 Edit / Gen 变体；不代表任意缺失字段都已解决。',617);

s=page('其他直接相关工作','这些先例分别覆盖性质条件、对称表示或文本引导生成',['atom','pi','cloud','cham','gruver']);
table(s,[['工作','已覆盖能力','不能从中推出的结论'],['AtomGPT (2024)','文本化结构做性质预测\n性质 / 组成条件生成结构','正向 GPT-2 回归器、逆向 Mistral 分别训练\n不能写成一个通用多任务 checkpoint'],['CrystaLLM-π (2025)','无性质标签 CIF 预训练\n再用性质 / XRD 条件生成','连续条件编码进入注意力\n并非所有条件都做普通文本 token'],['CLOUD / SCOPE (2026)','空间群生成元 + WP + 组成\n无标签 masked 预训练，再预测性质','对称字符串 + 基础表征已有先例\n未展示任意文本条件晶体生成'],['Chameleon (2025)','Crystal CLIP 对齐文本 / 结构\n文本条件扩散生成晶体','文本引导 diffusion 已有先例\n不等于通用预测 / 生成统一训练']],[245,400,475],{height:365,top:200,size:23,rows:[45,80,80,80,80]});
foot(s,'更早的 Gruver et al. (2024) 已展示 LLM 晶体文本生成、部分结构补全和文本条件生成。');

s=page('不完整数据如何参与训练','拟议训练组织：一条记录只监督实际观测到的内容',['nature','alm','multi'],
'The training design on this slide is a proposal for CrystalDLM, not an existing implementation or a claim copied from a paper. Let O be the observed-field set and M a randomly masked subset of O. Reconstruction loss applies only to M. Entirely absent fields provide no ground truth. Paired observations are necessary to learn identifiable cross-field relationships.');
table(s,[['手头有什么','可构造的训练任务','没有的字段怎么处理'],['组成 + 性质，没有结构','组成预测已知性质\n或以性质为条件补全组成','不计算晶体结构监督'],['结构，没有性质标签','结构 token 自监督重建\n或组成 / 部分位点条件补全','不把未知能量写成 0'],['组成 + 结构 + 部分性质','随机选择已知字段作条件\n其余已知字段作预测目标','只在有真值的目标上算损失'],['只有材料文本','文本 span 重建 / 问答\n保留材料语言知识','需要配对数据建立文本与结构联系']],[325,415,380],{height:326,top:200,size:23,rows:[42,71,71,71,71]});
tx(s,'观测缺失 ≠ 训练时遮盖',82,554,1090,38,27,C.green,true);
tx(s,'[ABSENT]：数据库没提供，无真值监督    [MASK]：已知值被遮住，有真值可学',82,602,1100,40,23);
foot(s,'能把数据编码成文本，只解决输入形式；单位、测量条件、材料身份和跨字段配对仍决定能否学到有效关系。',655);

s=page('可检验的研究差异','“多任务晶体基础模型”是目标，收益需要对照实验建立',['nature','alm','cloud','local','prior'],
'Everything in the experiment matrix is proposed future work. Same data splits and matched optimization/token budgets are essential. Hold out material identities/prototypes across sources to prevent duplicated formulas, polymorphs, derived texts and structures leaking between train and test.');
table(s,[['待检验的问题','最小对照','关键评价'],['不完整记录是否有用','仅完整样本 vs 加入不完整样本\n控制训练预算与数据重复','性质 MAE、结构补全、低标签迁移'],['边轨道是否有用','CIF / WP-only / 边轨道\n保持模型、训练集与条件一致','合法性、目标命中率、显存与吞吐'],['非自回归是否有用','相同表示的 AR vs masked / DLM','有效结构 / 秒、采样步数、生成质量'],['共享模型是否可复用','单任务模型 vs 共享模型\n留出条件组合 / 性质 / 原型','负迁移、组合泛化、少样本适配']],[310,455,355],{height:326,top:198,size:23,rows:[42,71,71,71,71]});
tx(s,'研究主张应具体到：数据利用机制 + 表示 + 学习目标带来的收益。',82,560,1100,72,27,C.ink,true);
foot(s,'目前边轨道候选已做可逆编码与单级条件几何去噪；离散生成、多步反向采样和新材料稳定性仍待验证。',647);

s=page('Wyckoff 4a：以 NaCl 为例','空间群 #225，Fm−3m，标准立方常规晶胞，分数坐标取模 1',['iucr','wp'],
'NaCl rock-salt example, full occupancy. Wyckoff multiplicity is the number of sites of one orbit per conventional cell. It is not a coordination number, bond count or number of independent coordinates. Na and Cl can be interchanged with an origin relabelling; the shown assignment is one standard choice. Conventional fcc cell contains 4 Na and 4 Cl, primitive cell contains one formula unit.');
tx(s,'4',92,202,150,118,94,C.green,true);tx(s,'每个常规晶胞内\n有 4 个对称等价位置',258,222,370,88,27);
tx(s,'a',670,202,120,118,94,C.green,true);tx(s,'该空间群 WP 表里的字母标签\n不是元素，也不是全局位置编号',820,222,385,88,25);
table(s,[['占位','代表点','展开后的全部位置'],['Na：4a','(0, 0, 0)','(0,0,0)   (0,½,½)   (½,0,½)   (½,½,0)'],['Cl：4b','(½, ½, ½)','(½,½,½)   (½,0,0)   (0,½,0)   (0,0,½)']],[230,255,635],{height:184,top:350,size:25,rows:[44,70,70]});
tx(s,'两个独立位点轨道，展开后是 8 个原子 = 4 个 NaCl。',82,568,1090,45,28,C.ink,true);
foot(s,'4a 与 4b 的 site symmetry 都是 m−3m，自由坐标数为 0。4 表示位置重数，和 Na 的 6 配位不同。',633);

s=page('Wyckoff 位置也可以带自由参数','字母确定允许的位置族，连续参数决定具体落点',['iucr','wp','local']);
tx(s,'#62 Pnma，4c：两个自由参数 x、z',82,203,1100,43,29,C.green,true);
table(s,[['代表点','同一轨道的另外 3 个位置'],['(x, ¼, z)','(½−x, ¾, ½+z)    (−x, ¾, −z)    (½+x, ¼, ½−z)'],['取 x=0.10, z=0.20','(0.40,0.75,0.70)   (0.90,0.75,0.80)   (0.60,0.25,0.30)']],[330,790],{height:182,top:265,size:23,rows:[42,70,70]});
tx(s,'#139 I4/mmm，KrF₂ 中的 F 占据 4e',82,482,1100,42,29,C.green,true);
tx(s,'标准代表 (0,0,z) 只有 1 个自由参数。实际 mp-30009：z = 0.290407。\n本地编码选择等价代表 (½,½,0.790407)，属于同一个 4e 轨道。',82,535,1100,92,25);
foot(s,'Pnma 的数值仅用于展开演示，不宣称某个真实材料。KrF₂ 数值来自已保存结构；4c 的 site symmetry 为 .m.。',655);

s=page('不同空间群里的“4a”并不相同','都写 4a，只保证本表中重数为 4；局部对称与可动范围可以完全不同',['iucr','wp']);
table(s,[['空间群 / 设置','WP','代表坐标','site symmetry','自由参数数'],['#225  Fm−3m','4a','(0,0,0)','m−3m','0'],['#216  F−43m','4a','(0,0,0)','−43m','0'],['#62  Pnma','4a','(0,0,0)','−1','0'],['#19  P2₁2₁2₁','4a','(x,y,z)','1','3']],[335,105,280,240,160],{height:305,top:198,size:25,rows:[49,64,64,64,64]});
tx(s,'#225 与 #216 的 4a 甚至具有相同的 4 个坐标，\n但整个晶体允许的空间群操作不同，位点对称也不同。',82,532,1100,89,27);
foot(s,'另一个同标签例子：#216 的 4c 固定在 (¼,¼,¼)；#62 的 4c 是 (x,¼,z)，有两个自由参数。',653);

s=page('跨空间群复用：共享特征，保留身份','site symmetry 可以共享表征，完整 WP 身份仍需要空间群与设置',['iucr','wp','wyformer'],
'The factorized record at the bottom is a proposed representation interface. It is not the actual edge-orbit codec JSON and not a globally unique WyFormer enumeration. Enumeration is meaningful only with its space-group-specific compatibility map.');
table(s,[['例子','可以复用的信息','必须继续区分的信息'],['#225：4a 与 4b','同为 m−3m\n重数 4，自由度 0','代表点 (0,0,0) 与 (½,½,½)\n同一空间群内也不是同一 WP'],['#221：1a 与 #225：4a','原点处均为 m−3m\n自由度均为 0','P 与 F 晶格不同，重数 1 与 4\n全局操作表不同'],['#216：4a 与 #225：4a','相同代表点与 F-格点坐标','位点对称 −43m 与 m−3m\n不能因字符串相同而合并身份']],[320,350,450],{height:280,top:204,size:24,rows:[46,78,78,78]});
tx(s,'建议接口：',82,524,1080,40,26,C.green,true);
tx(s,'(元素，共享对称特征，自由度) + (空间群 / Hall 设置，WP 身份)',82,574,1100,45,26);
foot(s,'共享的是参数和特征。解码时仍需 compatibility map，恢复合法重数、允许坐标与具体对称操作。',637);

s=page('边轨道 tokenizer：四类组件','把结构拆成离散 prototype 与连续 geometry，连接本身也做对称合并',['local','prior']);
table(s,[['组件','当前实际存什么','用途'],['全局操作表','space_group、hall_number\nSeitz 操作 (R, t)','明确坐标设置\n按操作编号找到旋转 / 分数平移'],['位点轨道','元素 Z、WP 字母\n仿射稳定子 (g, image)、dof','确定哪些坐标可以变化\n代表位点展开成全部等价原子'],['边轨道','[i, j, g, nₓ, nᵧ, n_z]','说明连接哪个位点的哪个对称像\n以及哪个周期晶胞'],['独立几何载荷','晶格矩阵 lattice\n各位点 free_vectors','给出晶胞形状 / 大小和自由坐标\n与 prototype 合起来恢复实际结构']],[225,440,455],{height:354,top:200,size:24,rows:[46,77,77,77,77]});
tx(s,'结构 = 全局规则 + 位点轨道 + 边轨道 + 几何载荷',82,584,1100,48,28,C.ink,true);
foot(s,'稳定子显式记录 WP 对应约束以便解码；固定设置与代表点后可查表导出。edge record 尚非训练好的词表 token。',652);

s=page('真实编码示例：KrF₂，mp-30009','空间群 #139，Hall #424；以下保留真实字段和值，长列表作省略',['local'],
'Exact JSON field names are shown. operations contains 32 Seitz entries; translations are stored as rational [numerator, denominator] pairs. nodes[0] has 16 affine stabilizer entries, nodes[1] has 8. The ellipses are for display only. JSON schema is not changed by this task. The projector formula q=b+Pu gives (0,0,0) for Kr and (1/2,1/2,z) for F in this chosen representative gauge.');
tx(s,'离散 prototype',82,201,550,44,28,C.green,true);
tx(s,'space_group: 139\nhall_number: 424\noperations: [32 个操作，平移用分数]\nnodes:\n  0: Z=36, wyckoff_letter="a", dof=0\n     stabilizer=[16 条 (operation,image)]\n  1: Z=9,  wyckoff_letter="e", dof=1\n     stabilizer=[8 条 (operation,image)]\nedge_tokens: [\n  [0,1,16,-1,-1,-1], … 共 7 条\n]',82,261,575,329,22);
tx(s,'连续 geometry',694,201,506,44,28,C.green,true);
tx(s,'lattice:\n  [4.383300, 0, 0]\n  [0, 4.383300, 0]\n  [0, 0, 6.668202]  Å\n\nfree_vectors:\n  [0, 0, 0]\n  [0, 0, 0.790407]',694,261,506,268,24);
tx(s,'代表点 q = b + P u\nKr: (0,0,0)    F: (½,½,0.790407)',694,552,506,75,23);
foot(s,'prototype 的分数用整数分子 / 分母保存，没有自由坐标或晶格浮点数。free_vectors 用三维容器存储有效自由度。',657);

s=page('一条边的地址如何解码','实际记录 [0, 1, 16, −1, −1, −1]：Kr 代表点连接到一个 F 对称像',['local']);
table(s,[['字段','实际含义'],['i = 0','源位点 Kr，q₀ = (0,0,0)'],['j = 1','目标位点 F，q₁ = (½,½,0.790407)'],['g = 16','当前操作表第 16 项（从 0 编号）：R = I，t = (½,½,½)'],['n = (−1,−1,−1)','目标像所在晶胞的整数平移，单位为晶格基矢']],[320,800],{height:253,top:192,size:24,rows:[41,53,53,53,53]});
tx(s,'端点 = R₁₆ q₁ + t₁₆ + n = (0, 0, 0.290407)',82,477,1120,48,31,C.ink,true);
tx(s,'距离 = 0.290407 × 6.668202 Å = 1.93649 Å',82,536,1100,43,28);
foot(s,'若只把 g 改成 0，端点变成 (−½,−½,−0.209593)，距离约 3.40000 Å。WP 相同，邻居地址不同。',602);
foot(s,'g=16 是本份 operations 列表的局部编号，不是跨空间群通用 token。n 是整数晶胞编号，不是连续位移。',649);

s=page('7 条边轨道记录如何恢复 44 条周期边','KrF₂ 的 4.5 Å 周期近邻图；这是实际编解码结果，不是生成模型的采样结果',['local'],
'Each row was isolated and expanded using expand(prototype, free_vectors). Orbit sizes sum to 44 and reproduce the union. edge_key and transform canonicalize undirected endpoint pairs, so the counts correspond to 44 UNDIRECTED periodic edges. Explicit reverse expansion yields 88 directed adjacency entries. The 7/44 count alone is not whole-file compression: the operations table and site constraints also occupy storage. Changing the free geometry preserves the abstract graph for generic coordinates, but it need not remain the same cutoff-neighbour graph; singular coordinates may also change orbit multiplicities.');
table(s,[['源 / 目标轨道','[i, j, g, nₓ, nᵧ, n_z]','恢复的无向周期边数'],['Kr / Kr','[0, 0, 0, −1, 0, 0]','4'],['Kr / F','[0, 1, 0, −1, −1, −1]','16'],['Kr / F','[0, 1, 16, −1, −1, −1]','4'],['F / F','[1, 1, 0, −1, 0, 0]','8'],['F / F','[1, 1, 1, 1, 1, 1]','2'],['F / F','[1, 1, 1, 1, 1, 2]','2'],['F / F','[1, 1, 17, 0, 0, 1]','8']],[300,515,305],{height:352,top:194,size:23,rows:[44,44,44,44,44,44,44,44]});
tx(s,'编码：对称等价边归为一组，保留代表地址。\n解码：对代表边施加群操作，去重得到 44 条无向边。',82,569,1100,77,27);
foot(s,'若为消息传递显式加反向边，则为 88 条有向邻接。44/7 不是总 token 压缩比；近邻边不全是化学键。',662);

s=page('接入多任务 text space 的形式','下列标签是拟议接口，不是当前 tokenizer 已实现的语言模型词表',['local','nature','alm'],
'Proposed record serialization only. The existing prototype/geometry payload is real, but task, observed, condition and target wrappers are not implemented. Numeric conditions must record property name, unit, reference state and measurement/computational conditions. Energy alone is ambiguous. No property value is fabricated for mp-30009 here.');
tx(s,'<task> structure_completion </task>\n<observed> composition, symmetry, partial_structure </observed>\n<composition> Kr:1 F:2 </composition>\n<symmetry> space_group=139 hall_number=424 </symmetry>\n<prototype> nodes=[…] edge_tokens=[…] </prototype>\n<geometry> lattice=[…] free_vectors=[MASK] </geometry>\n<property name="formation_energy_per_atom" unit="eV/atom">\n  [ABSENT]\n</property>',82,211,1110,300,25);
tx(s,'切换任务就是改变“哪些字段可见，哪些字段需要预测”。',82,535,1100,48,28,C.ink,true);
tx(s,'只有性质 / 文本的记录不附加伪造结构；完整结构记录可做结构补全。\n若需全离散 DLM，连续几何的量化或独立解码仍需选择并验证。',82,590,1100,72,24);

s=page('主要文献入口','新增页的结论按论文方法与实验范围整理，具体定位见各页备注',['nature','llamat','alm','matter','multi','bind']);
const entries=[
 ['NatureLM (2025)','arXiv:2502.07527v2','统一科学序列；§2.1、§5、§8.3'],
 ['LLaMat (2026)','doi:10.1038/s42256-026-01199-8','材料文献 + CIF；预训练 / 适配分支'],
 ['ALM (2026，预印本)','arXiv:2606.21395v1','§2；Appendix B.1 / B.2 的训练混合'],
 ['MatterChat (2026)','doi:10.1038/s42256-026-01214-y','多任务结构理解与性质预测'],
 ['MultiMat (2023/2025)','arXiv:2312.00111v3','§IV.2：缺失模态的损失处理'],
 ['MatBind (2026，预印本)','arXiv:2607.08470v1','§5：结构锚定的多模态对齐']];
table(s,[['论文','可检索标识','本次核对重点'],...entries],[330,420,370],{top:197,height:397,size:21,rows:[43,59,59,59,59,59,59]});
foot(s,'检索包括材料基础模型、异构 / 缺失模态、统一文本序列、条件生成与预测。AI 辅助检索与整理，未做穷尽性排除。',639);

s=page('补充文献与编码证据','论文原始来源 + 晶体学定义 + 可复算的本地实例',['atom','pi','cloud','cham','gruver','iucr','wp','local','prior']);
table(s,[['内容','入口'],['AtomGPT / CrystaLLM-π','arXiv:2405.03680；arXiv:2511.21299'],['CLOUD / Chameleon','doi:10.1038/s41467-026-70467-3\ndoi:10.1038/s41467-025-59636-y'],['Wyckoff 定义与坐标表','IUCr：dictionary.iucr.org/Wyckoff_position\nPyXtal：pyxtal/database/wyckoff_list.csv'],['KrF₂ 编解码实例','work/edge_orbit_release/examples/mp-30009.*.json\nwork/edge_orbit_release/codec/discrete_graph_prototype.py'],['候选方法与现有验证','research/edge_orbit_tokenizer_proposal.md\nexperiments/wyckoff_tokenizer_search/ASU_CONTROL_RESULTS.md']],[320,800],{top:202,height:358,size:23,rows:[44,53,67,64,65,65]});
tx(s,'边轨道的生成能力仍需通过完整采样与材料性质验证。',82,592,1100,45,27,C.ink,true);
foot(s,'ASU、Wyckoff 约束、周期平移图都有先例；候选贡献需限定到具体编码与任务收益。',652);

await fs.writeFile(dir+'manifest.json',JSON.stringify({original:15,total:p.slides.items.length,slides:manifest,tableOwners},null,2));
await fs.writeFile(dir+'references.json',JSON.stringify(refs,null,2));
await (await PresentationFile.exportPptx(p)).save(dir+'authored.pptx');
for(let i=15;i<p.slides.items.length;i++){const b=await p.slides.getItem(i).export({format:'png',scale:1});await fs.writeFile(dir+`draft-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log(JSON.stringify({slides:p.slides.items.length,added:manifest.length,tableOwners}));
