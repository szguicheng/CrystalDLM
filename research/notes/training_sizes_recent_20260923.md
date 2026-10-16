# DynaCrys / MaskGXT 實際訓練樣本數核對（2026-09-23）

## 結論先行

- **最可比的 MP-20 主訓練量：兩篇均約 27.1k 個獨立 train 結構。** DynaCrys 的正文說明以 MP-20 training partition 訓練；MaskGXT 說明 MP-20 用 60/20/20 切分。兩文原文都沒有在方法段落直接列 MP-20 train 的整數數目。按常用 MP-20 benchmark split，train 是 **27,136**（validation 9,047、test 9,046；合計 45,229）。故應將 27,136 標為由其聲明採用的 MP-20 split 對應出的數量，而非聲稱兩篇論文逐字報告了這個整數。
- **MaskGXT 的 MPTS-52 是另一套 benchmark 的獨立訓練設定，但實際 train 數未確認。** MaskGXT 原文稱採 60/20/20，未列每個子集數字；常見 MPTS-52 chronological split 的 27,380 只能作標準 split 參考，不能在確認 MaskGXT 實際採用的 split 前當作其 actual train count。論文列的 MPTS-52 batch size 是 128；不應把 MP-20 和 MPTS-52 數量相加成單次訓練樣本數。
- **MaskGXT MP-20 polymorph split 是額外評估切分，不是第三個擴大的訓練集。** 原文將它明確引入為 polymorph-aware evaluation；其主模型仍是 MP-20 訓練模型。
- **沒有找到 DynaCrys 或 MaskGXT 用 Alexandria / Alex-MP-20 數十萬晶體作為這兩篇報告模型的訓練集證據。** MaskGXT 全文搜尋無 Alexandria；DynaCrys 原文只說 MP-20 training partition 用於訓練，且其 shared symmetry codebook 從 operation-derived crystallographic descriptors 預訓練，這不等於在額外晶體庫上的樣本預訓練。故不能把其他研究中的 675,204 Alex-MP-20 數字移植到這兩篇。
- **對照目前的 6,000 個獨立 training examples：** 約為 MP-20 基準 27,136 的 **22.1%**，即少 **21,136**，基準約為 6,000 的 **4.52 倍**。所以參考這兩篇的 MP-20 實驗，建議把「同一量級」理解為約 2.7 萬訓練結構；6,000 是清楚較小的一檔。這是訓練樣本量比較，不表示兩方法任務或效能可直接比較。

## 分論文與設定

| Paper / run | 用於訓練的獨立結構數 | 如何解讀 | 可順帶記錄的訓練設定 |
|---|---:|---|---|
| DynaCrys canonical，MP-20 | 約 27,136（標準 split 對應值；paper 未明列整數） | MP-20 training partition 訓練 symbolic generator 與 geometry model；不是 45,229 全集，也不是 10,000 generation candidates | Symbolic denoiser batch 256；geometry batch 128，60,000 training steps；沒有報 epoch 數。另有 codebook 每階段 3,000 steps，該部分以 operation-derived crystallographic descriptors 訓練，不能按晶體數計入 MP-20 train。 |
| MaskGXT MP-20 | 27,136（標準 MP-20 split train 數；paper 本身只明說 60/20/20） | 主基準 / development 設定；MP-20 全 benchmark 約 45.2k，不是 45.2k train | 約 248M 參數；batch 512；最多 3,000 epochs，validation METRe early stopping。 |
| MaskGXT MPTS-52 | **實際數未確認**；27,380 僅為常見 chronological split 的 train 數參考 | 獨立的更高原子數 benchmark run；原始 MPTS-52 共約 40,476 structures，不是 train 數。MaskGXT 原文同時稱採 60/20/20，與常見 chronological split 數量不能直接視為同一划分 | batch 128；架構仍約 248M；最多 3,000 epochs + early stopping。 |
| MaskGXT MP-20 polymorph evaluation split | 不另報獨立的訓練樣本數 | 這是把同一 composition 的 polymorphs 收在同一 subset 的評估 protocol；不能當成額外一批 train data 加總 | 用於 polymorph coverage / METRe 評估。 |

### 不能混為一談的數字

1. **MP-20 全集 vs train split。** 常用處理後 benchmark 有 45,229 structures，拆為 27,136 / 9,047 / 9,046；45,229 是全量，27,136 才是常用 train split。文獻有時將 raw MP-20 約稱 45,231；不同打包版本總量差兩筆，這不改變文中採用的標準 train count 27,136。
2. **MPTS-52 全量 vs train split。** 全集約 40,476，常見 chronological split 參考為 27,380 / 5,000 / 8,096。MaskGXT 原文稱兩資料集採 60/20/20，但沒有報其 MPTS-52 子集整數；這與常見 chronological split 的子集數不能未經確認便等同。因此 MaskGXT MPTS-52 實際訓練數待核對，不要對 40,476 直接乘 60% 推算 train 數，也不要把 27,380 寫成已確認的 MaskGXT 實際數。
3. **10,000 generation candidates。** DynaCrys 在評估階段每個方法約生成 10,000 個候選，屬於推論與評估 budget，不是訓練資料。MaskGXT 按 test reference 一對一生成，亦另有每 composition 固定 S=2 等 polymorph 評估預算，這些也不是訓練樣本數。
4. **Alexandria 大規模預訓練。** Alex-MP-20 約 675,204 是其他工作/OMatG 資料目錄提供的合併資料規模，並非這兩篇所報 DynaCrys / MaskGXT run 的 train count。MaskGXT 的「pretraining」概念亦沒有出現在其全文的資料訓練描述中。

## 可引用的證據段落

### DynaCrys（arXiv:2608.07401v1）

- 本地版本鎖定：`research/source_manifest.json` 將 `dynacrys-2608.07401` 指向 `dynacrys-2608.07401.pdf/.txt`；manifest SHA-256 PDF=`76160a638f415f7d6bc6a29f13b733e05b2e740339d60c0c1a86bc0c8849c70f`，文本版本標記 `arXiv:2608.07401v1`。PDF 共 26 頁。
- 正文 §5.1（PDF p.7 / extracted text 約 lines 380–383）說明 DynaCrys 以 MP-20 的 train partition 訓練，並以該 split 作 novelty reference；每個評估集合約生成 10,000 個候選。這證明 10k 是候選評估量，而非訓練資料數。
- Appendix C.2（PDF pp.15–16，extracted text lines 892–917）：共享 symmetry codebook 的兩階段預訓練先擬合 operation-derived crystallographic labels，之後訓練 row embeddings；每階段 3,000 steps。這是晶體學描述符任務，不是 Alexandria 晶體預訓練。
- Appendix D.4（PDF p.18，lines 1045–1064）：symbolic denoiser architecture/optimization；batch 256，另有 2,000-step warmup（注意此數是 warmup，不是總訓練 steps）。
- Appendix F.4（PDF p.21，lines 1255–1268）：geometry model 對 training structures 的 conventional-cell lattice 與 orbit anchors 訓練；batch 128、500 warmup、60,000 training-step budget。
- Appendix G.1（PDF p.22，lines 1274–1277）：再次明確 training partition 是 novelty reference、test partition 是 coverage reference；每方法 generation budget 約 10,000/run。
- 官方全文：[arXiv HTML v1](https://arxiv.org/html/2608.07401v1)（§5.1；Appendix C.2、D.4、F.4、G.1）。

### MaskGXT（arXiv:2606.22866v1）

- 本地版本鎖定：`research/source_manifest.json` 將真 MaskGXT 指向 `maskgxt-2606.22866.pdf/.txt`，PDF SHA-256=`b1164acf10c58c3d232dc78f5a35f4abafe2aaa21eeeaa8de6d87ae232a3c7e7`，文本版本標記 `arXiv:2606.22866v1`，PDF 共 20 頁。名為 `maskgxt.pdf` 的 manifest 項實際對應 WyFormer `2503.02407v4`，不能拿來當 MaskGXT 原文。
- §5.1 Datasets（PDF p.8 / extracted text lines 522–534）：MP-20 和 MPTS-52；MPTS-52 最多 52 atoms/unit cell、按 publication year chronological split；兩 dataset 均採標準 60/20/20 train/validation/test。接著另述 MP-20 polymorph split：按 composition 將所有 polymorphs 放同一 subset，目的是 polymorph coverage 評估。
- §5.3（PDF pp.8–9 / lines 573–579）：模型用 test composition 作條件；報告模型約 248M 參數、最多 3,000 epochs、validation METRe early stopping。
- Appendix B / Table 6（PDF p.17 / lines 871–892）：batch size 512（MPTS-52 為 128）、最多 3,000 epochs + early stopping。這些是優化設定，不改變獨立結構數。
- 全文對 `Alexandria` 搜尋無命中；因此其 248M 模型不能被解讀為在 Alexandria 675k 晶體上預訓練。
- 官方全文：[arXiv HTML v1](https://arxiv.org/html/2606.22866v1)（§5.1–5.3；Appendix B）。官方程式庫 README 也區分 `mp_20`、`mp_20_ps`（polymorph split）和 `mpts_52` 三個 dataset key：[MaskGXT 官方 repository](https://github.com/kiyoung98/MaskGXT)。

### Split 數字的交叉核對

- MP-20 可用 split 包明列 27,136 train / 9,047 validation / 9,046 test，共 45,229；每 CSV row 一個 crystal structure：[OneScience MP-20 dataset card](https://huggingface.co/datasets/OneScience-Group/mp20)。這是對標準 60/20/20 split 的明數核對來源，不是兩篇論文自身的逐字數量宣告。
- MPTS-52 常見 chronological split 參考為 27,380 train / 5,000 validation / 8,096 test；此數值見採用 MPTS-52 的 benchmark 論文（例如 NeurIPS 2024 FlowMM：[3D Structure Prediction of Atomic Systems with Flow Matching](https://papers.nips.cc/paper_files/paper/2024/file/c6fdc94aeb2cb3a426d510d970045dab-Paper-Conference.pdf)）。但 MaskGXT 原文稱其採 60/20/20，且未報子集整數；其實際 MPTS-52 train 數仍待確認，27,380 只可作標準 chronological split 參考。
- 以 27,136 作為 MP-20 benchmark 比較基線：6,000/27,136 = 22.1%；27,136/6,000 = 4.52。這是純比例換算，不是來源論文中的實驗結果。

## 來源核對邊界

兩篇 arXiv v1 官方 HTML 頁面可連線且已逐段核對；本地 PDF/TXT 版本標記與 manifest checksum 也吻合。MP-20 整數 train count 是透過標準 split 與已發布 split 清單交叉核對得出，並非兩篇論文內明列的數字。MaskGXT MPTS-52 的 60/20/20 說法與常見 chronological split 整數尚未對上，故該 run 的實際 train count 保持待確認；不能用總量直接乘 60% 代替。
