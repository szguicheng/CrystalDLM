# 从给定几何的压缩，走向离散 prototype：下一关

已按本计划完成512个训练结构的检查，见 `DISCRETE_PROTOTYPE_RESULTS.md`。下面保留表示定义与原验证判据。

## 表示拆分

对源ASU坐标 q 和空间群操作 (R,t)，site stabilizer 包含满足 `R q+t-q=n` 为整数向量的操作。记录 `(operation_id, n)`；对应仿射操作 `(R,t-n)` 精确固定 q。这些是既有 site-symmetry / Seitz / Wyckoff 位置概念。

令 `P=mean(R)`、`b=mean(t-n)`，则 `P q+b=q`，位置族可写为 `q=b+P u`。其自由度为rank(P)；P、b由离散stabilizer记录决定，自由参数u不属于prototype。

候选离散prototype包含：G、当前设置的有理Seitz操作表、节点元素/WP标签/site-stabilizer记录，以及六字段edge-orbit token。几何载荷单独包含晶格与自由参数。当前阶段允许冗余3维u，但必须明确有效自由度，不能把冗余向量维数当成新增自由度。

标准化操作的平移应能编码为小分母有理数，需逐条检查误差后存整数分子/分母；不能悄悄把含任意实数的位点信息塞入“离散”prototype。

## 独立于真实自由坐标的解码检查

1. 用原始q，从新prototype展开图，与已验证codec的原子/周期边比较。
2. 删除原始自由参数，独立生成generic u，并用 `q_new=b+P u`。验证轨道重数与stabilizer不意外改变。
3. 验证相同离散prototype、不同generic几何实例展开后，对应同一抽象周期图；这不要求欧氏距离或截断半径邻接保持相同。

为避免晶胞内包装引起假差异，使用固定的空间群coset代表编号，节点坐标取首次选中的 `R_g q+t_g` **不强制wrap**。重复像按“相差整数晶格平移”去重。原始与generic实例应选择同一批coset代表，此时周期边整数标签应一致。

给定边token `(i,j,h,l)`：源端是q_i，目标端是`R_h q_j+t_h+l`。各自相对已选完整节点代表求整数offset，目标offset减源offset为边的cell image；再按所有空间群操作展开无向边。不要假设source ASU坐标恰好等于已选coset节点的unwrapped坐标，必须扣除source offset。

旧codec使用wrapped节点。若unwrapped节点满足`x_u=x_w+k`，则边标签满足`n_w=n_u+k_target-k_source`；按此gauge变化与旧图核对，不能直接比较未对齐的原子编号/整数image。

## 判据与边界

prototype序列化必须只含整数、字符串和容器；无晶格数值、无真实自由坐标。真实训练结构的旧图往返、generic实例的coset/抽象周期图一致性均通过，才说明它能独立表示“对称prototype+连接关系”。

generic实例的几何可能不合理；此检查不证明物理有效或稳定，也不代表已生成新的材料。若这一步通过，再确定离散边token的条件预测/去噪方式和几何生成接口。
