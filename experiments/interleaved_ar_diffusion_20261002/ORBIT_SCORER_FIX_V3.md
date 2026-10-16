# 坐标审计修复及影响

旧audit仅检查wp.ops[0]生成的固定代表点，错误拒绝某些属于同一解码轨道的等价代表点。真实Ti2O4/SG136例：Ti的2a写为0.500,0.500,0.500，官方decoder与0,0,0生成完全相同的原子集合，代表点距解码Ti为2.2e-16 Å，但旧审计报wyckoff_coordinate_constraint。

修复：使用与PyXtal一致的周期归一化，仅在审计计算中比较该点与所有wp.ops生成的轨道点，取最小周期残差；不改原文或模型预测。容差1e-3、组成/Sites、multiplicity、晶格、重叠及精确SG规则均不变。编辑仍要求原始坐标字段保持不变。

新增测试先复现失败，修复后全部test_checks通过；独立只读审查还对10个空间群344个点验证了残差等于实际atom_site(...,search=False).coords的周期成员距离。错误点0.5,0.5,0.25仍拒绝，修改编辑坐标到等价点仍记失败。

对当前cohort_001、完整recovery_001和order_probe_001的所有已完成输出统一v3重审，现有成功计数全部不变。这是必须修正的审计错误，但不能解释目前混合模型生成差。原文、初版/v2评分、权重均保留；v3另存原输出哈希。新通过G而旧流程未执行下游时，脚本不推断闭环成功。

证据：orbit_rescore_v3_report.json、各输出目录orbit_summary_v3.json/orbit_audits_v3.jsonl；实现score.py和diagnostics/orbit_rescore.py。修复没有重建或改变冻结数据，未参与已完成训练；在601362结束后才同步至远端后续审计。
