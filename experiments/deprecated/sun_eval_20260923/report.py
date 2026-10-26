"""Render the completed pilot counts; no ML or structure loading."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
r = json.loads((root / 'RESULTS.json').read_text())
runtime = json.loads((root / 'runtime.json').read_text())
lines = ['# 组成条件 S.U.N. 近似评估', '',
         '两种表示沿用完整 MP-20、500 轮训练后的现有各 512 次测试生成。没有重新训练或挑选新的生成结果。', '',
         '**结果属于 CHGNet 势模型估计，不是 DFT 验证，也不直接等同于论文的无条件 S.U.N. 分数。**', '',
         '| 指标（各模型总生成次数 512） | 显式位点约束 | Wyckoff 对照 |',
         '|---|---:|---:|']


def row(label, values):
    lines.append('| '+label+' | '+' | '.join(map(str, values))+' |')


reps = ['explicit', 'wyckoff']
for k, label in [('evaluated', '弛豫前结构与组成检查通过'), ('unique', '弛豫前去重后数量'),
                 ('novel_train', '弛豫前不匹配训练集的数量')]:
    row(label, [r['unrelaxed'][p][k] for p in reps])
for k, label in [('evaluated', '弛豫及匹配评估成功'), ('converged', '达到弛豫收敛标准'),
                 ('unique', '弛豫后去重后数量'), ('novel_train', '弛豫后不匹配训练集的数量')]:
    row(label, [r['relaxed_converged'][p][k] for p in reps])
for t in ['0.0', '0.08', '0.1']:
    row(f'能量差 < {t} eV/atom，收敛且基础有效',
        [r['relaxed_converged'][p]['thresholds'][t]['stable'] for p in reps])
    row(f'同时稳定/近稳、唯一、新颖（阈值 {t}）',
        [f"{r['relaxed_converged'][p]['thresholds'][t]['stable_unique_novel']}/512 "
         f"({r['relaxed_converged'][p]['thresholds'][t]['sun_percent_of_all_attempts']:.2f}%)" for p in reps])
lines += ['', 'U 和 N 的单项数量可重叠；S.U.N. 是筛选后去重的交集计数，并不是三列比例相乘。', '',
          '## 固定口径与解释边界', '',
          '- S：CHGNet 0.3.0（软件包 0.3.8）预测的单位原子能量减 MP 2023-02-07 hull 能量。FIRE 可变晶胞弛豫，最多 500 步，fmax=0.1 eV/Å；每例 120 秒上限。主表要求收敛；含未收敛终点的诊断另存 JSON。',
          '- U：每个模型自身的候选集合内按结构匹配去重。不能按化学式去重，同一组成可能存在不同晶型。',
          '- N：不匹配本次实验使用的 27,136 个训练集结构。train+val 参考库的敏感性结果保留在 JSON。这不表示该材料从未出现在整个 MP 数据库或文献中。',
          '- 匹配使用 StructureMatcher：ltol=0.3、stol=0.5、angle_tol=10°、scale=True、primitive_cell=True、attempt_supercell=False；该 discovery 匹配口径比之前参考重建指标更宽松。',
          '- 所有失败保留在总分母 512；使用之前的预测晶格对称化和结构/组成/对称检查，因而属于带这些额外门槛的 conditional SUN。无对称约束的弛豫后空间群可能变化，逐例记录 initial_sg/final_sg。',
          '- 本轮每个条件只有一次生成，不能证明同一组成下的多样性。只有一个训练种子，不作跨种子优越性结论。',
          '- CHGNet 与 MP20 存在训练数据重叠，而且此处将 ML 能量与 DFT 参考 hull 比较；能量模型误差和参考库/校正口径会影响结果。不能据此证明热力学稳定或可合成。', '',
          '## 计算记录', '',
          f"- Slurm job {runtime['job_id']}，节点 {runtime['host']}，GPU {runtime['gpu']}。",
          f"- 评估脚本耗时 {r['elapsed_seconds']/60:.1f} 分钟；不包含下载、排队和安装。",
          f"- 逐例计算错误：explicit={len(r['errors']['explicit'])}，wyckoff={len(r['errors']['wyckoff'])}。",
          '- RESULTS.json：原始、收敛后及所有终点的汇总；*_relaxed.jsonl：每例能量、力、匹配、收敛、对称群与错误；relaxed/：弛豫后的 CIF。', '',
          '来源和预先固定的规则见 [PROTOCOL.md](PROTOCOL.md)。']
(root / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:27]))
