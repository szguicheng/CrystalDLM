"""Report only completed fixed-budget training and evaluated test outputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    budget = json.loads((ROOT / 'budget.json').read_text())
    data = json.loads((ROOT / 'data/summary.json').read_text())
    selection = json.loads((ROOT / 'evaluation/selection.json').read_text())
    results = json.loads((ROOT / 'evaluation/test_summary.json').read_text())
    sys.path.insert(0, str(ROOT / 'codec'))
    import program_codec as sp
    import wyckoff_baseline as wp
    from test_program import parse, compare
    export_checks = []
    for result in results:
        rep, mode = result['representation'], result['mode']
        folder = ROOT / 'evaluation/test' / f'{rep}_{mode}'
        rows = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        assert len(rows) == 512 and len({r['id'] for r in rows}) == 512
        for row in rows:
            if not row['projected'].get('basic_and_symmetry_consistent'): continue
            path = folder / f"projected/runs/{rep}_17/samples/{row['id']}.cif"
            reconstructed = (sp if rep == 'explicit' else wp).decode(row['projected_program'])
            error = compare(reconstructed, parse(path))
            export_checks.append({'representation': rep, 'id': row['id'], 'error_A': error})
    (ROOT / 'evaluation/export_checks.json').write_text(json.dumps(export_checks, indent=2))
    lines = ['# 完整 MP-20 固定训练预算结果', '',
             f"已完成显式位点约束 SP1 与 Wyckoff 对照，各 **{budget['epochs']} 轮**，seed 17。每轮为训练集无放回随机遍历一次；按速度选定轮数后没有因收敛情况延长。", '',
             '## 数据与实际预算', '',
             '- 完整 train：27,136；完整 val：9,047；新 test：512（官方 test 按 material_id SHA256 排序的 1024:1536）。没有把验证/测试加入训练，也未丢弃训练材料。',
             f"- 词表 {data['vocab_size']}，共同画布 {data['max_length']}；最大训练序列 {data['maximum_training_length']}。表示、组成条件和基本网络沿用小试验。",
             f"- 实测后选 batch={budget['batch_size']}；每轮 {budget['steps_per_epoch']} 步，每模型 {budget['steps_per_model']:,} 步、{27136 * budget['epochs']:,} 次样本呈现。",
             '- 模型从头初始化；4 层、宽度 192、6 heads、FFN 768；AdamW lr=3e-4，不按 batch 自动放大学习率。',
             '- 每 10 轮评估完整验证集、保存可恢复状态；每 100 轮保留独立 checkpoint。固定轮数跑满，同时保留验证恢复损失最优权重。', '',
             '| 表示 | 完成轮数 | 实际训练时间 | 显存峰值 | 最佳 checkpoint 轮数 |',
             '|---|---:|---:|---:|---:|']
    training = {}
    for rep in ['explicit', 'wyckoff']:
        r = json.loads((ROOT / f'runs/{rep}_17/training_result.json').read_text())
        assert r['state'] == 'complete' and r['epochs'] == budget['epochs']
        training[rep] = r
        lines.append(f"| {rep} | {r['epochs']} | {r['elapsed_seconds']/60:.1f} 分钟 | {r['max_gpu_memory_MB']/1024:.2f} GiB | {r['best_epoch']} |")
    lines.extend(['', '上述时间为各训练循环含验证与 checkpoint，不包含预处理、测速度、排队和生成评估。完整资源收据位于 compute_allocations/。', '',
                  '## 固定测试集生成结果', '',
                  f"在验证集前 512 个材料上比较 one_step、confidence、remask，两种表示共用选中的 **{selection['mode']}**。选择按对称化后通过数、原始基础通过数、原始可解码数排序；冻结权重哈希与采样器后才评估测试。", '',
                  '| 表示 | 可解码 | 组成正确 | 原始基础＋对称一致 | 对称化后基础＋对称一致 | 对称化后参考匹配 |',
                  '|---|---:|---:|---:|---:|---:|'])
    for r in results:
        raw, p, n = r['raw'], r['projected'], r['n']
        lines.append(f"| {r['representation']} | {raw['decoded']}/{n} | {raw['composition_match']}/{n} | {raw['basic_and_symmetry_consistent']}/{n} | {p['basic_and_symmetry_consistent']}/{n} | {p['reference_match']}/{n} |")
    lines.extend(['', '每材料生成一次，不做 best-of-k。条件仅是约分元素比例，不提供原空间群、晶胞原子数、真实序列长度或目标几何。',
                  '基础检查为解码、有限正体积、组成比例一致和无 0.5 Å 内原子对；对称检查为预测操作保持晶格度量及同元素位点集合。晶格对称化只用预测的点群，不使用参考结构。原始与处理后结果分开报告。',
                  '参考匹配使用相同的 StructureMatcher 容差。不同晶型未匹配参考不等于错误；所有这些检查均不代表热力学稳定或实验可合成。', '',
                  '## 可复查的输出', '',
                  '- budget.json：吞吐实测和 500/1000 轮选择依据。',
                  '- data/summary.json：全部样本覆盖、未知 token 和数值边界例外。',
                  '- runs/{explicit,wyckoff}_17/：配置、每轮曲线、last.pt（含优化器和 RNG）、best.pt、每 100 轮权重。',
                  '- evaluation/：验证集选择、所有测试预测和逐例检查，包括失败。projected/runs/*/samples 中的 CIF 使用显式 P1 导出形式，真实对称群按结构重新识别。', '',
                  '## 边界与历史对照', '',
                  '此前小试验是 6,000 条、4,000 steps、batch 32，约 21.3 次平均呈现。本轮扩大数据与训练预算，并按实测吞吐改变 batch；不能把任何改善单独归因于其中一个因素。只有一个训练种子的配对，尚不能给稳健跨种子结论。',
                  'Ho5Pt3 的标准化后近似高对称识别问题通过较严格的共同 encoder 容差解决，保留该材料；codec 与 decoder 没改。原始节点 GPU 初始化不可用的诊断和换节点过程有单独收据。', ''])
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    (ROOT / 'RESULTS.json').write_text(json.dumps({'budget': budget, 'training': training, 'export_checks': export_checks,
                                                 'selection': selection, 'test': results, 'data': data}, indent=2))
    print(json.dumps({'epochs': budget['epochs'], 'test': results}), flush=True)


if __name__ == '__main__': main()
