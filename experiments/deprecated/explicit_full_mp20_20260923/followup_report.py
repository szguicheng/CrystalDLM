"""Post-selection physical-symmetry audit and concise follow-up report; no repair."""
import json
from pathlib import Path
import sys

import numpy as np
from pymatgen.io.cif import CifWriter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'followup'
sys.path.insert(0, str(ROOT / 'codec'))
import program_codec as sp
import wyckoff_baseline as wp


def predicted_operations(program, representation):
    tokens = program.split()
    structure = (sp if representation == 'explicit' else wp).decode(program)
    if representation == 'wyckoff':
        group = wp.reference_table()[int(tokens[tokens.index('SG') + 1])]
        rotations, translations = group['rotations'], group['translations']
    else:
        origin = np.zeros(3)
        if 'ORIGIN' in tokens:
            index = tokens.index('ORIGIN'); origin = np.array(list(map(float, tokens[index+1:index+4])))
        index = tokens.index('GENERATORS') + 1
        count = int(tokens[index]); index += 1
        rotations, translations = [], []
        for _ in range(count):
            assert tokens[index] == 'G'; index += 1
            coefficients = []
            for _ in range(3):
                expression, index = sp.read_expression(tokens, index)
                coefficients.append(sp.affine_coefficients(expression))
            coefficients = np.array(coefficients, dtype=float)
            rotation = coefficients[:, :3]
            rotations.append(rotation)
            translations.append(coefficients[:, 3] + origin - rotation @ origin)
    return structure, rotations, translations


def symmetry_audit(program, representation):
    structure, rotations, translations = predicted_operations(program, representation)
    metric = structure.lattice.matrix @ structure.lattice.matrix.T
    metric_error = site_error = 0.
    species = np.array(structure.atomic_numbers)
    for rotation, translation in zip(rotations, translations):
        metric_error = max(metric_error, float(np.linalg.norm(rotation.T @ metric @ rotation - metric) / np.linalg.norm(metric)))
        mapped = (structure.frac_coords @ rotation.T + translation) % 1
        distance = structure.lattice.get_all_distances(mapped, structure.frac_coords)
        distance = np.where(species[:, None] == species[None, :], distance, np.inf)
        site_error = max(site_error, float(distance.min(axis=1).max()))
    return {'metric_relative_error': metric_error, 'site_mapping_error_A': site_error,
            'symmetry_consistent': metric_error <= 1e-6 and site_error <= 1e-3}


def main():
    wp.reference_table()
    # A generator can be algebraically valid while not an isometry of its lattice.
    good = 'SP1 CELL 4 4 4 90 90 90 ROUND 6 GENERATORS 1 G y z x SITES 1 C FREE 0.1 FREE 0.2 FREE 0.3 RESIDUALS 0 END'
    assert symmetry_audit(good, 'explicit')['symmetry_consistent']
    assert not symmetry_audit(good.replace('CELL 4 4 4', 'CELL 4 5 6'), 'explicit')['symmetry_consistent']
    # Check the audit against fixed real, encoded training structures.
    checked = 0
    with (ROOT / 'data/records.jsonl').open() as file:
        for line in file:
            r = json.loads(line)
            if r['split'] != 'train' or r['status'] != 'ok': continue
            for rep in ['explicit', 'wyckoff']:
                assert symmetry_audit(r['programs'][rep], rep)['symmetry_consistent'], (r['id'], rep)
            checked += 1
            if checked == 16: break
    selection = json.loads((OUT / 'sampler_selection.json').read_text())
    method = selection['selected']
    rows, audits, examples = [], [], []
    for seed in [17, 29]:
        summaries = json.loads((OUT / f'fresh_frozen/fresh_512_{seed}/summary.json').read_text())
        for summary in summaries:
            rep, mode = summary['representation'], summary['mode']
            folder = OUT / f'fresh_frozen/fresh_512_{seed}' / f'{rep}_{mode}'
            metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
            programs = {r['id']: r for r in map(json.loads, (folder / 'generated.jsonl').read_text().splitlines())}
            consistent = 0
            for r in metrics:
                if not r['usable']: continue
                result = {**r, **symmetry_audit(programs[r['id']]['program'], rep), 'mode': mode}
                audits.append(result)
                consistent += result['symmetry_consistent']
                if rep == 'explicit' and mode == method:
                    examples.append({'id': r['id'], 'seed': seed, 'formula': r['formula'],
                                     'symmetry_consistent': result['symmetry_consistent'],
                                     'cif': str((folder / 'runs' / f'{rep}_{seed}' / 'samples' / f"{r['id']}.cif").relative_to(OUT)),
                                     'program': programs[r['id']]['program']})
            summary['basic_and_symmetry_consistent'] = consistent
            rows.append(summary)
    result = {'selection': selection, 'fresh_results': rows, 'symmetry_audits': audits,
              'explicit_basic_examples': examples, 'audit_real_structure_checks': checked * 2}
    (OUT / 'RESULTS.json').write_text(json.dumps(result, indent=2))
    names = {'confidence': '原 32 步置信度揭示', 'one_step': '一次并行预测', 'random': '随机揭示',
             'content_first': '非 PAD 优先', 'remask': '随机重新遮挡'}
    lines = ['# 显式位点约束模型：组成条件生成的续跑结果', '',
             f"验证集选择：**{names[method]}**。本轮复用已有四个模型，不改变 tokenizer、条件或权重，只比较采样策略。", '',
             '## 新测试数据上的结果', '',
             '新取官方 MP-20 test split 按 material_id SHA256 排序第 512:1024 行，共 512 个材料；与本轮之前的训练、验证和测试 ID 无交集。所有材料均通过两种 codec 的往返检查，无未知 token。采样策略在读取其生成结果之前已冻结。', '',
             '| 表示 | Seed | 采样 | 可解码 | 组成正确 | 基础检查通过 | 基础＋对称一致 | 参考匹配 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        c = row['counts']
        lines.append(f"| {row['representation']} | {row['seed']} | {names[row['mode']]} | {c['decoded']}/512 | {c['composition_match']}/512 | {c['usable']}/512 | {row['basic_and_symmetry_consistent']}/512 | {c['reference_match']}/512 |")
    lines.extend(['', '两个种子评估的是同一批 512 个材料，不能当成 1,024 个独立材料。一个组成只生成一次，没有挑选 best-of-k。', '',
                  '基础检查：解码成功、有限正体积、约分元素配比符合输入、周期结构没有小于 0.5 Å 的原子间距。',
                  '对称一致审计：预测的对称操作应保持晶格度量（||RᵀMR−M||/||M|| ≤ 1e-6），并把原子映射到同元素位点（误差 ≤ 1e-3 Å）。该审计只检查，不投影或修复结构，且没有用于重选采样器。它通过了 32 个真实编码结构的正例检查，以及立方晶格被改成不等轴晶格的反例。',
                  '参考匹配仍沿用第一轮 StructureMatcher 容差；参考匹配和基础检查分别统计，单独匹配原结构不保证无重叠。所有这些检查都不能代替稳定性或 DFT 验证。', '',
                  '## 完整验证集上的采样对照', '',
                  '| 采样 | 可解码 | 组成正确 | 基础检查通过 |', '|---|---:|---:|---:|'])
    for r in selection['ranking']:
        lines.append(f"| {names[r['mode']]} | {r['decoded']}/2048 | {r['composition_match']}/2048 | {r['usable']}/2048 |")
    lines.extend(['', '这里 2,048 次输出 = 同一 512 个验证材料 × 两种表示 × 两个训练种子。按基础通过数、组成正确数、可解码数依次选择，两种表示使用共同策略。一次预测与重新遮挡仅相差 1 个基础通过样本，不构成稳健优劣结论。', '',
                  '## 解释与当前限制', '',
                  '- 原多步采样的零通过数不是 tokenizer 不可学习的证明。随机重新遮挡明显提高解码率，但大部分结构仍不满足输入计量比例。',
                  '- 置信度揭示会优先填写容易的 PAD，并冻结先前输出。与训练中随机遮挡真实内容相比，输入形态和误差反馈不同；当前实验证明采样选择影响很大，尚未逐一确认每个机制的贡献。',
                  '- 本轮最优的验证选择退化为一次并行预测，不能把它称作成功的多步扩散采样。模型仍是掩码去噪训练的非自回归小模型，连续数值使用回归头。',
                  '- 显式表示仍有大量表达式、计数和生成元错误；Wyckoff 对照常见 SG/Hall 不一致、字母不属于预测空间群、自由参数数量错误。',
                  '- 当前数据没有证明显式表示优于 Wyckoff。下一步训练应优先处理语法一致性、组成计量和晶格与对称操作的联合约束，之后再评价表示的收益。', '',
                  '## 显式表示通过基础检查的全部实例', '',
                  '| Material ID | Seed | 生成化学式 | 对称一致 | CIF |', '|---|---:|---|---|---|'])
    for e in examples:
        lines.append(f"| {e['id']} | {e['seed']} | {e['formula']} | {'是' if e['symmetry_consistent'] else '否'} | [预测结构]({e['cif']}) |")
    lines.extend(['', '完整预测（包含所有失败）、每个样本的指标、模型源码/权重 SHA256、选择时间与数据 ID 分别见 fresh_frozen/、sampler_selection.json、fresh_selection.json、RESULTS.json。原始试验结果和模型权重保持不变。', ''])
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'fresh_results': [{k: v for k, v in row.items() if k != 'errors'} for row in rows],
                      'explicit_basic_examples': examples}), flush=True)


if __name__ == '__main__': main()
