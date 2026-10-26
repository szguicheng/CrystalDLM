"""Exploratory symmetry projection from predicted operations only; never reference SG."""
import json
import numpy as np
from pymatgen.core import Lattice

from followup_report import ROOT, OUT, sp, wp, predicted_operations, symmetry_audit
import evaluate


def project(program, representation):
    structure, rotations, _ = predicted_operations(program, representation)
    # Only rotations matter to the metric. Close explicit point generators; the
    # same closure is harmless for the already complete Wyckoff operation list.
    generators = [sp.op_key(r, [0, 0, 0], 1) for r in rotations]
    group = sp.closure(generators, 1)
    rotations = [np.array(key[:9]).reshape(3, 3) for key in group]
    metric = structure.lattice.matrix @ structure.lattice.matrix.T
    invariant = np.mean([r.T @ metric @ r for r in rotations], axis=0)
    parameters = Lattice(np.linalg.cholesky(invariant)).parameters
    tokens = program.split(); index = tokens.index('CELL') + 1
    tokens[index:index+6] = [format(v, '.14g') for v in parameters]
    return ' '.join(tokens)


def main():
    wp.reference_table()
    good = 'SP1 CELL 4 4 4 90 90 90 ROUND 6 GENERATORS 1 G y z x SITES 1 C FREE 0.1 FREE 0.2 FREE 0.3 RESIDUALS 0 END'
    bad = good.replace('CELL 4 4 4', 'CELL 4 5 6')
    assert not symmetry_audit(bad, 'explicit')['symmetry_consistent']
    repaired = project(bad, 'explicit')
    assert symmetry_audit(repaired, 'explicit')['symmetry_consistent']
    assert np.allclose(sp.decode(project(repaired, 'explicit')).lattice.metric_tensor,
                       sp.decode(repaired).lattice.metric_tensor, atol=1e-10, rtol=1e-10)
    checked = 0
    with (ROOT / 'data/records.jsonl').open() as file:
        for line in file:
            r = json.loads(line)
            if r['split'] != 'train' or r['status'] != 'ok': continue
            for rep in ['explicit', 'wyckoff']:
                original, _, _ = predicted_operations(r['programs'][rep], rep)
                transformed, _, _ = predicted_operations(project(r['programs'][rep], rep), rep)
                assert np.allclose(original.lattice.metric_tensor, transformed.lattice.metric_tensor, atol=1e-8, rtol=1e-8)
            checked += 1
            if checked == 16: break
    selection = json.loads((OUT / 'sampler_selection.json').read_text())
    method = selection['selected']
    references = {r['id']: r for r in map(json.loads, (OUT / 'fresh_records.jsonl').read_text().splitlines())}
    summaries = []
    for rep in ['explicit', 'wyckoff']:
        for seed in [17, 29]:
            source = OUT / f'fresh_frozen/fresh_512_{seed}' / f'{rep}_{method}'
            raw = {r['id']: r for r in map(json.loads, (source / 'metrics.jsonl').read_text().splitlines())}
            generated = [json.loads(line) for line in (source / 'generated.jsonl').read_text().splitlines()]
            destination = OUT / 'projected' / f'{rep}_{seed}'; destination.mkdir(parents=True, exist_ok=True)
            evaluate.ROOT = destination
            (destination / 'runs' / f'{rep}_{seed}').mkdir(parents=True, exist_ok=True)
            results, programs = [], []
            for record in generated:
                if raw[record['id']]['decoded']:
                    record = {**record, 'program': project(record['program'], rep)}
                    result = evaluate.score_one((record, references[record['id']]))
                    result.update(symmetry_audit(record['program'], rep))
                    result['basic_and_symmetry_consistent'] = result['usable'] and result['symmetry_consistent']
                    programs.append(record)
                else:
                    result = {**raw[record['id']], 'basic_and_symmetry_consistent': False}
                results.append(result)
            (destination / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results))
            (destination / 'programs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in programs))
            summary = {'representation': rep, 'seed': seed, 'n': len(results),
                       'counts': {key: sum(r[key] for r in results) for key in ['decoded', 'composition_match', 'usable', 'basic_and_symmetry_consistent', 'reference_match']},
                       'passed_examples': [{k: r.get(k) for k in ['id', 'formula', 'generated_sg', 'reference_match']} for r in results if r['basic_and_symmetry_consistent']]}
            summaries.append(summary)
            (destination / 'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
    (OUT / 'projection_results.json').write_text(json.dumps({'status': 'exploratory postprocessing after fresh test inspection; no retuning or fresh independent validation',
                'uses_reference_information': False, 'method': 'M_new = mean(R.T @ M_pred @ R) over predicted finite point group',
                'checks': {'positive_negative_and_idempotence': True, 'unchanged_real_structure_metrics': checked * 2},
                'results': summaries}, indent=2))
    lines = ['\n## 探索性后处理：仅按预测对称操作约束晶格\n',
             '这一步在查看新测试集的原始结果后追加，复用同一批材料；没有新的独立测试，因此与上面的冻结测试分开报告。模型权重、生成原子与组成均不变。仅用预测的点群计算 M_new = mean(Rᵀ M_pred R)，然后更新六个 CELL 数值。没有使用真实空间群、原晶格、目标位点或目标坐标。\n',
             '| 表示 | Seed | 可解码 | 组成正确 | 基础＋对称一致 | 参考匹配 |', '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        c = r['counts']; lines.append(f"| {r['representation']} | {r['seed']} | {c['decoded']}/512 | {c['composition_match']}/512 | {c['basic_and_symmetry_consistent']}/512 | {c['reference_match']}/512 |")
    lines.extend(['', '该操作只能消除预测晶格与预测对称操作之间的矛盾，不能修复错误的化学配比、语法、空间群选择或材料稳定性。通过结构检查的比例仍很低。检查通过的 CIF 保存在 projected/{representation}_{seed}/runs/{representation}_{seed}/samples/；具体通过清单见 projection_results.json。', ''])
    with (OUT / 'RESULTS.md').open('a') as file: file.write('\n'.join(lines))


if __name__ == '__main__': main()
