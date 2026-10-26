"""Recount paired retry trajectories without loading a model or running GRACE."""
import argparse
from collections import Counter
import json
from pathlib import Path

ARMS = ('independent', 'feedback')


def summarize_groups(groups, scheduled):
    assert scheduled > 0 and len(groups) <= scheduled
    assert len({(g['request_id'], g['group_index']) for g in groups}) == len(groups)
    result = dict(status='complete' if len(groups) == scheduled else 'incomplete',
                  scheduled_groups=scheduled, completed_groups=len(groups), arms={},
                  paired=dict(both_success=0, independent_only=0, feedback_only=0, neither=0),
                  actual_candidate_generations=0)
    for group in groups:
        assert group['status'] == 'complete'
        left, right = (group['arms'][arm]['attempts'] for arm in ARMS)
        assert left[0] == right[0], 'initial candidate differs between policies'
        for a, b in zip(left, right):
            assert a['seed'] == b['seed'], 'retry seeds are not paired'
        successes = [bool(group['arms'][arm]['success']) for arm in ARMS]
        key = ('both_success' if all(successes) else 'independent_only' if successes[0]
               else 'feedback_only' if successes[1] else 'neither')
        result['paired'][key] += 1
        result['actual_candidate_generations'] += len(left) + len(right) - 1
    for arm in ARMS:
        within = {str(k): 0 for k in (1, 2, 3)}
        attempts_count, seconds, transitions, repeated = 0, 0., 0, 0
        categories, per_request = Counter(), {}
        for group in groups:
            record = group['arms'][arm]
            attempts = record['attempts']
            assert 1 <= len(attempts) <= 3 and record['attempts_used'] == len(attempts)
            assert record['success'] == attempts[-1]['success']
            assert not any(a['success'] for a in attempts[:-1]), 'continued after success'
            assert record['success'] or len(attempts) == 3, 'unfinished failure group'
            assert [a['round'] for a in attempts] == list(range(len(attempts)))
            for k in within:
                within[k] += bool(record['success'] and len(attempts) <= int(k))
            attempts_count += len(attempts)
            seconds += sum(a['timing_seconds']['total'] for a in attempts)
            for a in attempts:
                categories.update(a['failure_categories'])
            for previous, current in zip(attempts, attempts[1:]):
                transitions += 1
                repeated += bool(set(previous['failure_categories']) & set(current['failure_categories']))
            request = per_request.setdefault(group['request_id'], dict(completed_groups=0, successful_groups=0))
            request['completed_groups'] += 1
            request['successful_groups'] += record['success']
        result['arms'][arm] = dict(success_within_attempts=within,
            success_rate_within_3=within['3'] / scheduled, logical_candidate_attempts=attempts_count,
            mean_attempts_completed_groups=attempts_count / len(groups) if groups else None,
            mean_seconds_completed_groups=seconds / len(groups) if groups else None,
            logical_total_seconds=seconds, retry_transitions=transitions,
            repeated_error_transitions=repeated, failure_categories=dict(categories), per_request=per_request)
    result['scope'] = 'Structural validity and original requested space group only; ehull is an input condition, unevaluated.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.out / 'run_manifest.json').read_text())
    paths = sorted(args.out.glob('shard-*/groups/*/group-*.json'))
    saved = [json.loads(p.read_text()) for p in paths]
    planned = {(p['request_id'], p['attempt_id']): p for p in manifest['group_plan']}
    for group in saved:
        assert group['plan_item'] == planned[(group['request_id'], group['group_index'])]
        assert group['run_identity_sha256'] == manifest['run_identity_sha256']
    groups = [g for g in saved if g['status'] == 'complete']
    result = summarize_groups(groups, scheduled=len(manifest['group_plan']))
    if not args.allow_partial:
        assert result['status'] == 'complete'
    result['manifest'] = 'run_manifest.json'
    result['running_groups'] = len(saved) - len(groups)
    lines = ['# 结构反馈循环试验', '', f"状态：{result['status']}；配对组 {len(groups)}/{result['scheduled_groups']}。", '',
             '| 策略 | 1次内成功 | 2次内成功 | 3次内成功 | 平均尝试次数 | 平均累计秒数 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s = result['arms'][arm]
        values = [s['success_within_attempts'][str(k)] for k in (1, 2, 3)]
        avg = lambda key: f"{s[key]:.2f}" if s[key] is not None else '—'
        lines.append(f"| {arm} | {values[0]} | {values[1]} | {values[2]} | {avg('mean_attempts_completed_groups')} | {avg('mean_seconds_completed_groups')} |")
    p = result['paired']
    lines += ['', f"配对结果：两组均成功 {p['both_success']}；仅独立重采样成功 {p['independent_only']}；仅反馈成功 {p['feedback_only']}；均失败 {p['neither']}。", '',
              '每组原始要求保持不变，最多3次，第一次候选共享。失败保留在全部计划组的分母中；平均成本按已完成组计算，首次生成成本分别计入两种策略。', '',
              f"实际执行候选生成 {result['actual_candidate_generations']} 次（共享首轮只执行一次）。", '',
              '本轮没有训练、GRACE反馈或性质评分。成功仅指严格结构有效且实际空间群符合原始要求，不代表ehull达标。']
    for name, text in [('summary.json', json.dumps(result, indent=2, ensure_ascii=False) + '\n'),
                       ('RESULTS.md', '\n'.join(lines) + '\n')]:
        path = args.out / name
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(text)
        tmp.replace(path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
