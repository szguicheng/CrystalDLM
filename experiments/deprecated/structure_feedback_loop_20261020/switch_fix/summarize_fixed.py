"""Report corrected retries and keep diagnostic groups separate from validation."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('paired_retry_summary', HERE.parent / 'summarize.py')
BASE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BASE)


def switch_counts(groups):
    rows = [r for g in groups for r in g['arms']['feedback']['attempts'][1:]]
    return dict(retries=len(rows), natural_start=sum(r['lead_status'] == 'ok' for r in rows),
                strict_valid=sum(r['structural_valid'] for r in rows),
                successful_retries=sum(r['success'] for r in rows),
                failures=dict(Counter(e for r in rows for e in r['failure_categories'])))


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--out', type=Path, default=HERE / 'run_001')
    parser.add_argument('--allow-partial', action='store_true');args=parser.parse_args()
    manifest=json.loads((args.out / 'run_manifest.json').read_text())
    plan={(p['request_id'],p['attempt_id']):p for p in manifest['group_plan']}
    saved=[json.loads(p.read_text()) for p in args.out.glob('shard-*/groups/*/*.json')]
    groups=[g for g in saved if g['status']=='complete']
    for g in saved:
        assert g['plan_item']==plan[(g['request_id'],g['group_index'])]
        assert g['run_identity_sha256']==manifest['run_identity_sha256']
        case_id=f"{g['request_id']}-{g['group_index']:03d}"
        assert g['evaluation_split']==('diagnostic' if case_id in manifest['diagnostic_case_ids'] else 'heldout')
        source=ROOT/g['reused_from']['path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==g['reused_from']['sha256']
        old=json.loads(source.read_text())
        assert g['arms']['independent']==old['arms']['independent']
        assert g['arms']['feedback']['attempts'][0]==old['arms']['feedback']['attempts'][0]
    result=BASE.summarize_groups(groups,len(plan))
    if not args.allow_partial: assert result['status']=='complete'
    result['switching']=switch_counts(groups)
    result['new_candidate_generations']=result['switching']['retries']
    result['source_reuse']='Independent policy and shared first candidates reused verbatim from pinned original receipts; actual_candidate_generations includes those historical candidates.'
    result['subsets']={}
    for name,count in [('diagnostic',14),('heldout',56)]:
        selected=[g for g in groups if g['evaluation_split']==name]
        subset=BASE.summarize_groups(selected,count)
        subset['switching']=switch_counts(selected)
        result['subsets'][name]=subset
    lines=['# 输入上下文修复验证','',f"状态：{result['status']}；完成 {len(groups)}/{len(plan)} 组。",'',
           '| 集合 | 直接重试成功 | 修复后带反馈重试成功 | 新重试自然进入Diffusion |',
           '|---|---:|---:|---:|']
    for label,record in [('全部70组',result),('诊断用14组',result['subsets']['diagnostic']),('未参与选方案的56组',result['subsets']['heldout'])]:
        n=record['scheduled_groups'];s=record['switching']
        a=record['arms']['independent']['success_within_attempts']['3'];b=record['arms']['feedback']['success_within_attempts']['3']
        lines.append(f"| {label} | {a}/{n} | {b}/{n} | {s['natural_start']}/{s['retries']} |")
    lines+=['','原始目标、权重、种子、采样上限、Diffusion和结构检查未改变；只修改反馈输入的组织方式，没有强制插入START。',
             '共享首轮与直接重试对照按源文件哈希复用，原始评分及生成来源保留。只新生成修复后的反馈重试。',
             '未运行GRACE或训练；成功表示结构严格有效且符合请求空间群。']
    for name,text in [('summary.json',json.dumps(result,indent=2,ensure_ascii=False)+'\n'),('RESULTS.md','\n'.join(lines)+'\n')]:
        path=args.out/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text);tmp.replace(path)
    print(json.dumps(dict(status=result['status'],complete=len(groups),switching=result['switching'],paired=result['paired']),ensure_ascii=False))


if __name__ == '__main__':main()
