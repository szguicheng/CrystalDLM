"""Regrade saved responses and rebuild counts; never run model or energy inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


READ = module('refresh_readback', 'experiments/evaluation_common/readback.py')
SCORE = module('refresh_structure', 'experiments/interleaved_ar_diffusion_20261002/score.py')
RUN = module('refresh_counts', 'experiments/structure_request_eval_20261017/run.py')
GRACE = module('refresh_grace', 'experiments/structure_request_eval_20261017/grace_score.py')
FIELDS = ('formula', 'space_group', 'cell_atom_count', 'independent_orbit_count')
CHANGED = []


def read(path):
    return json.loads(path.read_text())


def write(path, value, jsonl=False):
    raw = (''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in value) if jsonl
           else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    if not path.exists() or path.read_text() != raw:
        RUN._write_json(path, value) if not jsonl else atomic_text(path, raw)
        CHANGED.append(str(path.relative_to(ROOT)))


def atomic_text(path, raw):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(raw)
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(answers):
    return dict(attempted=len(answers), all_four_correct=sum(a['all_four_correct'] for a in answers),
                per_field={f: sum(a['fields'][f] for a in answers) for f in FIELDS},
                evaluator_version=READ.VERSION)


def strip_grades(row):
    row = copy.deepcopy(row)
    for key in ('readback', 'tail_read'):
        for field in ('fields', 'all_four_correct', 'evaluator_version', 'evaluation'):
            row.get(key, {}).pop(field, None)
    return row


def old_jsonl(relative):
    out = ROOT / relative
    path = out / 'evaluation.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    original = [strip_grades(r) for r in rows]
    cases = []
    for row in rows:
        if not row.get('crystal_audit', {}).get('valid') or not row.get('tail_read', {}).get('raw'):
            continue
        facts = SCORE.parse(row['crystal']['raw'])
        grade = READ.grade_four_facts(row['tail_read']['raw'], facts['formula'],
            row['crystal_audit']['detected_sg'], row['crystal_audit']['decoded_atoms'], facts['orbits'])
        row['tail_read']['evaluation'] = grade
        cases.append(dict(material_id=row['material_id'], **grade))
    assert original == [strip_grades(r) for r in rows], 'raw generation changed'
    write(path, rows, jsonl=True)
    total = counts(cases)
    write(out / 'tail-four-facts.json', dict(**total, cases=cases))
    summary_path = out / 'summary.json'
    if summary_path.exists():
        summary = read(summary_path)
        summary['readback_evaluation'] = total
        write(summary_path, summary)
    return dict(run=relative, attempts=len(rows), strict_valid=sum(r.get('crystal_audit', {}).get('valid') is True for r in rows), readback=total)


def refresh_modern(out):
    paths = sorted(out.glob('shard-*/attempts/*/*.json'))
    rows = []
    for path in paths:
        row = read(path)
        original = strip_grades(row)
        answer = row.get('readback', {})
        if answer.get('response') is not None:
            facts = answer['reference_facts']
            answer.update(READ.grade_four_facts(answer['response'], facts['validated_cell_formula'],
                facts['decoded_space_group_number'], facts['decoded_cell_atom_count'],
                facts['validated_independent_orbit_count']))
        assert original == strip_grades(row), 'raw generation changed'
        write(path, row)
        rows.append(row)
        score_path = out / 'grace_score/attempts' / path.parent.name / path.name
        if score_path.exists():
            physics = read(score_path)
            original_physics = copy.deepcopy(physics)
            physics['generation_receipt']['sha256'] = sha(path)
            original_physics['generation_receipt']['sha256'] = sha(path)
            assert physics == original_physics, 'property score changed'
            write(score_path, physics)
    manifest = read(out / 'run_manifest.json')
    plan = manifest['attempt_plan']
    assert len(rows) == len({(r['request_id'], r['attempt_id']) for r in rows}) == len(plan)
    for base, subset in [(out, rows)] + [(p, [read(q) for q in sorted(p.glob('attempts/*/*.json'))]) for p in sorted(out.glob('shard-*')) if p.is_dir()]:
        slots = {(r['request_id'], r['attempt_id']) for r in subset}
        subplan = [p for p in plan if (p['request_id'], p['attempt_id']) in slots]
        path = base / 'summary.json'
        summary = read(path) if path.exists() else {}
        before = summary.get('strict_valid_structures')
        summary.update(RUN.summarize(subplan, subset))
        assert before is None or before == summary['strict_valid_structures']
        if any('readback' in r for r in subset):
            answers = [r['readback'] for r in subset if r.get('readback', {}).get('fields')]
            total = counts(answers)
            summary.update(readback_all_four_correct=total['all_four_correct'],
                readback_field_correct_counts=total['per_field'], readback_denominator=total['attempted'],
                readback_attempts=total['attempted'],
                readback_eos_completed=sum(a.get('status') == 'eos' for a in answers),
                readback_skipped_invalid_structures=sum(r.get('structural_valid') is not True for r in subset),
                readback_evaluator_version=READ.VERSION)
        write(path, summary)
    result = dict(run=str(out.relative_to(ROOT)), attempts=len(rows), strict_valid=sum(r.get('structural_valid') is True for r in rows),
                  valid_requested_sg=sum(r.get('structural_valid') is True and r.get('requested_sg_match') is True for r in rows))
    if any('readback' in r for r in rows):
        result['readback'] = counts([r['readback'] for r in rows if r.get('readback', {}).get('fields')])
    path = out / 'grace_score/summary.json'
    if path.exists():
        summary = read(path)
        scores = [read(q) for q in sorted((out / 'grace_score/attempts').glob('*/*.json'))]
        recomputed = GRACE.summarize_scores(plan, scores)
        assert summary['threshold_counts'] == recomputed['threshold_counts'], 'physics aggregation drift'
        summary.update(recomputed)
        write(path, summary)
        result['final_unique_novel_conditioned_80meV'] = summary['threshold_counts']['0.08']['global_unique_novel_conditioned_converged_leq']
    return result


def main():
    frozen = ROOT / 'experiments/interleaved_ar_diffusion_20261002/score.py'
    assert sha(frozen) == 'c1dfe1b87715b5b650b7152e02fed567bd5f7f0fd9109295ec87208f23165c6b'
    results = [old_jsonl(p) for p in (
        'experiments/ehull_pure_ar_baseline_20261013/run_001',
        'experiments/ehull_pure_ar_qwen3_8b_20261015/evaluation_001')]
    for family in ('structure_request_eval_20261017', 'structure_conditioned_ar8b_20261017',
                   'structure_conditioned_hybrid8b_20261018', 'flare_crystal_full_20261019'):
        for manifest in sorted((ROOT / 'experiments' / family).glob('*/run_manifest.json')):
            if list(manifest.parent.glob('shard-*/attempts/*/*.json')):
                results.append(refresh_modern(manifest.parent))
    full = ROOT / 'experiments/flare_crystal_full_20261019'
    combined = read(full / 'summary.json')
    readback = dict(evaluator_version=READ.VERSION, scope='Canonical regraded receipts; raw model outputs unchanged', arms={})
    for arm in ('ar', 'flare'):
        generation = read(full / f'{arm}_evaluation/summary.json')
        combined['arms'][arm]['generation'] = generation
        combined['arms'][arm]['scoring'] = read(full / f'{arm}_evaluation/grace_score/summary.json')
        readback['arms'][arm] = dict(n=generation['readback_denominator'],
            formula_correct=generation['readback_field_correct_counts']['formula'],
            all_four_correct=generation['readback_all_four_correct'], fields=generation['readback_field_correct_counts'])
    write(full / 'summary.json', combined)
    write(full / 'review_20261020/readback-summary.json', readback)
    report = dict(status='complete', evaluator_version=READ.VERSION,
        grader_sha256=sha(ROOT / 'experiments/evaluation_common/readback.py'),
        frozen_structure_evaluator_sha256=sha(frozen), runs=results,
        invariants='Raw model responses, token IDs, CIFs, structural decisions and per-attempt energy/novelty values unchanged; aggregate counts recomputed.',
        scope='Saved evidence only; no model inference, training, relaxation or DFT.')
    write(HERE / 'results.json', report)
    # The union remains stable on reruns and identifies precisely which artifacts to synchronize.
    change_path = HERE / 'changed-files.json'
    previous = read(change_path) if change_path.exists() else []
    atomic_text(change_path, json.dumps(sorted(set(previous + CHANGED)), indent=2) + '\n')
    print(json.dumps(dict(runs=results, files_changed_this_pass=len(CHANGED)), indent=2))


if __name__ == '__main__':
    main()
