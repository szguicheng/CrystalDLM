"""Recheck the old 32 raw outputs under the new server's scoring environment."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/interleaved_ar_diffusion_20261002'))
from score import audit


def main():
    source = ROOT / 'experiments/ehull_pure_ar_baseline_20261013/run_001/evaluation.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    comparisons = []
    for row in rows:
        original = row['crystal_audit']
        actual = audit(row['crystal']['raw'])
        comparisons.append({'material_id': row['material_id'], 'old_valid': original['valid'],
                            'new_valid': actual['valid'], 'old_reasons': original['reasons'],
                            'new_reasons': actual['reasons']})
    report = {'cases': len(rows), 'old_valid': sum(r['old_valid'] for r in comparisons),
              'new_valid': sum(r['new_valid'] for r in comparisons),
              'validity_equal': all(r['old_valid'] == r['new_valid'] for r in comparisons),
              'reasons_equal': all(r['old_reasons'] == r['new_reasons'] for r in comparisons),
              'cases_compared': comparisons}
    output = ROOT / 'research/liugang_setup_20261015/scorer-regression.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    assert report['cases'] == 32 and report['old_valid'] == report['new_valid'] == 7
    assert report['validity_equal'] and report['reasons_equal'], 'scorer behavior changed'
    print(json.dumps({k:v for k,v in report.items() if k != 'cases_compared'}))


if __name__ == '__main__':
    main()
