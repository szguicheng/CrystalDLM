"""Exact one-element Wyckoff occupation-profile counts; run on Slurm CPUs."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import re
import resource
import socket
import string
import time
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


def evaluate(node, xyz):
    if isinstance(node, ast.Expression):
        return evaluate(node.body, xyz)
    if isinstance(node, ast.Constant):
        return Fraction(str(node.value))
    if isinstance(node, ast.Name):
        return xyz['xyz'.index(node.id)]
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, xyz)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = evaluate(node.left, xyz), evaluate(node.right, xyz)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(ast.dump(node))


def exact_rank(matrix):
    a = [row[:] for row in matrix]
    rank = 0
    for column in range(3):
        pivot = next((r for r in range(rank, 3) if a[r][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][column]
        a[rank] = [v / scale for v in a[rank]]
        for r in range(rank + 1, 3):
            factor = a[r][column]
            a[r] = [x - factor * y for x, y in zip(a[r], a[rank])]
        rank += 1
    return rank


def coordinate_matrix(operation):
    matrix = []
    for expression in operation.split(','):
        expression = re.sub(r'(\d)([xyz])', r'\1*\2', expression)
        node = ast.parse(expression.strip(), mode='eval')
        origin = evaluate(node, (0, 0, 0))
        coeff = [Fraction(evaluate(node, tuple(int(j == k) for j in range(3))) - origin)
                 for k in range(3)]
        assert evaluate(node, (2, 3, 5)) == origin + sum(c * v for c, v in zip(coeff, (2, 3, 5)))
        matrix.append(coeff)
    assert len(matrix) == 3
    return matrix


def read_groups(directory):
    symbols = json.loads((directory / 'symbols.json').read_text())['space_group']
    groups = {}
    with (directory / 'wyckoff_list.csv').open() as stream:
        reader = csv.reader(stream)
        next(reader)
        for index, raw in reader:
            if index == '0':
                continue
            sg = int(index)
            positions = ast.literal_eval(raw)
            rows = []
            for i, operations in enumerate(positions):
                matrix = coordinate_matrix(operations[0])
                dof = exact_rank(matrix)
                rows.append({'letter': string.ascii_letters[len(positions) - i - 1],
                             'multiplicity': len(operations), 'dof': dof,
                             'representative_map': operations[0],
                             'capacity': 1 if dof == 0 else None})
            groups[sg] = {'symbol': symbols[sg - 1], 'positions': rows}
    assert set(groups) == set(range(1, 231))
    assert sum(len(g['positions']) for g in groups.values()) == 1731
    return groups


def coefficients(positions, maximum):
    """[x^N] product(1+x^mu) for fixed WPs, product(1-x^mu)^-1 otherwise."""
    dp = [1] + [0] * maximum
    for position in positions:
        weight = position['multiplicity']
        values = range(maximum, weight - 1, -1) if position['dof'] == 0 else range(weight, maximum + 1)
        for n in values:
            dp[n] += dp[n - weight]
    return dp


def multiplicity_partition_coefficients(positions, maximum):
    """Collapse WP labels sharing a multiplicity; preserve possible total orbit counts."""
    caps = {}
    for p in positions:
        mu = p['multiplicity']
        if p['dof'] > 0:
            caps[mu] = None
        elif caps.get(mu, 0) is not None:
            caps[mu] = caps.get(mu, 0) + 1
    dp = [1] + [0] * maximum
    for mu, cap in sorted(caps.items()):
        updated = [0] * (maximum + 1)
        for total in range(maximum + 1):
            limit = total // mu if cap is None else min(cap, total // mu)
            updated[total] = sum(dp[total - k * mu] for k in range(limit + 1))
        dp = updated
    return dp


def enumerate_profiles(positions, total):
    """Independent bounded DFS, used only for small-case verification."""
    profile = [0] * len(positions)

    def visit(index, remaining):
        if index == len(positions):
            if remaining == 0:
                yield tuple(profile)
            return
        p = positions[index]
        limit = remaining // p['multiplicity']
        if p['dof'] == 0:
            limit = min(limit, 1)
        for k in range(limit + 1):
            profile[index] = k
            yield from visit(index + 1, remaining - k * p['multiplicity'])

    yield from visit(0, total)


def verify(groups):
    import numpy as np
    start = time.perf_counter()
    cases = profiles_total = 0
    for sg, group in groups.items():
        positions = group['positions']
        for p in positions:
            matrix = coordinate_matrix(p['representative_map'])
            assert p['dof'] == int(np.linalg.matrix_rank(np.array(matrix, dtype=float)))
        expected = coefficients(positions, 6)
        partition_expected = multiplicity_partition_coefficients(positions, 6)
        for n in range(1, 7):
            profiles = list(enumerate_profiles(positions, n))
            assert len(profiles) == expected[n], (sg, n, len(profiles), expected[n])
            partitions = {tuple(sorted(mu for p, k in zip(positions, profile)
                                       for mu in [p['multiplicity']] * k)) for profile in profiles}
            assert len(partitions) == partition_expected[n], (sg, n)
            cases += 1
            profiles_total += len(profiles)
    assert coefficients(groups[1]['positions'], 40) == [1] * 41
    inversion = coefficients(groups[2]['positions'], 40)
    assert inversion[1] == 8 and inversion[2] == 29 and inversion[40] == 128
    cubic = coefficients(groups[225]['positions'], 8)
    assert cubic[1:4] == [0, 0, 0] and cubic[4] == 2 and cubic[8] == 2
    return {'status': 'PASS', 'brute_force_contexts': cases,
            'brute_force_profiles': profiles_total, 'dof_rank_crosschecks': 1731,
            'analytic_fixtures': ['P1: one occupation profile for every N',
                                  'P-1: (1+x)^8/(1-x^2)',
                                  'Fm-3m: V(4)=2, V(8)=2'],
            'seconds': time.perf_counter() - start}


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def describe(rows):
    positive = [r['profiles'] for r in rows if r['profiles'] > 0]
    return {'contexts': len(rows), 'feasible_contexts': len(positive),
            'infeasible_contexts': len(rows) - len(positive),
            'sum_group_scoped_profiles': sum(r['profiles'] for r in rows),
            'median_feasible': percentile(positive, .5),
            'p90_feasible': percentile(positive, .9),
            'p95_feasible': percentile(positive, .95),
            'p99_feasible': percentile(positive, .99),
            'max': max(positive),
            'feasible_at_most': {str(t): sum(v <= t for v in positive)
                                 for t in [10, 100, 1000, 10000, 100000, 1000000]},
            'top_contexts': sorted(rows, key=lambda r: (-r['profiles'], r['space_group'], r['atom_count']))[:20]}


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, destination, maximum):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    matrix = np.array([[r['profiles'] for r in rows if r['space_group'] == sg]
                       for sg in range(1, 231)], dtype=float)
    cmap = plt.colormaps['viridis'].copy()
    cmap.set_bad('#eeeeee')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    im = axes[0].imshow(np.ma.masked_equal(matrix, 0), origin='lower', aspect='auto',
                        extent=[.5, maximum + .5, .5, 230.5], cmap=cmap,
                        norm=LogNorm(vmin=1, vmax=matrix.max()), interpolation='nearest')
    axes[0].set(xlabel='Atoms of one element in the tabulated cell (N)', ylabel='Space-group number',
                title='Exact occupation-profile count\nGray = incompatible atom count')
    fig.colorbar(im, ax=axes[0], label='Profiles (log scale)')
    for upper, color in [(20, '#207b8b'), (maximum, '#bd6d32')]:
        data = sorted(r['profiles'] for r in rows if 0 < r['atom_count'] <= upper and r['profiles'])
        axes[1].step(data, np.arange(1, len(data) + 1) / len(data), where='post',
                     label=f'N = 1-{upper}; compatible contexts', color=color)
    axes[1].set(xscale='log', xlabel='Candidate profiles per (space group, N)',
                ylabel='Cumulative fraction of contexts', ylim=(0, 1.02),
                title='Unweighted context distribution\nNot a material-frequency distribution')
    axes[1].grid(alpha=.2)
    axes[1].legend(loc='lower right')
    fig.savefig(destination / 'vocabulary_scale.png', dpi=180)
    fig.savefig(destination / 'vocabulary_scale.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('input'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--max-atoms', type=int, default=40)
    args = parser.parse_args()
    assert args.max_atoms >= 20
    assert os.environ.get('SLURM_JOB_ID'), 'Run this experiment as a Slurm batch job.'
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.input / 'provenance.json').read_text())
    for name, expected in provenance['files'].items():
        assert hashlib.sha256((args.input / Path(name).name).read_bytes()).hexdigest() == expected
    groups = read_groups(args.input)
    validation = verify(groups)
    print(json.dumps({'validation': validation}), flush=True)
    count_start = time.perf_counter()
    rows = []
    metadata = []
    for sg, group in groups.items():
        positions = group['positions']
        counts = coefficients(positions, args.max_atoms)
        partitions = multiplicity_partition_coefficients(positions, args.max_atoms)
        metadata += [{'space_group': sg, 'symbol': group['symbol'], **p} for p in positions]
        for n in range(1, args.max_atoms + 1):
            assert counts[n] >= partitions[n]
            rows.append({'space_group': sg, 'symbol': group['symbol'], 'atom_count': n,
                         'profiles': counts[n], 'multiplicity_partitions': partitions[n],
                         'distinct_wyckoff_positions': len(positions),
                         'fixed_positions': sum(p['dof'] == 0 for p in positions),
                         'free_positions': sum(p['dof'] > 0 for p in positions)})
    count_seconds = time.perf_counter() - count_start
    write_csv(args.output / 'counts.csv', rows)
    write_csv(args.output / 'wyckoff_metadata.csv', metadata)
    write_csv(args.output / 'top_contexts.csv', sorted(rows, key=lambda r: -r['profiles'])[:50])
    by_n = [{'atom_count': n, 'feasible_groups': sum(r['profiles'] > 0 for r in rows if r['atom_count'] == n),
             'sum_profiles': sum(r['profiles'] for r in rows if r['atom_count'] == n),
             'max_profiles': max(r['profiles'] for r in rows if r['atom_count'] == n)}
            for n in range(1, args.max_atoms + 1)]
    write_csv(args.output / 'by_atom_count.csv', by_n)
    summary = {'scope': {'space_groups': 230, 'N_min': 1, 'N_max': args.max_atoms,
                         'cell': 'PyXtal tabulated conventional setting; rhombohedral groups use its default hexagonal cell',
                         'counted_object': 'One-element integer occupation multiplicities per labelled Wyckoff position',
                         'quotient_by_equivalent_origins': False, 'cross_element_collisions_checked': False,
                         'geometry_or_stability_evaluated': False, 'material_frequency_weighted': False,
                         'enumeration_truncated': False},
               'source': provenance, 'validation': validation,
               'N_1_20': describe([r for r in rows if r['atom_count'] <= 20]),
               'N_1_max': describe(rows),
               'multiplicity_partition_max': max(r['multiplicity_partitions'] for r in rows),
               'multiplicity_partition_sum': sum(r['multiplicity_partitions'] for r in rows),
               'execution': {'host': socket.gethostname(), 'python': platform.python_version(),
                             'slurm_job_id': os.environ['SLURM_JOB_ID'],
                             'slurm_partition': os.environ.get('SLURM_JOB_PARTITION'),
                             'cpu_allocation': os.environ.get('SLURM_CPUS_PER_TASK'),
                             'counting_seconds': count_seconds,
                             'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'finished_utc': datetime.now(timezone.utc).isoformat()}}
    plot_results(rows, args.output, args.max_atoms)
    summary['execution']['total_script_seconds'] = time.perf_counter() - start
    summary['execution']['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'completed': True, 'maximum': summary['N_1_max']['max'],
                      'execution': summary['execution']}), flush=True)


if __name__ == '__main__':
    main()
