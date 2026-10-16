"""Publication-style paired-seed summary from the frozen evaluation receipt."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    assert os.environ.get('SLURM_JOB_ID')
    data = json.loads(Path('frozen_word_evaluation/test/summary.json').read_text())
    seeds = data['seeds']
    variants = ['rle', 'rlemerge']
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    colors = ['#2463A8', '#D37B28', '#448B62']
    for i, seed in enumerate(seeds):
        nll = [data['results'][f'{v}_seed{seed}']['mean_profile_nll'] for v in variants]
        accuracy = [100 * data['results'][f'{v}_seed{seed}']['joint_exact_match'] for v in variants]
        memory = [data['peak_training_cuda_bytes'][v][i] / 2**20 for v in variants]
        for ax, values in zip(axes, [nll, accuracy, memory]):
            ax.plot([0, 1], values, marker='o', markersize=6, linewidth=1.5,
                    color=colors[i], alpha=.85, label=f'Seed {seed}')
    for ax, title, label in zip(axes,
            ['Conditional profile likelihood', 'Protostructure prediction', 'Training memory'],
            ['Profile NLL (lower is better)', 'Top-1 exact match (%)', 'Peak CUDA allocation (MiB)']):
        ax.set_title(title, fontsize=11, pad=12)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xticks([0, 1], ['RLE', 'WP count blocks'])
        ax.set_xlim(-.3, 1.3)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
    axes[2].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Frozen test: {data['structures']:,} structures, three paired training seeds", fontsize=13, y=.99)
    fig.text(.5, .015, '10,000 steps per model; equal parameter count; identical legal-candidate readout. Lines pair the same seed.',
             ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout(rect=[0, .055, 1, .94])
    output = Path('final_report')
    output.mkdir(exist_ok=True)
    fig.savefig(output / 'paired_test_results.png', dpi=180)
    fig.savefig(output / 'paired_test_results.pdf')
    print(json.dumps({'plot_files': ['final_report/paired_test_results.png', 'final_report/paired_test_results.pdf'],
                      'job_id': os.environ['SLURM_JOB_ID']}), flush=True)


if __name__ == '__main__':
    main()
