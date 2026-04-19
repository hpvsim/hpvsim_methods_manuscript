"""
Side-by-side comparison of plot-ready baselines across hpvsim versions.

Usage: python compare_baselines.py [--baselines v1.0.0_published v2.2.6_baseline]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare_fig5_partners(baselines, resfolder, outpath):
    dfs = {label: pd.read_csv(Path(resfolder) / label / 'fig5_partners.csv') for label in baselines}
    locations = sorted(next(iter(dfs.values()))['location'].unique())

    fig, axes = plt.subplots(1, len(locations), figsize=(6 * len(locations), 4), layout='tight')
    if len(locations) == 1:
        axes = [axes]
    n = len(baselines)
    width = 0.8 / n
    for ax, loc in zip(axes, locations):
        for i, (label, df) in enumerate(dfs.items()):
            sub = df[(df.location == loc) & (df.sex == 'f')].sort_values('partner_count_bin')
            x = sub['partner_count_bin'].values + (i - (n - 1) / 2) * width
            ax.bar(x, sub['probability'].values, width=width, label=label)
        ax.set_title(f'Fig 5 partners — {loc}')
        ax.set_xlabel('Lifetime casual partners')
        ax.set_ylabel('Probability')
        ax.legend()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def compare_fig5_dwelltime(baselines, resfolder, outpath):
    dfs = {label: pd.read_csv(Path(resfolder) / label / 'fig5_dwelltime_summary.csv') for label in baselines}
    events = sorted(next(iter(dfs.values()))['Health event'].unique())
    locations = sorted(next(iter(dfs.values()))['location'].unique())

    fig, axes = plt.subplots(1, len(locations), figsize=(6 * len(locations), 4.5), layout='tight')
    if len(locations) == 1:
        axes = [axes]
    n = len(baselines)
    width = 0.8 / n
    x_base = np.arange(len(events))
    for ax, loc in zip(axes, locations):
        for i, (label, df) in enumerate(dfs.items()):
            sub = df[df.location == loc].set_index('Health event').reindex(events)
            medians = sub['50%'].values
            q1 = sub['25%'].values
            q3 = sub['75%'].values
            err = np.vstack([medians - q1, q3 - medians])
            xs = x_base + (i - (n - 1) / 2) * width
            ax.bar(xs, medians, width=width, yerr=err, capsize=4, label=label)
        ax.set_xticks(x_base, events, rotation=15)
        ax.set_ylabel('Age (median ± IQR)')
        ax.set_title(f'Fig 5 dwelltime — {loc}')
        ax.legend()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def compare_fig6(baselines, resfolder, outpath):
    dfs = {label: pd.read_csv(Path(resfolder) / label / 'fig6_asr.csv') for label in baselines}
    scenarios = next(iter(dfs.values()))['scenario'].tolist()

    fig, ax = plt.subplots(figsize=(9, 5), layout='tight')
    n = len(baselines)
    width = 0.8 / n
    x_base = np.arange(len(scenarios))
    for i, (label, df) in enumerate(dfs.items()):
        df = df.set_index('scenario').reindex(scenarios)
        mean = df['asr_cancer_incidence_2020'].values
        err_lo = mean - df['asr_cancer_incidence_low_2020'].values
        err_hi = df['asr_cancer_incidence_high_2020'].values - mean
        xs = x_base + (i - (n - 1) / 2) * width
        ax.bar(xs, mean, width=width, yerr=np.vstack([err_lo, err_hi]), capsize=4, label=label)
    ax.axhline(18, linestyle='--', color='k', alpha=0.5, label='WHO target (18/100k)')
    ax.set_xticks(x_base, scenarios, rotation=15)
    ax.set_ylabel('ASR cancer incidence (per 100k), 2020')
    ax.set_title('Fig 6 — ASR by screening coverage')
    ax.legend()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', nargs='+', default=['v1.0.0_published', 'v2.2.6_baseline'])
    parser.add_argument('--resfolder', default='results')
    parser.add_argument('--outdir', default='figures/compare')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    compare_fig5_partners(args.baselines, args.resfolder, outdir / 'fig5_partners_compare.png')
    compare_fig5_dwelltime(args.baselines, args.resfolder, outdir / 'fig5_dwelltime_compare.png')
    compare_fig6(args.baselines, args.resfolder, outdir / 'fig6_compare.png')

    print(f'Wrote comparison figures to {outdir}')
