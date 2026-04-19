"""
Freeze Fig 5 and Fig 6 plot-ready arrays for cross-version comparison.

Extracts from the existing committed results/*.obj files, which were produced
on hpvsim 1.0.0 per requirements_frozen.txt. Run once to snapshot the
"published" baseline; subsequent regenerations on v2.2.6+ will go to their
own versioned dirs.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import hpvsim as hpv
import numpy as np
import pandas as pd
import sciris as sc


def extract_partners(resfolder):
    partner_dict = sc.loadobj(f'{resfolder}/partner_dict.obj')
    bins = np.concatenate([np.arange(21), [100]])
    rows = []
    for location, partners in partner_dict.items():
        for sex in ['f']:
            arr = partners[sex]
            counts, _ = np.histogram(arr, bins=bins)
            total = counts.sum()
            for bi, c in zip(bins[:-1], counts):
                rows.append({
                    'location': location, 'sex': sex,
                    'partner_count_bin': int(bi),
                    'probability': c / total,
                })
    return pd.DataFrame(rows)


def extract_dwelltime(dwelltime_df):
    return (
        dwelltime_df.groupby(['location', 'Health event'])['Age']
        .describe(percentiles=[0.25, 0.5, 0.75])
        .reset_index()
    )


def extract_fig6(resfolder):
    results = sc.loadobj(f'{resfolder}/scen_results.obj')
    rows = []
    for label, res in results.items():
        rows.append({
            'scenario': label,
            'asr_cancer_incidence_2020': float(res.asr_cancer_incidence[-1]),
            'asr_cancer_incidence_low_2020': float(res.asr_cancer_incidence.low[-1]),
            'asr_cancer_incidence_high_2020': float(res.asr_cancer_incidence.high[-1]),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resfolder', default='results')
    parser.add_argument('--outdir', default=f'results/v{hpv.__version__}_baseline')
    parser.add_argument('--dwelltime-csv', default=None,
                        help='Path to a CSV of dwelltime_df if the .obj cannot be unpickled locally')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sources = []

    partners_df = extract_partners(args.resfolder)
    partners_df.to_csv(outdir / 'fig5_partners.csv', index=False)
    sources.append(f'{args.resfolder}/partner_dict.obj')

    try:
        if args.dwelltime_csv:
            dwelltime_df = pd.read_csv(args.dwelltime_csv)
            sources.append(args.dwelltime_csv)
        else:
            dwelltime_df = sc.loadobj(f'{args.resfolder}/dwelltime_df.obj')
            sources.append(f'{args.resfolder}/dwelltime_df.obj')
        extract_dwelltime(dwelltime_df).to_csv(outdir / 'fig5_dwelltime_summary.csv', index=False)
    except Exception as e:
        print(f'WARNING: could not extract dwelltime ({type(e).__name__}: {e}). '
              f'Save a CSV on the VM and re-run with --dwelltime-csv.')

    extract_fig6(args.resfolder).to_csv(outdir / 'fig6_asr.csv', index=False)
    sources.append(f'{args.resfolder}/scen_results.obj')

    manifest = {
        'figures': ['fig5', 'fig6'],
        'source_objects': sources,
        'hpvsim_version': hpv.__version__,
        'date': date.today().isoformat(),
    }
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f'Saved fig5/fig6 baseline to {outdir}')
