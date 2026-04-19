"""
Freeze Fig 1 plot-ready arrays for cross-version comparison.

Usage: python save_fig1_baseline.py [--outdir results/v2.2.6_baseline]
"""

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import hpvsim as hpv
import hpvsim.parameters as hppar
import numpy as np
import sciris as sc
from scipy.stats import lognorm

import utils as ut


def compute_fig1_arrays():
    genotypes = ['hpv16', 'hpv18', 'hi5', 'ohr']
    sim = hpv.Sim(genotypes=[16, 18, 'hi5', 'ohr'])
    sim.initialize()

    dt = 0.25
    years = np.arange(1, 16, 1).astype(float)
    precinx = np.arange(dt, 15 + dt, dt)
    cinx = np.arange(dt, 30 + dt, dt)

    arrays = {'years': years, 'precinx': precinx, 'cinx': cinx}
    for gt in genotypes:
        dur_precin = sim['genotype_pars'][gt]['dur_precin']
        dur_cin = sim['genotype_pars'][gt]['dur_cin']
        cin_fn = sim['genotype_pars'][gt]['cin_fn']
        cancer_fn = sim['genotype_pars'][gt]['cancer_fn']

        sigma_p, scale_p = ut.lognorm_params(dur_precin['par1'], dur_precin['par2'])
        sigma_c, scale_c = ut.lognorm_params(dur_cin['par1'], dur_cin['par2'])

        arrays[f'panelA_{gt}'] = lognorm(sigma_p, 0, scale_p).pdf(years)
        arrays[f'panelB_{gt}'] = hppar.compute_severity(precinx, pars=cin_fn)
        arrays[f'panelC_{gt}'] = lognorm(sigma_c, 0, scale_c).pdf(cinx)
        arrays[f'panelD_{gt}'] = hppar.compute_severity(
            cinx, pars=sc.mergedicts(cin_fn, cancer_fn)
        )
    return arrays


def get_hpvsim_commit():
    try:
        hpvsim_dir = Path(hpv.__file__).resolve().parent.parent
        return subprocess.check_output(
            ['git', '-C', str(hpvsim_dir), 'rev-parse', 'HEAD'], text=True
        ).strip()
    except Exception:
        return 'unknown'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=f'results/v{hpv.__version__}_baseline')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    arrays = compute_fig1_arrays()
    np.savez(outdir / 'fig1_data.npz', **arrays)

    manifest = {
        'figure': 'fig1',
        'hpvsim_version': hpv.__version__,
        'hpvsim_commit': get_hpvsim_commit(),
        'date': date.today().isoformat(),
        'arrays': sorted(arrays.keys()),
    }
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f'Saved fig1 baseline to {outdir}')
