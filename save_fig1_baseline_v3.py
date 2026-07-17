"""
Freeze Fig 1 plot-ready arrays on hpvsim v3 (Starsim-based), matching the
schema of save_fig1_baseline.py so results/v3.0.0_baseline/fig1_data.npz can
be diffed against the frozen v2.3 baseline.

v3 API differences from the v2 script:
  - per-genotype natural-history pars come from hpv.get_genotype_pars(gt)
    (a GenotypePars), not sim['genotype_pars'][gt];
  - dur_precin / dur_cin are ss.lognorm_ex Dists; the lognormal mean/std are
    read from dist.pars['mean'] / ['std'] instead of the v2 'par1'/'par2';
  - compute_severity lives in hpvsim.utils, not hpvsim.parameters.

Run with the v3 venv, e.g.:
  ../../hpvsim_claudecontrol/.venv-ss35/Scripts/python.exe save_fig1_baseline_v3.py
"""

import argparse
import json
from datetime import date
from pathlib import Path

import hpvsim as hpv
import numpy as np
import sciris as sc
from hpvsim.utils import compute_severity
from scipy.stats import lognorm

import utils as ut


def compute_fig1_arrays():
    genotypes = ['hpv16', 'hpv18', 'hi5', 'ohr']

    dt = 0.25
    years = np.arange(1, 16, 1).astype(float)
    precinx = np.arange(dt, 15 + dt, dt)
    cinx = np.arange(dt, 30 + dt, dt)

    arrays = {'years': years, 'precinx': precinx, 'cinx': cinx}
    for gt in genotypes:
        gp = hpv.get_genotype_pars(gt)
        dur_precin = gp['dur_precin']
        dur_cin = gp['dur_cin']
        cin_fn = gp['cin_fn']
        cancer_fn = gp['cancer_fn']

        # v3 durations are ss.lognorm_ex Dists parameterized by mean/std.
        p1p, p2p = float(dur_precin.pars['mean']), float(dur_precin.pars['std'])
        p1c, p2c = float(dur_cin.pars['mean']), float(dur_cin.pars['std'])

        sigma_p, scale_p = ut.lognorm_params(p1p, p2p)
        sigma_c, scale_c = ut.lognorm_params(p1c, p2c)

        arrays[f'panelA_{gt}'] = lognorm(sigma_p, 0, scale_p).pdf(years)
        arrays[f'panelB_{gt}'] = compute_severity(precinx, pars=cin_fn)
        arrays[f'panelC_{gt}'] = lognorm(sigma_c, 0, scale_c).pdf(cinx)
        arrays[f'panelD_{gt}'] = compute_severity(
            cinx, pars=sc.mergedicts(cin_fn, cancer_fn)
        )
    return arrays


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
        'date': date.today().isoformat(),
        'arrays': sorted(arrays.keys()),
    }
    (outdir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f'Saved fig1 v3 baseline to {outdir}')
