"""
Freeze Fig 5 plot-ready arrays on hpvsim v3 (Starsim-based), matching the
schema of save_fig56_baseline.py so results/<ver>_baseline/fig5_*.csv can be
diffed against the frozen v2.3 baselines.

Consolidates the M10 reference scripts run_v3_fig5_partners.py and
run_v3_fig5_dwelltime.py: for each seed it runs ONE baseline india sim
(no screening) carrying both analyzers, then writes:
  - fig5_partners.csv         (location, sex, partner_count_bin, probability)
  - fig5_dwelltime_summary.csv (location, Health event, count, mean, std,
                                min, 25%, 50%, 75%, max)

v3 API notes vs the v2 plot_fig56.run_sims:
  - v2's people.n_rships (cumulative per-agent partner counter) is gone; v3
    keeps only a live edge table, so LifetimeCasualPartners accumulates casual
    formations per female by counting edges whose start_ti == the current tick.
  - dwell-time uses the built-in hpv.age_causal_infection analyzer (records
    age_causal / age_cin / age_cancer per cancer, scale-weighted, engine-correct
    at any ms_agent_ratio) instead of the hand-rolled dwelltime_by_genotype.

Run with the v3 venv:
  .venv/Scripts/python.exe save_fig5_baseline_v3.py --outdir results/v3.0.0_baseline
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss
import hpvsim as hpv

from v3_india_config import make_india_sim

LOCATION = 'india'
BINS = np.concatenate([np.arange(21), [100]])  # 0,1,...,20,100  (bin 20 = 20+)


class LifetimeCasualPartners(ss.Analyzer):
    """Cumulative lifetime casual partnerships per active base (level0) female.

    v3 keeps only a current edge table (dissolved edges are dropped), so we
    accumulate casual-layer formations per female by counting edges whose
    start_ti == the current tick each step. Fine multiscale agents don't
    partner (excluded from the network), matching v2's level0 filter.
    """
    def init_pre(self, sim):
        self.net = next(n for n in sim.networks.values() if isinstance(n, hpv.SexualNetwork))
        super().init_pre(sim)
        self.casual = self.net._layer_idx['c']
        self.count = {}
        self.snapshot = None

    def step(self):
        e = self.net.edges
        start_ti = np.asarray(e.start_ti)
        new = (start_ti == self.sim.ti) & (np.asarray(e.layer_id) == self.casual)
        for u in np.asarray(e.p1)[new]:  # p1 == female partner
            self.count[int(u)] = self.count.get(int(u), 0) + 1

    def finalize(self):
        super().finalize()
        people = self.sim.people
        uids = people.auids
        female = people.female.values
        fine = (people.fine.values if 'fine' in people.states
                else np.zeros(len(uids), dtype=bool))
        active = people.age.values >= self.net.debut.values
        sel = female & (~fine) & active
        sel_uids = np.asarray(uids)[sel]
        self.snapshot = np.array([self.count.get(int(u), 0) for u in sel_uids])


def run(seeds, ms, n_agents):
    partner_counts = []
    age_causal, age_cin, age_cancer = [], [], []
    for s in seeds:
        az_part = LifetimeCasualPartners()
        az_dwell = hpv.age_causal_infection()
        sim = make_india_sim(s, ms=ms, analyzers=[az_part, az_dwell], n_agents=n_agents)
        sim.run()
        part = next(a for a in sim.analyzers.values() if hasattr(a, 'snapshot'))
        dwell = next(a for a in sim.analyzers.values() if hasattr(a, 'age_causal'))
        partner_counts.append(part.snapshot)
        age_causal += list(dwell.age_causal)
        age_cin += list(dwell.age_cin)
        age_cancer += list(dwell.age_cancer)
        print(f'seed {s}: n_females={len(part.snapshot)} n_cancers={len(dwell.age_cancer)}', flush=True)

    # --- Fig 5 partners ---
    counts = np.concatenate(partner_counts)
    hist, _ = np.histogram(counts, bins=BINS)
    prob = hist / hist.sum()
    partners_df = pd.DataFrame({
        'location': LOCATION, 'sex': 'f',
        'partner_count_bin': np.arange(21), 'probability': prob,
    })

    # --- Fig 5 dwelltime (same cross-filters as plot_fig56.run_sims) ---
    age_causal = np.array(age_causal)
    age_cin = np.array(age_cin)
    age_cancer = np.array(age_cancer)
    rows = []
    for ev, arr in [
        ('Causal HPV infection', age_causal[age_causal < 50]),
        ('CIN2+',                age_cin[age_causal < 65]),
        ('Cancer',               age_cancer[age_causal < 90]),
    ]:
        rows.append(pd.DataFrame({'Age': arr, 'Health event': ev, 'location': LOCATION.capitalize()}))
    dwelltime_df = pd.concat(rows)
    dwell_summary = (dwelltime_df.groupby(['location', 'Health event'])['Age']
                     .describe(percentiles=[0.25, 0.5, 0.75]).reset_index())
    return partners_df, dwell_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=f'results/v{hpv.__version__}_baseline')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--ms', type=int, default=100)
    parser.add_argument('--n-agents', type=int, default=20_000)
    args = parser.parse_args()

    assert hpv.__version__.startswith('3'), f'expected v3, got {hpv.__version__} from {hpv.__file__}'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    partners_df, dwell_summary = run(args.seeds, args.ms, args.n_agents)
    partners_df.to_csv(outdir / 'fig5_partners.csv', index=False)
    dwell_summary.to_csv(outdir / 'fig5_dwelltime_summary.csv', index=False)

    manifest_path = outdir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        'fig5': {'seeds': args.seeds, 'ms_agent_ratio': args.ms, 'n_agents': args.n_agents,
                 'hpvsim_version': hpv.__version__, 'date': date.today().isoformat()},
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f'\nSaved fig5 v3 baseline to {outdir}')
    print(dwell_summary.to_string(index=False))
