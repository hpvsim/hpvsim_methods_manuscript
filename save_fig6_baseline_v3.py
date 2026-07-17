"""
Freeze Fig 6 (ASR cancer incidence by screening coverage) on hpvsim v3.

Migrates plot_fig56.make_screening / run_screening to the v3 intervention API
and writes results/<ver>_baseline/fig6_asr.csv in the compare_baselines schema
(scenario, asr_cancer_incidence_2020, _low_2020, _high_2020).

v3 intervention API notes vs v2 plot_fig56:
  - hpv.routine_screening / hpv.treat_num keep the same names but are Starsim
    modules. The interventions dict is keyed by ``name`` (defaults to the class
    name), NOT ``label`` -- so cross-references must pass an explicit ``name=``.
  - A treatment's ``name`` must differ from its product name ('ablation'),
    else people.add_module raises "Module ablation already added".
  - There is no ``sim.get_intervention``; use ``sim.interventions[name]``.
  - v2's re-screen eligibility used ``sim.people.date_screened``; that per-agent
    state now lives on the screening module as ``ti_screened`` (integer ti,
    NaN = never). We reconstruct "never screened OR screened >5y ago".
  - v2's global ``beta`` is per-genotype in v3 (set to 0.28 in make_india_sim).

ASR: v3's built-in AgeResults captures only the final dt sub-step at dt<1
(undercounts ~4x). This uses the same AnnualCancerASR analyzer as
run_v3_annual_asr.py -- full-year new-cancer numerator by age / year-end
at-risk female denominator x1e5, dotted with WHO weights -- so ASR_v3 at 2020
is directly comparable to v2's native asr_cancer_incidence.

Run with the v3 venv:
  .venv/Scripts/python.exe save_fig6_baseline_v3.py --outdir results/v3.0.0_baseline
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import starsim as ss
import hpvsim as hpv
from hpvsim.hpv import HPV

from v3_india_config import make_india_sim, DT

AGE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 100])
STD_WEIGHTS = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06,
                        .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0])[:-1]


class AnnualCancerASR(ss.Analyzer):
    """Annual (v2-faithful) age-standardized cancer incidence at 2020, dt<1 safe."""
    def __init__(self, year=2020, edges=AGE_EDGES, weights=STD_WEIGHTS, **kw):
        super().__init__(**kw)
        self.year = float(year)
        self.edges = np.asarray(edges, float)
        self.weights = np.asarray(weights, float)
        self.nbins = len(self.edges) - 1
        self.num = np.zeros(self.nbins)
        self.denom = None
        self.asr = None
        self.hpv_modules = None
        self._year_end_ti = None

    def init_pre(self, sim):
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        tvy = sim.timevec.years if hasattr(sim.timevec, 'years') else np.array(
            [t.year if hasattr(t, 'year') else t for t in sim.timevec])
        ticks = np.where((tvy >= self.year) & (tvy < self.year + 1))[0]
        self._year_end_ti = int(ticks[-1]) if len(ticks) else None

    def step(self):
        sim = self.sim
        ti = sim.ti
        yr = float(int(sim.t.now('year')))
        people = sim.people
        ages = people.age.values
        w = people.scale.values if getattr(people, 'scale', None) is not None else None
        alive = people.alive.values
        female = people.female.values
        new_c = np.zeros_like(alive)
        canc = np.zeros_like(alive)
        for m in self.hpv_modules:
            new_c |= (m.ti_cancerous.values == ti) & m.cancerous.values
            canc |= m.cancerous.values
        if yr == self.year:
            mask = new_c & alive
            self.num += np.histogram(ages[mask], self.edges,
                                     weights=(w[mask] if w is not None else None))[0]
        if ti == self._year_end_ti:
            at_risk = alive & female & ~canc
            self.denom = np.histogram(ages[at_risk], self.edges,
                                      weights=(w[at_risk] if w is not None else None))[0]

    def finalize(self):
        super().finalize()
        d = self.denom
        if d is None:
            return
        asi = np.divide(self.num, d, out=np.zeros(self.nbins), where=d > 0) * 1e5
        self.asr = float(np.dot(asi, self.weights))


def make_screening(end_prob, ablate_prob=0.7, dt=DT):
    """v3 port of plot_fig56.make_screening for a single coverage level."""
    years = np.linspace(2000, 2020, 21)
    primary_screen_prob = np.linspace(0, end_prob, 21)

    def screen_eligible(sim):
        scr = sim.interventions['via_primary']
        vals = scr.ti_screened.values          # integer ti, NaN = never
        never = np.isnan(vals)
        due = ~never & ((sim.ti - vals) * dt > 5)  # re-screen after 5 years
        return sim.people.auids[never | due]

    via_primary = hpv.routine_screening(
        product='via', prob=primary_screen_prob, years=years,
        eligibility=screen_eligible, name='via_primary', label='via primary',
    )
    via_positive = lambda sim: sim.interventions['via_primary'].outcomes['positive']
    ablation = hpv.treat_num(
        prob=ablate_prob, product='ablation', eligibility=via_positive,
        name='ablation_rx', label='ablation',
    )
    return [via_primary, ablation]


def run(seeds, end_probs, ms, n_agents):
    rows = []
    for end_prob in end_probs:
        per_seed = []
        for s in seeds:
            interventions = make_screening(end_prob) if end_prob > 0 else []
            az = AnnualCancerASR(year=2020)
            # stop=2021 so calendar-year 2020 has all 4 dt-substeps (a sim that
            # stops AT 2020 leaves 2020 with only its terminal tick, undercounting
            # the annual cancer numerator ~4x at dt=0.25). v2 reports an annual
            # rate at 2020, so this makes v3's 2020 directly comparable.
            sim = make_india_sim(s, ms=ms, analyzers=[az], interventions=interventions,
                                 n_agents=n_agents, stop=2021)
            sim.run()
            a = next(x for x in sim.analyzers.values() if hasattr(x, 'asr'))
            per_seed.append(a.asr)
            print(f'end_prob={end_prob} seed={s}: asr2020={a.asr:.2f}', flush=True)
        per_seed = np.array(per_seed)
        rows.append({
            'scenario': f'Screening {end_prob}',
            'asr_cancer_incidence_2020': float(per_seed.mean()),
            'asr_cancer_incidence_low_2020': float(per_seed.min()),
            'asr_cancer_incidence_high_2020': float(per_seed.max()),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=f'results/v{hpv.__version__}_baseline')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--end-probs', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    parser.add_argument('--ms', type=int, default=100)
    parser.add_argument('--n-agents', type=int, default=20_000)
    args = parser.parse_args()

    assert hpv.__version__.startswith('3'), f'expected v3, got {hpv.__version__} from {hpv.__file__}'
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = run(args.seeds, args.end_probs, args.ms, args.n_agents)
    df.to_csv(outdir / 'fig6_asr.csv', index=False)

    manifest_path = outdir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        'fig6': {'seeds': args.seeds, 'end_probs': args.end_probs, 'ms_agent_ratio': args.ms,
                 'n_agents': args.n_agents, 'hpvsim_version': hpv.__version__,
                 'date': date.today().isoformat()},
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f'\nSaved fig6 v3 baseline to {outdir}')
    print(df.to_string(index=False))
