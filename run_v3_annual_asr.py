"""v3 india ASR with an ANNUAL cancer numerator (matches v2's convention).

v2 reports annually: ASR numerator = full-year new cancers by age, denominator
= alive non-cancerous females by age at the year-end tick, x1e5, dotted with
WHO weights. v3's built-in AgeResults captures only the final dt sub-step at
dt<1, undercounting ~4x. This custom analyzer accumulates new-cancer events by
age across all sub-steps of each calendar year, then applies the identical
denominator/weights, so ASR_v3 is directly comparable to v2's asr_cancer_incidence.
"""
import numpy as np
import sciris as sc
import starsim as ss
import hpvsim as hpv
from hpvsim.data import country as C
from hpvsim.hpv import HPV

DT = 0.25
SEEDS = [0, 1, 2, 3, 4]
YEARS = [2016, 2017, 2018, 2019, 2020]
AGE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 100])
STD_WEIGHTS = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06,
                        .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0])[:-1]


class AnnualCancerASR(ss.Analyzer):
    """Annual age-standardized cancer incidence, v2-faithful, at dt=0.25."""
    def __init__(self, years, edges, weights, **kw):
        super().__init__(**kw)
        self.years = [float(y) for y in years]
        self.edges = np.asarray(edges, float)
        self.weights = np.asarray(weights, float)
        self.nbins = len(self.edges) - 1
        self.num = {y: np.zeros(self.nbins) for y in self.years}   # annual cancers by age
        self.denom = {y: None for y in self.years}                 # year-end at-risk females by age
        self.asr = {}
        self.hpv_modules = None
        self._year_end_ti = None

    def init_pre(self, sim):
        self.hpv_modules = [d for d in sim.diseases.values() if isinstance(d, HPV)]
        super().init_pre(sim)
        tvy = sim.timevec.years
        self._year_end_ti = {}
        for y in self.years:
            ticks = np.where((tvy >= y) & (tvy < y + 1))[0]
            if len(ticks):
                self._year_end_ti[int(ticks[-1])] = y

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
        # Accumulate this sub-step's new cancers into the calendar year.
        if yr in self.num:
            mask = new_c & alive
            self.num[yr] += np.histogram(ages[mask], self.edges,
                                         weights=(w[mask] if w is not None else None))[0]
        # At each requested year's final tick, snapshot the at-risk denominator.
        if ti in self._year_end_ti:
            y = self._year_end_ti[ti]
            at_risk = alive & female & ~canc
            self.denom[y] = np.histogram(ages[at_risk], self.edges,
                                         weights=(w[at_risk] if w is not None else None))[0]

    def finalize(self):
        super().finalize()
        for y in self.years:
            d = self.denom[y]
            if d is None:
                continue
            asi = np.divide(self.num[y], d, out=np.zeros(self.nbins), where=d > 0) * 1e5
            self.asr[y] = float(np.dot(asi, self.weights))


def _to_annual_prob(p, dt):
    p = np.clip(p, 0, 1 - 1e-10)
    return 1 - (1 - p) ** (1 / dt)


def _india_network():
    lp = dict(
        m=np.array([
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            [0, 0, 0.05, 0.25, 0.60, 0.80, 0.95, 0.80, 0.80, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],
            [0, 0, 0.01, 0.05, 0.10, 0.70, 0.90, 0.90, 0.90, 0.90, 0.80, 0.60, 0.60, 0.60, 0.60, 0.60]]),
        c=np.array([
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            [0, 0, 0.10, 0.50, 0.60, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.01, 0.01],
            [0, 0, 0.10, 0.20, 0.25, 0.35, 0.40, 0.70, 0.90, 0.90, 0.95, 0.95, 0.70, 0.30, 0.10, 0.10]]),
    )
    for k in lp:
        for row in (1, 2):
            lp[k][row, :] = _to_annual_prob(lp[k][row, :].astype(float), DT)
    overrides = dict(
        layer_probs=lp,
        m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.1)),
        f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='neg_binomial', par1=2, par2=0.025)),
        m_cross_layer=_to_annual_prob(0.25, DT), f_cross_layer=_to_annual_prob(0.025, DT),
        debut=dict(f=dict(dist='lognormal', par1=15., par2=2.),
                   m=dict(dist='lognormal', par1=20., par2=2.)),
    )
    return hpv.SexualNetwork(**C._network_pars('india', overrides=overrides))


def make(seed, ms):
    gpars = {g: {'beta': 0.28} for g in ['hpv16', 'hpv18', 'hi5', 'ohr']}
    az = AnnualCancerASR(years=YEARS, edges=AGE_EDGES, weights=STD_WEIGHTS)
    return hpv.Sim(location='india', genotypes=[16, 18, 'hi5', 'ohr'], genotype_pars=gpars,
                   start=1960, stop=2020, dt=DT, n_agents=20_000, ms_agent_ratio=ms,
                   networks=[_india_network()], analyzers=[az], rand_seed=seed, verbose=0)


for ms in [1, 100]:
    vals = []
    for s in SEEDS:
        sim = make(s, ms)
        sim.run()
        az = next(a for a in sim.analyzers.values() if hasattr(a, 'asr'))
        vals.append(np.mean([az.asr[y] for y in YEARS]))
    vals = np.array(vals)
    print(f'v3 ANNUAL ASR  ms={ms:<3d} mean-5yr = {vals.mean():.2f} [{vals.min():.2f}, {vals.max():.2f}]', flush=True)
print('v2 (annual, same config): ms=1 -> 60.67 | ms=100 -> 71.01')
