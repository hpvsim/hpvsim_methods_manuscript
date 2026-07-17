"""Within-engine ms=1 vs ms=100 diagnostic for hpvsim v3 (grow multiscale).

Same india config + same ASR formula (AgeResults cancer_incidence . WHO
weights) as the v2 diagnostic, averaged over the last 5 years to cut noise.
"""
import numpy as np
import sciris as sc
import hpvsim as hpv
from hpvsim.data import country as C

DT = 0.25
SEEDS = [0, 1, 2, 3, 4]
YEARS = [2016, 2017, 2018, 2019, 2020]
AGE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 100])
STD_WEIGHTS = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06,
                        .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0])[:-1]


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
    az = hpv.AgeResults(result_args=sc.objdict(
        cancer_incidence=sc.objdict(years=YEARS, edges=AGE_EDGES)))
    return hpv.Sim(location='india', genotypes=[16, 18, 'hi5', 'ohr'], genotype_pars=gpars,
                   start=1960, stop=2020, dt=DT, n_agents=20_000, ms_agent_ratio=ms,
                   networks=[_india_network()], analyzers=[az], rand_seed=seed, verbose=0)


print('engine hpvsim', hpv.__version__, '(v3 grow multiscale)', flush=True)
for ms in [1, 100]:
    vals = []
    for s in SEEDS:
        sim = make(s, ms)
        sim.run()
        out = sim.analyzers['ageresults'].outputs['cancer_incidence']
        yr_asr = [float(np.dot(out[float(y)], STD_WEIGHTS)) for y in YEARS]
        vals.append(np.mean(yr_asr))
    vals = np.array(vals)
    print(f'  ms={ms:<3d} mean-5yr ASR = {vals.mean():.2f} [{vals.min():.2f}, {vals.max():.2f}]', flush=True)
