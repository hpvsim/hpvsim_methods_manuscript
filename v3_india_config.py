"""Shared v3 build of the manuscript's india config (network + sim)."""
import numpy as np
import hpvsim as hpv
from hpvsim.data import country as C

DT = 0.25


def _to_annual_prob(p, dt):
    p = np.clip(p, 0, 1 - 1e-10)
    return 1 - (1 - p) ** (1 / dt)


def india_network(dt=DT):
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
            lp[k][row, :] = _to_annual_prob(lp[k][row, :].astype(float), dt)
    overrides = dict(
        layer_probs=lp,
        m_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.1)),
        f_partners=dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='neg_binomial', par1=2, par2=0.025)),
        m_cross_layer=_to_annual_prob(0.25, dt), f_cross_layer=_to_annual_prob(0.025, dt),
        debut=dict(f=dict(dist='lognormal', par1=15., par2=2.),
                   m=dict(dist='lognormal', par1=20., par2=2.)),
    )
    return hpv.SexualNetwork(**C._network_pars('india', overrides=overrides))


def make_india_sim(seed, ms=100, analyzers=None, start=1960, stop=2020, dt=DT, n_agents=20_000):
    gpars = {g: {'beta': 0.28} for g in ['hpv16', 'hpv18', 'hi5', 'ohr']}
    return hpv.Sim(location='india', genotypes=[16, 18, 'hi5', 'ohr'], genotype_pars=gpars,
                   start=start, stop=stop, dt=dt, n_agents=n_agents, ms_agent_ratio=ms,
                   networks=[india_network(dt)], analyzers=(analyzers or []),
                   rand_seed=seed, verbose=0)
