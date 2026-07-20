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


# Per-genotype transmissibility. Kept at v2's 0.28 (identical config to the
# patched-v2.3.1 reference). A light 1-parameter beta re-fit was attempted to
# close the Fig 6 baseline ASR gap (v3 ~67 vs patched-v2.3.1 ~53, +26%) but is
# not achievable via beta: the India epidemic is at endemic equilibrium
# (female HPV prevalence saturates at ~13% by ~1980 and is flat through 2020),
# so 2020 cancer incidence is beta-insensitive. A sweep over beta in [0.16, 0.50]
# leaves the 2020 ASR in a flat/noisy 67-80 band with beta=0.28 at its MINIMUM
# (~67); raising OR lowering beta only increases ASR. The residual +26% is a
# cancer-pathway difference (v3's unbiased-multiscale cancer runs ~23% hotter
# than even patched v2.3.1 at ms_agent_ratio=100), not a transmission
# difference, so it requires a cancer-side recalibration, not a beta change.
BETA = 0.28


def make_india_sim(seed, ms=100, analyzers=None, interventions=None,
                   start=1960, stop=2020, dt=DT, n_agents=20_000, beta=BETA):
    gpars = {g: {'beta': beta} for g in ['hpv16', 'hpv18', 'hi5', 'ohr']}
    return hpv.Sim(location='india', genotypes=[16, 18, 'hi5', 'ohr'], genotype_pars=gpars,
                   start=start, stop=stop, dt=dt, n_agents=n_agents, ms_agent_ratio=ms,
                   networks=[india_network(dt)], analyzers=(analyzers or []),
                   interventions=(interventions or []),
                   rand_seed=seed, verbose=0)
