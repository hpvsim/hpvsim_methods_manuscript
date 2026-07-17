"""
Multi-seed ASR for the manuscript's india config on hpvsim v3, no recalibration.

Computes the SAME age-standardized cancer incidence as v2.3.1's native
asr_cancer_incidence: AgeResults 'cancer_incidence' (new cancers at the
year-end tick / at-risk alive females, per 100k, by age) dotted with the WHO
World Standard Population weights over the identical age bins. This makes
ASR_v3 directly comparable to the fixed-multiscale v2.3.1 number.
"""
import numpy as np
import sciris as sc
import hpvsim as hpv
from hpvsim.data import country as C

DT = 0.25
SEEDS = [0, 1, 2, 3, 4]

# WHO standard population — identical to v2 hpvsim parameters.py.
AGE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 100])
STD_WEIGHTS = np.array([.12, .10, .09, .09, .08, .08, .06, .06, .06, .06,
                        .05, .04, .04, .03, .02, .01, 0.005, 0.005, 0])[:-1]  # 18 bins


def _to_annual_prob(p, dt):
    p = np.clip(p, 0, 1 - 1e-10)
    return 1 - (1 - p) ** (1 / dt)


def _layer_probs_to_annual(layer_probs, dt):
    out = {}
    for lkey, lp in layer_probs.items():
        lp_new = lp.copy().astype(float)
        for row in (1, 2):
            lp_new[row, :] = _to_annual_prob(lp_new[row, :], dt)
        out[lkey] = lp_new
    return out


def _india_network():
    layer_probs = dict(
        m=np.array([
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            [0, 0, 0.05, 0.25, 0.60, 0.80, 0.95, 0.80, 0.80, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],
            [0, 0, 0.01, 0.05, 0.10, 0.70, 0.90, 0.90, 0.90, 0.90, 0.80, 0.60, 0.60, 0.60, 0.60, 0.60]]),
        c=np.array([
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
            [0, 0, 0.10, 0.50, 0.60, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.01, 0.01],
            [0, 0, 0.10, 0.20, 0.25, 0.35, 0.40, 0.70, 0.90, 0.90, 0.95, 0.95, 0.70, 0.30, 0.10, 0.10]]),
    )
    m_partners = dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='poisson1', par1=0.1))
    f_partners = dict(m=dict(dist='poisson1', par1=0.01), c=dict(dist='neg_binomial', par1=2, par2=0.025))
    layer_probs = _layer_probs_to_annual(layer_probs, DT)
    overrides = dict(
        layer_probs=layer_probs, m_partners=m_partners, f_partners=f_partners,
        m_cross_layer=_to_annual_prob(0.25, DT), f_cross_layer=_to_annual_prob(0.025, DT),
        debut=dict(f=dict(dist='lognormal', par1=15., par2=2.),
                   m=dict(dist='lognormal', par1=20., par2=2.)),
    )
    return hpv.SexualNetwork(**C._network_pars('india', overrides=overrides))


def make_v3_sim(seed):
    gpars = {g: {'beta': 0.28} for g in ['hpv16', 'hpv18', 'hi5', 'ohr']}
    az = hpv.AgeResults(result_args=sc.objdict(
        cancer_incidence=sc.objdict(years=[2020], edges=AGE_EDGES)))
    return hpv.Sim(
        location='india', genotypes=[16, 18, 'hi5', 'ohr'], genotype_pars=gpars,
        start=1960, stop=2020, dt=DT, n_agents=20_000, ms_agent_ratio=100,
        networks=[_india_network()], analyzers=[az], rand_seed=seed, verbose=0,
    )


asrs = []
for s in SEEDS:
    sim = make_v3_sim(s)
    sim.run()
    inc = np.asarray(sim.analyzers['ageresults'].outputs['cancer_incidence'][2020.0])
    asr = float(np.dot(inc, STD_WEIGHTS))
    asrs.append(asr)
    print(f'seed {s}: asr[2020]={asr:.3f}', flush=True)

asrs = np.array(asrs)
print('\n=== v3 india ASR cancer incidence 2020 (per 100k), no recalibration ===')
print(f'mean={asrs.mean():.3f}  min={asrs.min():.3f}  max={asrs.max():.3f}  n={len(SEEDS)}')
print('v2.3.1-fixed (same config, same formula): mean=71.69 [67.50, 78.44]')
