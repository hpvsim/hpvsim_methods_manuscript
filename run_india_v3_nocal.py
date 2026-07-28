"""
Run the manuscript's india config on hpvsim v3 WITHOUT recalibrating — using
the v2.3-calibrated network + beta as-is — to gauge how far v3's cancer level
drifts from the frozen v2.3 baseline (fig6 'Screening 0.0' ASR = 13.48/100k).

v3 has no global `beta` (per-genotype) and no ASR result, so beta=0.28 is set
per genotype and we report crude cancer incidence + cumulative cancers.
Run with the v3 venv.
"""
import numpy as np
import starsim as ss
import hpvsim as hpv
from hpvsim.data import country as C

DT = 0.25


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


# --- india network params, verbatim from plot_fig56.make_network('india') ---
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
m_cross_layer, f_cross_layer = 0.25, 0.025
debut = dict(f=dict(dist='lognormal', par1=15., par2=2.),
             m=dict(dist='lognormal', par1=20., par2=2.))

# v2.3 treats these as annual; convert as make_sim does.
layer_probs = _layer_probs_to_annual(layer_probs, DT)
m_cross_layer = _to_annual_prob(m_cross_layer, DT)
f_cross_layer = _to_annual_prob(f_cross_layer, DT)

overrides = dict(layer_probs=layer_probs, m_partners=m_partners, f_partners=f_partners,
                 m_cross_layer=m_cross_layer, f_cross_layer=f_cross_layer, debut=debut)
netpars = C._network_pars('india', overrides=overrides)
net = hpv.SexualNetwork(**netpars)

# v2 global beta=0.28 -> set per genotype (v3 has no sim-level beta; rel_beta stacks).
gpars = {g: {'beta': 0.28} for g in ['hpv16', 'hpv18', 'hi5', 'ohr']}

sim = hpv.Sim(
    location='india', genotypes=[16, 18, 'hi5', 'ohr'], genotype_pars=gpars,
    start=1960, stop=2020, dt=DT, n_agents=20_000, ms_agent_ratio=100,
    networks=[net], rand_seed=0, verbose=0.1,
)
sim.run()

res = sim.results
tv = np.asarray(res.timevec)
yrs = np.array([t.year if hasattr(t, 'year') else t for t in tv])
i2020 = np.where(yrs == 2020)[0]

# v3.0 release renamed the pooled results group hpvtotal -> all_hpv.
_pooled = getattr(res, 'all_hpv', None)
if _pooled is None:
    _pooled = res.hpvtotal
new_cancers = np.asarray(_pooled.new_cancers)
cum_cancers = float(np.asarray(_pooled.cum_cancers)[-1])
n_alive = np.asarray(res.n_alive)
n_female = np.asarray(res.n_female)

cancers_2020 = float(new_cancers[i2020].sum())
females_2020 = float(n_female[i2020].mean())
crude_inc_2020 = cancers_2020 / females_2020 * 1e5

print('=== v3 india, no recalibration, ms_agent_ratio=100 ===')
print(f'cumulative cancers (scaled): {cum_cancers:,.0f}')
print(f'new cancers in 2020 (scaled): {cancers_2020:,.1f}')
print(f'mean female pop 2020 (scaled): {females_2020:,.0f}')
print(f'CRUDE cancer incidence 2020: {crude_inc_2020:,.1f} per 100k females')
print('v2.3 baseline fig6 "Screening 0.0" ASR 2020: 13.48 per 100k (ASR, not crude)')
