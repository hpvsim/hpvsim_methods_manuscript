"""Fig 5 dwelltime on v3: age at causal HPV infection / CIN2+ / cancer.

Uses v3's age_causal_infection analyzer (records age_causal/age_cin/age_cancer
per cancer, scale-weighted, engine-correct at any ms_agent_ratio). Applies the
same age_causal cross-filters as plot_fig56.run_sims and compares the summary
(count/mean/median/IQR) to the frozen v2.3 India baseline.
"""
import numpy as np
import pandas as pd
import hpvsim as hpv
from v3_india_config import make_india_sim

SEEDS = [0, 1, 2]

age_causal, age_cin, age_cancer = [], [], []
for s in SEEDS:
    az = hpv.age_causal_infection()
    sim = make_india_sim(s, ms=100, analyzers=[az])
    sim.run()
    a = next(x for x in sim.analyzers.values() if hasattr(x, 'age_causal'))
    age_causal += list(a.age_causal)
    age_cin += list(a.age_cin)
    age_cancer += list(a.age_cancer)

age_causal = np.array(age_causal)
age_cin = np.array(age_cin)
age_cancer = np.array(age_cancer)

# Same cross-filters on age_causal as plot_fig56.run_sims.
events = {
    'Causal HPV infection': age_causal[age_causal < 50],
    'CIN2+':                age_cin[age_causal < 65],
    'Cancer':               age_cancer[age_causal < 90],
}

# Frozen v2.3 India baseline (results/v2.3.0_baseline/fig5_dwelltime_summary.csv).
v2 = {
    'Causal HPV infection': dict(mean=32.33, p50=32.5, p25=24.25, p75=40.0),
    'CIN2+':                dict(mean=39.55, p50=38.99, p25=30.17, p75=48.13),
    'Cancer':               dict(mean=52.24, p50=52.0, p25=42.0, p75=61.81),
}

print('=== v3 india Fig5 dwelltime (ms=100, no recalibration) vs v2.3 baseline ===')
print(f'{"event":22s} {"n":>7s} {"v3 mean":>8s} {"v2 mean":>8s} {"v3 med":>7s} {"v2 med":>7s} {"v3 IQR":>13s} {"v2 IQR":>13s}')
for ev, arr in events.items():
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    b = v2[ev]
    print(f'{ev:22s} {len(arr):7d} {arr.mean():8.1f} {b["mean"]:8.1f} {q2:7.1f} {b["p50"]:7.1f} '
          f'{q1:5.1f}-{q3:<7.1f} {b["p25"]:5.1f}-{b["p75"]:<7.1f}')
