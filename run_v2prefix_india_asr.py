"""
Multi-seed ASR for the manuscript's india config on the PRE-bugfix v2.3.0
engine (hpvsim_v23_frozen @ 44872efd, before the multiscale CIN-regate fix) —
the deflated-multiscale engine the committed fig6 baseline was made on.
Same harness as run_v2fixed_india_asr.py for a clean 3-way comparison.
"""
import sys
sys.path.insert(0, r'C:\Users\ryanhu\PycharmProjects\hpvsim_v23_prefix')
import numpy as np
import hpvsim as hpv

assert hpv.__version__ == '2.3.0', f'expected pre-fix 2.3.0, got {hpv.__version__}'
import plot_fig56 as p

SEEDS = [0, 1, 2, 3, 4]
asrs = []
for s in SEEDS:
    sim = p.make_sim('india', seed=s)
    sim.run()
    asrs.append(float(np.asarray(sim.results['asr_cancer_incidence'])[-1]))
    print(f'seed {s}: asr[2020]={asrs[-1]:.3f}', flush=True)

asrs = np.array(asrs)
print('\n=== v2.3.0 PRE-fix india ASR cancer incidence 2020 (per 100k) ===')
print(f'mean={asrs.mean():.3f}  min={asrs.min():.3f}  max={asrs.max():.3f}  n={len(SEEDS)}')
print('committed buggy-v2.3 baseline: 13.48 | v3: 17.87 | v2.3.1-fixed: 71.69')
