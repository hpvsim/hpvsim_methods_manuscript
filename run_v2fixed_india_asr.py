"""
Multi-seed ASR for the manuscript's india config on the FIXED-multiscale
v2.3.1 engine (hpvsim_v23_frozen @ fix-multiscale-cin-regate) — the correct
v2 comparison point (the committed fig6 baseline used the buggy v2.3 multiscale
that deflates cancer). Reuses plot_fig56.make_sim verbatim so the config is
identical. Reports the native asr_cancer_incidence (age-standardized, WHO
standard pop) at 2020, mean +/- range across seeds.
"""
import sys
# Pin the frozen v2.3.1 engine ahead of any editable install (editable-trap).
sys.path.insert(0, r'C:\Users\ryanhu\PycharmProjects\hpvsim_v23_frozen')
import numpy as np
import hpvsim as hpv

assert hpv.__version__ == '2.3.1', f'expected frozen 2.3.1, got {hpv.__version__} from {hpv.__file__}'
import plot_fig56 as p

SEEDS = [0, 1, 2, 3, 4]
asrs = []
for s in SEEDS:
    sim = p.make_sim('india', seed=s)  # debug=0 -> real config, ms_agent_ratio=100
    sim.run()
    asr_series = np.asarray(sim.results['asr_cancer_incidence'])
    asrs.append(float(asr_series[-1]))
    print(f'seed {s}: asr[2020]={asrs[-1]:.3f}', flush=True)
    if s == 0:
        # Lock the convention: is asr[-1] annual-steady or per-timestep-noisy?
        print('  last 8 asr timesteps:', np.round(asr_series[-8:], 3).tolist())

asrs = np.array(asrs)
print('\n=== v2.3.1-fixed india ASR cancer incidence 2020 (per 100k) ===')
print(f'mean={asrs.mean():.3f}  min={asrs.min():.3f}  max={asrs.max():.3f}  n={len(SEEDS)}')
print(f'(committed buggy-v2.3 baseline was 13.48)')
