"""Within-engine ms=1 vs ms=100 diagnostic for a v2 engine (path via argv[1]).

Deflation bug <=> ASR(ms=100) << ASR(ms=1) within the same engine. Uses the
native asr_cancer_incidence averaged over the last 5 years to cut single-tick
noise (critical at ms=1, where cancers are sparse).
"""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import hpvsim as hpv
import plot_fig56 as p

SEEDS = [0, 1, 2, 3, 4]
print(f'engine hpvsim {hpv.__version__} from {hpv.__file__[:55]}', flush=True)
for ms in [1, 100]:
    vals = []
    for s in SEEDS:
        sim = p.make_sim('india', seed=s)
        sim['ms_agent_ratio'] = ms
        sim.run()
        asr = np.asarray(sim.results['asr_cancer_incidence'])
        vals.append(float(np.mean(asr[-20:])))  # last 5 yr (dt=0.25 -> 20 ticks)
    vals = np.array(vals)
    print(f'  ms={ms:<3d} mean-5yr ASR = {vals.mean():.2f} [{vals.min():.2f}, {vals.max():.2f}]', flush=True)
