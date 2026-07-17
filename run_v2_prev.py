"""HPV prevalence at ms=1 on a v2 engine (path via argv[1]), india config.
Metric = currently-infected prevalence = n_infected/n_alive, to match v3."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import hpvsim as hpv
import plot_fig56 as p

SEEDS = [0, 1, 2]
YRS = [1990, 2000, 2010, 2020]
h16, tot = [], []
for s in SEEDS:
    sim = p.make_sim('india', seed=s)
    sim['ms_agent_ratio'] = 1
    sim.run()
    year = np.asarray(sim.results['year'])
    na = np.asarray(sim.results['n_alive'])
    ni16 = np.asarray(sim.results['n_infected_by_genotype'])[0]
    ni_all = np.asarray(sim.results['n_infected'])
    idx = {y: int(np.argmin(np.abs(year - y))) for y in YRS}
    h16.append([ni16[idx[y]] / na[idx[y]] for y in YRS])
    tot.append([ni_all[idx[y]] / na[idx[y]] for y in YRS])
h16, tot = np.array(h16), np.array(tot)
print(f'{hpv.__version__} ms=1 india, years {YRS}')
print('  hpv16 prevalence:', np.round(h16.mean(0), 4).tolist())
print('  any-HPV prevalence:', np.round(tot.mean(0), 4).tolist())
