"""Fig 5 partners on v3: lifetime casual-partner distribution among active
base (level0) females at 2020, vs the frozen v2.3 India baseline.

v2's n_rships is a cumulative per-agent partnership counter. v3 keeps only a
current edge table (dissolved edges are dropped), so this analyzer accumulates
casual-layer formations per female by counting edges whose start_ti == the
current tick each step. Fine multiscale agents don't partner (excluded from the
network), so only base agents contribute — matching v2's level0 filter.
"""
import numpy as np
import pandas as pd
import starsim as ss
import hpvsim as hpv
from v3_india_config import make_india_sim

BINS = np.concatenate([np.arange(21), [100]])  # 0,1,...,20,100  (bin 20 = 20+)


class LifetimeCasualPartners(ss.Analyzer):
    def init_pre(self, sim):
        self.net = next(n for n in sim.networks.values() if isinstance(n, hpv.SexualNetwork))
        super().init_pre(sim)
        self.casual = self.net._layer_idx['c']
        self.count = {}          # uid -> cumulative casual partnerships
        self.snapshot = None     # per-active-female counts at the target year

    def step(self):
        e = self.net.edges
        start_ti = np.asarray(e.start_ti)
        new = (start_ti == self.sim.ti) & (np.asarray(e.layer_id) == self.casual)
        for u in np.asarray(e.p1)[new]:      # p1 == female partner
            self.count[int(u)] = self.count.get(int(u), 0) + 1

    def finalize(self):
        super().finalize()
        people = self.sim.people
        uids = people.auids                       # alive agent uids
        female = people.female.values             # all .values are auids-aligned
        fine = (people.fine.values if 'fine' in people.states
                else np.zeros(len(uids), dtype=bool))
        active = people.age.values >= self.net.debut.values
        sel = female & (~fine) & active           # boolean over auids
        sel_uids = np.asarray(uids)[sel]
        self.snapshot = np.array([self.count.get(int(u), 0) for u in sel_uids])


counts = []
for s in [0, 1, 2]:
    az = LifetimeCasualPartners()
    sim = make_india_sim(s, ms=100, analyzers=[az])
    sim.run()
    a = next(x for x in sim.analyzers.values() if hasattr(x, 'snapshot'))
    counts.append(a.snapshot)
counts = np.concatenate(counts)

v3_hist, _ = np.histogram(counts, bins=BINS)
v3_prob = v3_hist / v3_hist.sum()

v2 = pd.read_csv('results/v2.3.0_baseline/fig5_partners.csv')
v2 = v2[(v2.location == 'india') & (v2.sex == 'f')].sort_values('partner_count_bin')
v2_prob = v2['probability'].values

print('=== v3 india Fig5 lifetime casual partners (females) vs v2.3 baseline ===')
print(f'{"bin":>4s} {"v3 prob":>9s} {"v2 prob":>9s}')
for b in range(len(v3_prob)):
    label = f'{b}+' if b == 20 else str(b)
    print(f'{label:>4s} {v3_prob[b]:9.4f} {v2_prob[b]:9.4f}')
print(f'\nv3 mean partners = {counts.mean():.3f} | frac with 0 = {(counts==0).mean():.3f}')
print(f'v2 P(0) = {v2_prob[0]:.3f}')
tvd = 0.5 * np.abs(v3_prob - v2_prob).sum()
print(f'total variation distance (v3 vs v2) = {tvd:.3f}')
