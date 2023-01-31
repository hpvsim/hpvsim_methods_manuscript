'''
Define custom analyzers for use case
'''

import numpy as np
import sciris as sc
import hpvsim as hpv
import pandas as pd

#%% Define analyzer for computing DALYs
class dalys(hpv.Analyzer):

    def __init__(self, max_age=84, cancer=None, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self.cancer = cancer if cancer else dict(dur=1, wt=0.16325) # From GBD 2017, calculated as 1 yr @ 0.288 (primary), 4 yrs @ 0.049 (maintenance), 0.5 yrs @ 0.451 (late), 0.5 yrs @ 0.54 (tertiary), scaled up to 12 years
        return

    def initialize(self, sim):
        super().initialize(sim)
        return

    def apply(self, sim):
        pass

    def finalize(self, sim):
        scale = sim['pop_scale']

        # Years of life lost
        dead = sim.people.dead_cancer
        years_left = np.maximum(0, self.max_age - sim.people.age)
        self.yll = (years_left*dead).sum()*scale
        self.deaths = dead.sum()*scale

        # Years lived with disability
        cancer = sc.objdict(self.cancer)
        n_cancer = (sim.people.cancerous).sum()*scale
        self.n_cancer = n_cancer
        self.yld = n_cancer*cancer.dur*cancer.wt
        self.dalys = self.yll + self.yld
        return
