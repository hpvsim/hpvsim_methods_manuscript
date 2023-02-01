"""
This script produces figure 5 of the HPVsim methods paper, showing the natural history
"""


import pylab as pl
import pandas as pd
import numpy as np
import sciris as sc
import seaborn as sns
import hpvsim as hpv
import hpvsim.utils as hpu

class prop_exposed(hpv.Analyzer):
    ''' Store proportion of agents exposed '''
    def __init__(self, years=None):
        super().__init__()
        self.years = years
        self.timepoints = []

    def initialize(self, sim):
        super().initialize(sim)
        for y in self.years:
            try:    tp = sc.findinds(sim.yearvec, y)[0]
            except: raise ValueError('Year not found')
            self.timepoints.append(tp)
        self.prop_exposed = dict()
        for y in self.years: self.prop_exposed[y] = []

    def apply(self, sim):
        if sim.t in self.timepoints:
            tpi = self.timepoints.index(sim.t)
            year = self.years[tpi]
            prop_exposed = sc.autolist()
            for a in range(10,25):
                ainds = hpv.true((sim.people.age >= a) & (sim.people.age < a+1) & (sim.people.sex==0))
                prop_exposed += sc.safedivide(sum((~np.isnan(sim.people.date_exposed[:, ainds])).any(axis=0)), len(ainds))
            self.prop_exposed[year] = np.array(prop_exposed)
        return

    @staticmethod
    def reduce(analyzers, quantiles=None):
        if quantiles is None: quantiles = {'low': 0.1, 'high': 0.9}
        base_az = analyzers[0]
        reduced_az = sc.dcp(base_az)
        reduced_az.prop_exposed = dict()
        for year in base_az.years:
            reduced_az.prop_exposed[year] = sc.objdict()
            allres = np.empty([len(analyzers), len(base_az.prop_exposed[year])])
            for ai,az in enumerate(analyzers):
                allres[ai,:] = az.prop_exposed[year][:]
            reduced_az.prop_exposed[year].best  = np.quantile(allres, 0.5, axis=0)
            reduced_az.prop_exposed[year].low   = np.quantile(allres, quantiles['low'], axis=0)
            reduced_az.prop_exposed[year].high  = np.quantile(allres, quantiles['high'], axis=0)

        return reduced_az

class age_causal_infection_by_genotype(hpv.Analyzer):
    '''
    Determine the age at which people with cervical cancer were causally infected and
    time spent between infection and cancer.
    '''

    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.ng = sim['n_genotypes']
        self.genotypes = sim['genotypes']
        self.genotype_map = sim['genotype_map']
        self.age_causal = dict()
        self.age_cancer = dict()
        self.dwelltime = dict()
        for genotype in self.genotypes:
            self.age_causal[genotype] = []
            self.age_cancer[genotype] = []
            self.dwelltime[genotype] = dict()
            for state in ['hpv', 'cin1', 'cin2', 'cin3', 'total']:
                self.dwelltime[genotype][state] = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                for gtype in np.unique(cancer_genotypes):
                    cancer_inds_gtype = cancer_inds[hpu.true(cancer_genotypes == gtype)]
                    current_age = sim.people.age[cancer_inds_gtype]
                    date_exposed = sim.people.date_exposed[gtype, cancer_inds_gtype]
                    date_cin1 = sim.people.date_cin1[gtype, cancer_inds_gtype]
                    date_cin2 = sim.people.date_cin2[gtype, cancer_inds_gtype]
                    date_cin3 = sim.people.date_cin3[gtype, cancer_inds_gtype]
                    hpv_time = (date_cin1 - date_exposed) * sim['dt']
                    cin1_time = (date_cin2 - date_cin1) * sim['dt']
                    cin2_time = (date_cin3 - date_cin2) * sim['dt']
                    cin3_time = (sim.t - date_cin3) * sim['dt']
                    total_time = (sim.t - date_exposed) * sim['dt']
                    self.age_causal[self.genotype_map[gtype]] += (current_age - total_time).tolist()
                    self.age_cancer[self.genotype_map[gtype]] += current_age.tolist()
                    self.dwelltime[self.genotype_map[gtype]]['hpv'] += hpv_time.tolist()
                    self.dwelltime[self.genotype_map[gtype]]['cin1'] += cin1_time.tolist()
                    self.dwelltime[self.genotype_map[gtype]]['cin2'] += cin2_time.tolist()
                    self.dwelltime[self.genotype_map[gtype]]['cin3'] += cin3_time.tolist()
                    self.dwelltime[self.genotype_map[gtype]]['total'] += total_time.tolist()
        return

    def finalize(self, sim=None):
        ''' Convert things to arrays '''


#%% Plotting function
def plot_fig5(calib_pars=None):
    # Group genotypes
    genotypes = ['hpv16', 'hpv18', 'hrhpv']

    dt = age_causal_infection_by_genotype(start_year=2000)
    exposure_years = [2020]
    pe = prop_exposed(years=exposure_years)
    az = hpv.age_results(
        result_keys=sc.objdict(
            cancers_by_genotype=sc.objdict(
                timepoints=['2020'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 60., 65., 70., 75., 80., 100.]),
            ),
            cancers=sc.objdict(
                timepoints=['2020'],
                edges=np.array([0., 15., 20., 25., 30., 40., 45., 50., 55., 60., 65., 70., 75., 80., 100.]),            )
        )
    )
    analyzers = [dt]

    sim = hpv.Sim(
        n_agents=50e3,
        dt=0.5,
        location='nigeria',
        start=1950,
        end=2020,
        condoms=dict(m=0.01, c=0.2, o=0.1),
        genotypes=genotypes,
        analyzers=analyzers,
        ms_agent_ratio=100,
    )
    sim.initialize()
    # Create sim to get baseline prognoses parameters
    if calib_pars is not None:
        ## hpv 16 pars
        calib_pars['genotype_pars'].hpv16['dur_dysp']['par2'] = 3.8  # 4
        calib_pars['genotype_pars'].hpv16['dur_dysp']['par1'] = 7.25  # 13
        calib_pars['genotype_pars'].hpv16['prog_rate'] = 0.18  # 0.099
        calib_pars['genotype_pars'].hpv16['cancer_prob'] = 0.022  # 0.017

        ## hpv 18 pars
        calib_pars['genotype_pars'].hpv18['dur_dysp']['par2'] = 0.75
        calib_pars['genotype_pars'].hpv18['rel_beta'] = 1.22
        calib_pars['genotype_pars'].hpv18['cancer_prob'] = 0.13
        # calib_pars['genotype_pars'].hpv18['prog_rate'] = 0.9

        ## hr hpv pars
        calib_pars['genotype_pars'].hrhpv['dur_dysp']['par2'] = 18
        # calib_pars['genotype_pars'].hrhpv['dur_dysp']['par1'] = 18
        calib_pars['genotype_pars'].hrhpv['rel_beta'] = 0.76
        calib_pars['genotype_pars'].hrhpv['cancer_prob'] = 0.0026
        # calib_pars['genotype_pars'].hrhpv['prog_rate'] = 0.08

        sim.update_pars(calib_pars)

    sim.run()

    dt_res = sim.get_analyzer(age_causal_infection_by_genotype)
    dfs = sc.autolist()
    var_name_dict = {
        'age_causal': 'Causal HPV infection',
        'age_cancer': 'Cancer acquisition'
    }
    for gtype in genotypes:
        gt_dfs = sc.autolist()
        for val, call in zip(['age_causal', 'age_cancer'], [dt_res.age_causal, dt_res.age_cancer]):
            dwelltime_df = pd.DataFrame()
            dwelltime_df['age'] = call[gtype]
            dwelltime_df['var'] = var_name_dict[val]
            dwelltime_df['genotype'] = gtype
            gt_dfs += dwelltime_df
        gt_df = pd.concat(gt_dfs)
        dfs += gt_df
    dt_df = pd.concat(dfs)

    dfs = sc.autolist()
    for gtype in genotypes:
        gt_dfs = sc.autolist()
        for val, call in zip(['cin1', 'cin2', 'cin3', 'total'], [dt_res.dwelltime[gtype]['cin1'], dt_res.dwelltime[gtype]['cin2'],
                                                                dt_res.dwelltime[gtype]['cin3'], dt_res.dwelltime[gtype]['total']]):
            dwelltime_df = pd.DataFrame()
            dwelltime_df['years'] = call
            dwelltime_df['var'] = val
            dwelltime_df['genotype'] = gtype
            gt_dfs += dwelltime_df
        gt_df = pd.concat(gt_dfs)
        dfs += gt_df
    dt_df2 = pd.concat(dfs)

    hpu.set_font(size=20)
    f, axes = pl.subplots(1, 2, figsize=(16, 10))
    ax = axes[0]
    sns.violinplot(data=dt_df, x='var', y='age', hue='genotype', ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Age")
    ax.set_xticklabels(['Causal HPV\ninfection', 'Cancer\nacquisition'])
    sc.SIticks(ax)

    ax=axes[1]
    sns.violinplot(data=dt_df2, x='var', y='years', hue='genotype', ax=ax)
    ax.set_xlabel("Dwelltime in health state prior to cancer")
    ax.set_ylabel("Years")
    ax.set_xticklabels(['CIN1', 'CIN2', 'CIN3', 'TOTAL'])
    sc.SIticks(ax)
    f.tight_layout()
    pl.savefig(f"{hpu.figfolder}/fig5.png", dpi=100)



#%% Run as a script
if __name__ == '__main__':

    file = sc.thispath() / 'nigeria_pars.obj'
    calib_pars = sc.loadobj(file)

    plot_fig5(calib_pars=calib_pars)

    print('Done.')
