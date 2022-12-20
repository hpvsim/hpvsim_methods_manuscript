"""
This script is used for running calibration tool
"""
import hpvsim as hpv
import hpvsim.utils as hpu
import pylab as pl
import pandas as pd
from scipy.stats import lognorm
import numpy as np
import sciris as sc
import utils as ut
import seaborn as sns

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
def run_calib_tool(calib_pars=None):
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
    analyzers = [dt, pe, az]

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
        print(calib_pars)
        calib_pars['genotype_pars'].hpv18['dur_dysp']['par2'] = 1
        calib_pars['genotype_pars'].hpv16['dur_dysp']['par2'] = 6
        calib_pars['genotype_pars'].hrhpv['dur_dysp']['par2'] = 8
        calib_pars['genotype_pars'].hpv16['dur_dysp']['par1'] = 13
        calib_pars['genotype_pars'].hrhpv['dur_dysp']['par1'] = 18
        calib_pars['genotype_pars'].hpv18['rel_beta'] = 1.25
        calib_pars['genotype_pars'].hpv18['cancer_prob'] = 0.15
        calib_pars['genotype_pars'].hpv18['prog_rate'] = 0.9
        calib_pars['genotype_pars'].hrhpv['prog_rate'] = 0.08
        sim.update_pars(calib_pars)

    sim.run()

    # Get parameters
    genotype_pars = sim['genotype_pars']


    ut.set_font(size=20)
    # set palette
    colors = sc.gridcolors(10)
    n_samples = 10
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])

    fig, ax = pl.subplots(4, 2, figsize=(16, 18))
    pn = 0
    x = np.linspace(0.01, 2, 200)

    for gi, gtype in enumerate(genotypes):
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_precin']['par1'],
                                         genotype_pars[gtype]['dur_precin']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, 0].plot(x, rv.pdf(x), color=colors[gi], lw=2, label=gtype.upper())
        ax[1, 0].plot(x, ut.logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[gi], lw=2, label=gtype.upper())
        pn += 1

        ax[1, 0].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")
        for row in [0, 1]:
            ax[row, 0].set_ylabel("")
            ax[row, 0].grid()
            ax[row, 0].set_xticks(np.arange(3))
            ax[row, 0].set_xticklabels([0, 12, 24])
    ax[0, 0].set_ylabel("Frequency")
    ax[1, 0].set_ylabel("Probability of developing\ndysplasia")
    ax[0, 0].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")

    pn = 0
    thisx = np.linspace(0.01, 25, 100)

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        ai=1
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_dysp']['par1'],
                                         genotype_pars[gtype]['dur_dysp']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, ai].plot(thisx, rv.pdf(thisx), color=colors[gi], lw=2, label=gtype.upper())
        ax[1, ai].plot(thisx, ut.logf1(thisx, genotype_pars[gtype]['prog_rate']), color=colors[gi], lw=2,
                       label=gtype.upper())
        for year in range(1, 26):
            peaks = ut.logf1(year, hpu.sample(dist='normal', par1=genotype_pars[gtype]['prog_rate'],
                                              par2=genotype_pars[gtype]['prog_rate_sd'], size=n_samples))
            ax[1, ai].plot([year] * n_samples, peaks, color=colors[gi], lw=0, marker='o', alpha=0.5)
        pn += 1

        ax[0, ai].set_ylabel("")
        ax[0, ai].legend(fontsize=18)
        ax[0, ai].grid()
        ax[0, ai].set_ylabel("Frequency")
        ax[0, ai].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")

        # Severity
        ax[1, ai].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")
        ax[1, ai].set_ylabel("Clinical severity")
        ax[1, ai].set_ylim([0, 1])
        ax[1, ai].axhline(y=0.33, ls=':', c='k')
        ax[1, ai].axhline(y=0.67, ls=':', c='k')
        ax[1, ai].axhspan(0, 0.33, color=cmap[0], alpha=.4)
        ax[1, ai].axhspan(0.33, 0.67, color=cmap[1], alpha=.4)
        ax[1, ai].axhspan(0.67, 1, color=cmap[2], alpha=.4)
        ax[1, ai].text(-0.3, 0.08, 'CIN1', rotation=90)
        ax[1, ai].text(-0.3, 0.4, 'CIN2', rotation=90)
        ax[1, ai].text(-0.3, 0.73, 'CIN3', rotation=90)

    az_res = sim.get_analyzer(hpv.age_results)
    data_cancers = pd.read_csv('nigeria_cancer_cases.csv')
    data_types = pd.read_csv('nigeria_cancer_types.csv')
    for gi, gtype in enumerate(genotypes):
        ax[2,0].plot(az_res.results['cancers_by_genotype']['bins'],
                az_res.results['cancers_by_genotype']['2020.0'][:,gi], color=colors[gi], label=gtype)
    ax[3,0].plot(az_res.results['cancers']['bins'], az_res.results['cancers']['2020.0'], label='model')
    ax[2,0].set_title('Cancers by genotype/age')
    ax[3,0].set_title('Cancers by age')
    data_cancers.plot(x='age', y='value', label='data', ax=ax[3,0])
    ax[3,0].set_xlabel('')
    ax[3,0].legend()
    ax[2,1].plot(sim.res_yearvec[10:], sim.results['cancer_incidence'].values[10:], color=colors[8], label='Crude cancer incidence')
    ax[2,1].plot(sim.res_yearvec[10:], sim.results['asr_cancer_incidence'].values[10:], color=colors[9], label='Age-standardized cancer incidence')
    ax[2,1].legend()
    ax[3,1].scatter(genotypes, sim.results['cancerous_genotype_dist'][:,-1], label='model')
    ax[3,1].scatter(genotypes, data_types['value'], label='data')
    ax[3,1].legend()
    ax[3,1].set_title('HPV type distribution in cancer')
    sc.SIticks(ax[2,0])
    sc.SIticks(ax[3,0])
    sc.SIticks(ax[2,1])
    sc.SIticks(ax[3,1])
    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/calib_tool.png", dpi=100)
    fig.show()



#%% Run as a script
if __name__ == '__main__':

    file = f'nigeria_pars.obj'
    calib_pars = sc.loadobj(file)

    run_calib_tool(calib_pars=calib_pars)

    print('Done.')
