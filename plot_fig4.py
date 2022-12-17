"""
This script produces figure 4 of the HPVsim methods paper, showing the natural history
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
def plot_fig4(calib_pars=None):
    # Group genotypes
    genotypes = ['hpv16', 'hpv18', 'hrhpv']

    dt = age_causal_infection_by_genotype(start_year=2010)
    analyzers = [dt]

    sim = hpv.Sim(n_agents=50e3, location='nigeria', genotypes=genotypes, analyzers=analyzers)
    sim.initialize()
    # Create sim to get baseline prognoses parameters
    if calib_pars is not None:
        # calib_pars['genotype_pars'].hpv16.sero_prob = 0.8
        # calib_pars['genotype_pars'].hpv18.sero_prob = 0.8
        # calib_pars['genotype_pars'].hrhpv.sero_prob = 0.8
        # calib_pars['genotype_pars'].hpv18['dur_dysp']['par1'] = 5
        # calib_pars['genotype_pars'].hpv16['dur_dysp']['par1'] = 15
        # calib_pars['genotype_pars'].hrhpv['dur_dysp']['par1'] = 20
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

    # Get parameters
    ng = sim['n_genotypes']
    genotype_map = sim['genotype_map']

    if calib_pars is not None:
        genotype_pars = calib_pars['genotype_pars']
    else:
        genotype_pars = sim['genotype_pars']

    # Shorten duration names
    dur_precin = [genotype_pars[genotype_map[g]]['dur_precin'] for g in range(ng)]
    dur_dysp = [genotype_pars[genotype_map[g]]['dur_dysp'] for g in range(ng)]
    dysp_rate = [genotype_pars[genotype_map[g]]['dysp_rate'] for g in range(ng)]
    prog_rate = [genotype_pars[genotype_map[g]]['prog_rate'] for g in range(ng)]
    cancer_prob = [genotype_pars[genotype_map[g]]['cancer_prob'] for g in range(ng)]

    ut.set_font(size=20)
    # Set common attributes
    colors = sc.gridcolors(ng)
    n_samples = 10
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])

    fig, ax = pl.subplots(2, 3, figsize=(16, 10))
    pn = 0
    x = np.linspace(0.01, 2, 200)

    for gtype in genotypes:
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_precin']['par1'],
                                         genotype_pars[gtype]['dur_precin']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, 0].plot(x, rv.pdf(x), color=colors[pn], lw=2, label=gtype.upper())
        ax[1, 0].plot(x, ut.logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[pn], lw=2, label=gtype.upper())
        pn += 1

        ax[1, 0].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")
        for row in [0, 1]:
            ax[row, 0].set_ylabel("")
            ax[row, 0].grid()
            ax[row, 0].set_xticklabels([0, 0, 12, 24])
    ax[0, 0].set_ylabel("Frequency")
    ax[1, 0].set_ylabel("Probability of developing\ndysplasia")
    ax[0, 0].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")

    pn = 0
    thisx = np.linspace(0.01, 25, 100)

    # Durations and severity of dysplasia
    for gtype in genotypes:
        ai=1
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_dysp']['par1'],
                                         genotype_pars[gtype]['dur_dysp']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, ai].plot(thisx, rv.pdf(thisx), color=colors[pn], lw=2, label=gtype.upper())
        ax[1, ai].plot(thisx, ut.logf1(thisx, genotype_pars[gtype]['prog_rate']), color=colors[pn], lw=2,
                       label=gtype.upper())
        for year in range(1, 26):
            peaks = ut.logf1(year, hpu.sample(dist='normal', par1=genotype_pars[gtype]['prog_rate'],
                                              par2=genotype_pars[gtype]['prog_rate_sd'], size=n_samples))
            if pn == 1:
                ax[1, ai].plot([year] * n_samples, peaks, color=colors[pn], lw=0, marker='o', alpha=0.5)
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

    shares = []
    gtypes = []
    noneshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], []
    longx = np.linspace(0.01, 25, 1000)
    for g in range(ng):
        sigma, scale = ut.lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        aa = np.diff(rv.cdf(longx))
        bb = ut.logf1(longx, dysp_rate[g])[1:]
        shares.append(np.dot(aa, bb))
        gtypes.append(genotype_map[g].replace('hpv', ''))

    for g in range(ng):
        sigma, scale = ut.lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        dd = ut.logf1(longx, prog_rate[g])
        indcin1 = sc.findinds(dd < .33)[-1]
        if (dd > .33).any():
            indcin2 = sc.findinds((dd > .33) & (dd < .67))[-1]
        else:
            indcin2 = indcin1
        if (dd > .67).any():
            num_cancers = len(hpu.true(hpu.n_binomial(cancer_prob[g], len(sc.findinds((dd > .67))))))
            if num_cancers:
                indcin3 = sc.findinds((dd > .67) )[-num_cancers]
            else:
                indcin3 = sc.findinds((dd > .67) )[-1]

        else:
            indcin3 = indcin2
            num_cancers = 0

        noneshares.append(1 - shares[g])
        cin1shares.append(((rv.cdf(longx[indcin1]) - rv.cdf(longx[0])) * shares[g]))
        cin2shares.append(((rv.cdf(longx[indcin2]) - rv.cdf(longx[indcin1])) * shares[g]))
        cin3shares.append(((rv.cdf(longx[indcin3]) - rv.cdf(longx[indcin2])) * shares[g]))
        cancershares.append((num_cancers/len(dd)) * shares[g])

    ai=2

    sns.violinplot(data=dt_df, x='var', y='age', hue='genotype', ax=ax[0,ai])
    ax[0,ai].set_xlabel("")
    ax[0, ai].set_ylabel("Age")
    ax[0,ai].set_xticklabels(['Causal HPV\ninfection', 'Cancer\nacquisition'])
    sc.SIticks(ax[0, ai])

    bottom = np.zeros(ng + 1)
    all_shares = [noneshares + [sum([j * 1 / ng for j in noneshares])],
                  cin1shares + [sum([j * 1 / ng for j in cin1shares])],
                  cin2shares + [sum([j * 1 / ng for j in cin2shares])],
                  cin3shares + [sum([j * 1 / ng for j in cin3shares])],
                  cancershares + [sum([j * 1 / ng for j in cancershares])],
                  ]
    for gn, grade in enumerate(['No dysplasia', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        if len(ydata.shape) > 1: ydata = ydata[:, 0]
        color = cmap[gn - 1] if gn > 0 else 'gray'
        ax[1,ai].bar(np.arange(1, ng + 2), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata
    ax[1,ai].set_xticks(np.arange(ng + 1) + 1)
    ax[1,ai].set_xticklabels(gtypes + ['Average'])
    ax[1,ai].set_ylabel("")
    ax[1,ai].legend()

    pl.figtext(0.06, 0.94, 'A', fontweight='bold', fontsize=30)
    pl.figtext(0.375, 0.94, 'C', fontweight='bold', fontsize=30)
    pl.figtext(0.69, 0.94, 'E', fontweight='bold', fontsize=30)
    pl.figtext(0.06, 0.51, 'B', fontweight='bold', fontsize=30)
    pl.figtext(0.375, 0.51, 'D', fontweight='bold', fontsize=30)
    pl.figtext(0.69, 0.51, 'F', fontweight='bold', fontsize=30)

    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/fig4.png", dpi=100)


#%% Run as a script
if __name__ == '__main__':

    file = f'nigeria_pars.obj'
    calib_pars = sc.loadobj(file)

    plot_fig4(calib_pars=calib_pars)

    print('Done.')
