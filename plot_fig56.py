"""
This script compares two distinct sexual networks
"""

import hpvsim as hpv
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import utils as ut

# %% Run configurations
debug = 0
resfolder = 'results'
figfolder = 'figures'
location = 'india'

# Run settings
n_trials    = [2000, 1][debug]  # How many trials to run for calibration
n_workers   = [40, 1][debug]    # How many cores to use
storage     = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations
n_seeds     = [10, 1][debug]

to_run = [
    # 'run_simple',
    # 'calibrate'
    # 'plot_calibrate',
    # 'run_sims',
    'plot_fig5_sims',
    # 'run_screening',
    'plot_fig6_screening',
]


# %% Define functions to run
def make_network(location):
    # Set network pars

    if location == 'rwanda':
        layer_probs = dict(
            m=np.array([
                [0, 5,  10,    15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],
                [0, 0, 0.05, 0.30, 0.60, 0.70, 0.70, 0.70, 0.70, 0.70, 0.60, 0.40, 0.40, 0.40, 0.40, 0.40],  # Females
                [0, 0, 0.01, 0.01, 0.10, 0.50, 0.60, 0.70, 0.70, 0.70, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60]]  # Males
            ),
            c=np.array([
                # Share of people of each age in casual partnerships
                [0, 5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],
                [0, 0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.50, 0.40, 0.20, 0.01, 0.01, 0.01, 0.01, 0.01],  # Females
                [0, 0, 0.05, 0.30, 0.30, 0.40, 0.40, 0.40, 0.40, 0.40, 0.30, 0.10, 0.05, 0.01, 0.01, 0.01]],  # Males
            ),
        )

    elif location == 'india':
        layer_probs = dict(
            m=np.array([
                [0, 5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],
                [0, 0, 0.05, 0.25, 0.60, 0.80, 0.95, 0.80, 0.80, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],  # Share f
                [0, 0, 0.01, 0.05, 0.10, 0.70, 0.90, 0.90, 0.90, 0.90, 0.80, 0.60, 0.60, 0.60, 0.60, 0.60]]  # Share m
            ),
            c=np.array([
                [0, 5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],
                [0, 0, 0.10, 0.50, 0.60, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.50, 0.01, 0.01],  # Share f
                [0, 0, 0.10, 0.20, 0.25, 0.35, 0.40, 0.70, 0.90, 0.90, 0.95, 0.95, 0.70, 0.30, 0.10, 0.10]],  # Share m
            ),
        )

    else:
        raise ValueError('Unknown network')

    if location == 'rwanda':
        m_partners = dict(
            m=dict(dist='poisson1', par1=0.001),
            c=dict(dist='poisson1', par1=1),
        )
        f_partners = dict(
            m=dict(dist='poisson1', par1=0.001),
            c=dict(dist='poisson1', par1=0.2),
        )

    elif location == 'india':
        m_partners = dict(
            m=dict(dist='poisson1', par1=0.01),
            c=dict(dist='poisson1', par1=0.1),
        )
        f_partners = dict(
            m=dict(dist='poisson1', par1=0.01),
            c=dict(dist='neg_binomial', par1=2, par2=0.025),
        )

    else:
        raise ValueError('Unknown network')

    f_cross_layer = 0.025
    m_cross_layer = 0.25

    return layer_probs, m_partners, f_partners, m_cross_layer, f_cross_layer


class partner_count(hpv.Analyzer):
    def __init__(self, layer='casual', year=2020):
        super().__init__()
        self.layer = layer
        self.layer_args = sc.objdict({
            'marital': 0,
            'casual': 1
        })
        self.lno = self.layer_args[self.layer]
        self.year = year
        self.partners = None

    def initialize(self, sim):
        super().initialize()

    def apply(self, sim):
        if sim.yearvec[sim.t] == self.year:
            f_conds = sim.people.is_female * sim.people.alive * sim.people.level0 * sim.people.is_active
            m_conds = sim.people.is_male * sim.people.alive * sim.people.level0 * sim.people.is_active
            self.partners = {
                'f': sim.people.n_rships[self.lno, f_conds],
                'm': sim.people.n_rships[self.lno, m_conds],
            }


class dwelltime_by_genotype(hpv.Analyzer):
    def __init__(self, start_year=None, **kwargs):
        super().__init__(**kwargs)
        self.start_year = start_year
        self.years = None

    def initialize(self, sim):
        super().initialize(sim)
        self.years = sim.yearvec
        if self.start_year is None:
            self.start_year = sim['start']
        self.age_causal = []
        self.age_cancer = []
        self.age_cin = []

    def apply(self, sim):
        if sim.yearvec[sim.t] >= self.start_year:
            cancer_genotypes, cancer_inds = (sim.people.date_cancerous == sim.t).nonzero()
            if len(cancer_inds):
                current_age = sim.people.age[cancer_inds]
                date_exposed = sim.people.date_exposed[cancer_genotypes, cancer_inds]
                dur_cin = sim.people.dur_cin[cancer_genotypes, cancer_inds]
                total_time = (sim.t - date_exposed) * sim['dt']
                self.age_causal += (current_age - total_time).tolist()
                self.age_cin += (current_age - dur_cin).tolist()
                self.age_cancer += current_age.tolist()
        return


def make_sim(location, seed=0, debug=0, add_analyzers=False, interventions=None):
    """ Make a single sim for a given network """

    # Parameters
    layer_probs, m_partners, f_partners, m_cross_layer, f_cross_layer = make_network(location)

    pars = dict(
        n_agents=[20e3, 5e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 2000][debug],
        burnin=[30, 0][debug],
        end=2020,
        beta={'india': 0.28, 'rwanda': 0.16}[location],
        genotypes=[16, 18, 'hi5', 'ohr'],
        location=location,
        debut=dict(f=dict(dist='lognormal', par1=15., par2=2.),
                   m=dict(dist='lognormal', par1=20., par2=2.)),
        layer_probs=layer_probs,
        m_partners=m_partners,
        f_partners=f_partners,
        m_cross_layer=m_cross_layer,
        f_cross_layer=f_cross_layer,
        ms_agent_ratio=100,
        rand_seed=seed,
        verbose=0.0,
    )
    analyzers = sc.autolist()
    if add_analyzers:
        analyzers += partner_count()
        analyzers += dwelltime_by_genotype()

    sim = hpv.Sim(pars=pars, analyzers=analyzers, interventions=interventions)

    return sim


def calibrate(location=None, n_trials=None, n_workers=None, do_save=True, filestem=''):
    """ Calibrate """

    sim = make_sim(location)
    datafiles = [
        f'data/{location}_cancer_cases.csv',
    ]
    if location == 'india':
        datafiles += [
            'data/india_cin_types.csv',
            'data/india_cancer_types.csv',
        ]

    # Define the calibration parameters
    calib_pars = dict(
        beta=[0.2, 0.1, 0.34, 0.02]
    )

    # Different priors depending on location
    if location == 'india':
        calib_pars['m_cross_layer'] = [0.5, 0.3, 0.7, 0.05]
        calib_pars['m_partners'] = dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        )
        calib_pars['f_cross_layer']=[0.05, 0.01, 0.1, 0.01]
    elif location == 'rwanda':
        calib_pars['m_cross_layer']=[0.1, 0.05, 0.2, 0.01]
        calib_pars['m_partners']=dict(
            c=dict(par1=[3, 1, 5, 0.5])
        )
        calib_pars['f_cross_layer']=[0.02, 0.01, 0.1, 0.01]

    calib = hpv.Calibration(sim, calib_pars=calib_pars,
                            name=f'{location}_calib_final',
                            datafiles=datafiles,
                            total_trials=n_trials, n_workers=n_workers,
                            storage=storage
                            )
    calib.calibrate()
    filename = f'{location}_calib{filestem}'
    if do_save:
        sc.saveobj(f'results/{filename}.obj', calib)

    print(f'Best pars are {calib.best_pars}')

    return sim, calib

def plot_calib(location, which_pars=0, save_pars=True, filestem=''):
    """ Plot calibration """
    filename = f'{location}_calib{filestem}'
    calib = sc.load(f'results/{filename}.obj')

    sc.fonts(add=sc.thisdir(aspath=True) / 'Libertinus Sans')
    sc.options(font='Libertinus Sans')
    fig = calib.plot(res_to_plot=200, plot_type='sns.boxplot', do_save=False)
    fig.tight_layout()
    fig.savefig(f'{filename}.png')

    if save_pars:
        calib_pars = calib.trial_pars_to_sim_pars(which_pars=which_pars)
        trial_pars = sc.autolist()
        for i in range(100):
            trial_pars += calib.trial_pars_to_sim_pars(which_pars=i)
        sc.save(f'results/{location}_pars{filestem}.obj', calib_pars)
        sc.save(f'results/{location}_pars{filestem}_all.obj', trial_pars)

    return calib


def run_sim(location, verbose=None, seed=0, debug=0):
    """ Make and run a single sim """
    sim = make_sim(seed=seed, debug=debug, location=location)
    sim.run(verbose=verbose)
    sim.shrink()
    return sim


def make_screening(end_probs=None):
    """ Create the different screening and treatment algorithms """

    # Shared parameters
    algos = dict()
    if end_probs is None: end_probs = [0.2, 0.4, 0.6]
    years = np.linspace(2000,2020,21)
    ablate_prob = 0.7
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))

    for end_prob in end_probs:
        primary_screen_prob = np.linspace(0, end_prob, 21)

        via_primary = hpv.routine_screening(
            product='via',
            prob=primary_screen_prob,
            eligibility=screen_eligible,
            years=years,
            label='via primary',
        )

        via_positive = lambda sim: sim.get_intervention('via primary').outcomes['positive']
        ablation = hpv.treat_num(
            prob = ablate_prob,
            product = 'ablation',
            eligibility = via_positive,
            label = 'ablation'
        )

        algos[end_prob] = [via_primary, ablation]

    return algos


def plot_degree(partner_dict, dwelltime_df):
    """ Plot partner distributions """
    ut.set_font(size=16)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="tight")

    bins = np.concatenate([np.arange(21), [100]])
    rn = 0

    for location, partners in partner_dict.items():
        for cn, slabel in enumerate(['females']):
            sex = slabel[0]
            counts, bins = np.histogram(partners[sex], bins=bins)
            total = sum(counts)
            counts = counts / total
            degree_dist = {'rwanda': 'Poisson', 'india': 'Negative binomal'}[location]

            axes[rn, cn].bar(bins[:-1], counts)
            axes[rn, cn].set_xlabel(f'Number of lifetime casual partners')
            axes[rn, cn].set_title(f'Casual partners for females\n{degree_dist} degree distribution')
            axes[rn, cn].set_ylim([0, 1])
            stats = f"Mean: {np.mean(partners[sex]):.1f}\n"
            stats += f"Median: {np.median(partners[sex]):.1f}\n"
            stats += f"Std: {np.std(partners[sex]):.1f}\n"
            stats += f"%>20: {np.count_nonzero(partners[sex] >= 20) / total * 100:.2f}\n"
            axes[rn, cn].text(15, 0.5, stats)

        sns.boxplot(
            x="Health event",
            y="Age",
            data=dwelltime_df[dwelltime_df.location == location.capitalize()],
            ax=axes[rn, cn+1],
            showfliers=False,
        )
        axes[rn, cn+1].set_title(f'Age distribution of key health events\n{degree_dist} degree distribution')
        axes[rn, cn+1].set_xlabel('')
        axes[rn, cn+1].set_ylim([0, 100])

        rn += 1

    plt.savefig(f"{figfolder}/fig5.png", dpi=100)
    plt.show()

    return


def make_sims(locations, seed=0, debug=0, add_analyzers=True):
    """ Make multiple sims """
    sims = sc.autolist()
    for location in locations:
        sims += make_sim(seed=seed, debug=debug, location=location, add_analyzers=add_analyzers, )
    return sims


def make_scens(location, end_probs=None, n_seeds=1):
    """ Make screening scenarios """
    algos = make_screening(end_probs=end_probs)
    sims = sc.autolist()
    for end_prob, algo in algos.items():
        for seed in range(n_seeds):
            sim = make_sim(location=location, interventions=algo, seed=seed)
            sim.label = f'Screening {end_prob}'
            sim['verbose'] = 0.1
            sims += sim
    return sims


def run_sims(locations, add_analyzers=True, seed=0, debug=0):
    """ Run the simulations """
    sims = make_sims(locations, seed=seed, debug=debug, add_analyzers=add_analyzers)
    msim = hpv.parallel(sims)
    partner_dict = dict()
    dfs = sc.autolist()

    for sim in msim.sims:
        location = sim['location']
        partner_count = sim.get_analyzer('partner_count')
        partner_dict[location] = partner_count.partners

        # Make dwelltime dataframe
        dt_res = sim.get_analyzer('dwelltime_by_genotype')
        dt_dfs = sc.autolist()

        dt_df = pd.DataFrame()
        dt_df['Age'] = np.array(dt_res.age_causal)[np.array(dt_res.age_causal)<50]
        dt_df['Health event'] = 'Causal HPV infection'
        dt_df['location'] = location.capitalize()
        dt_dfs += dt_df

        dt_df = pd.DataFrame()
        dt_df['Age'] = np.array(dt_res.age_cin)[np.array(dt_res.age_causal)<65]
        dt_df['Health event'] = 'CIN2+'
        dt_df['location'] = location.capitalize()
        dt_dfs += dt_df

        dt_df = pd.DataFrame()
        dt_df['Age'] = np.array(dt_res.age_cancer)[np.array(dt_res.age_causal)<90]
        dt_df['Health event'] = 'Cancer'
        dt_df['location'] = location.capitalize()
        dt_dfs += dt_df

        df = pd.concat(dt_dfs)

        dfs += df

    dwelltime_df = pd.concat(dfs)

    return msim, partner_dict, dwelltime_df


# %% Run as a script
if __name__ == '__main__':

    if 'run_simple' in to_run:
        sim = run_sim(location, verbose=0.1)
        sim.plot()

    if 'calibrate' in to_run:
        sim, calib = calibrate(location=location, n_trials=n_trials, n_workers=n_workers, do_save=True, filestem='')

    if 'plot_calibrate' in to_run:
        calib = plot_calib(location=location)

    if 'run_sims' in to_run:
        msim, partner_dict, dwelltime_df = run_sims(debug=debug, locations=['rwanda', 'india'])
        sc.saveobj(f'{resfolder}/partner_dict.obj', partner_dict)
        sc.saveobj(f'{resfolder}/dwelltime_df.obj', dwelltime_df)

    if 'plot_fig5_sims' in to_run:
        partner_dict = sc.loadobj(f'{resfolder}/partner_dict.obj')
        dwelltime_df = sc.loadobj(f'{resfolder}/dwelltime_df.obj')
        plot_degree(partner_dict, dwelltime_df)

    if 'run_screening' in to_run:
        end_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]
        sims = make_scens('india', end_probs=end_probs, n_seeds=n_seeds)
        big_msim = hpv.parallel(sims, n_cpus=n_workers)
        mlist = big_msim.split(chunks=len(end_probs))
        results = dict()
        for msim in mlist:
            msim.reduce()
            results[msim.sims[0].label] = msim.results

        sc.saveobj(f'{resfolder}/scen_results.obj', results)

    if 'plot_fig6_screening' in to_run:
        end_probs = [0.0, 0.05, 0.1, 0.15, 0.2]
        end_prob_labels = ['0%', '5%', '10%', '15%', '20%']
        results = sc.loadobj(f'{resfolder}/scen_results.obj')

        for rnum, res in results.items():
            print(f'{rnum.ljust(14,"0")}: {res.asr_cancer_incidence[-1]:.2f}, ({res.asr_cancer_incidence.low[-1]:.2f}, {res.asr_cancer_incidence.high[-1]:.2f})')

        # Bar plot with errors
        x = [1, 2, 3, 4, 5]
        y = [res.asr_cancer_incidence[-1] for res in results.values()]
        y_errormin = [res.asr_cancer_incidence.low[-1] for res in results.values()]
        y_errormax = [res.asr_cancer_incidence.high[-1] for res in results.values()]
        y_error = [[y[i]-y_errormin[i] for i in range(5)], [y_errormax[i]-y[i] for i in range(5)]]
        ut.set_font(size=18)
        fig, ax = plt.subplots(1, 1, figsize=(9, 5), layout="tight")
        ax.bar(x[:5], y[:5])
        ax.errorbar(x[:5], y[:5], yerr=y_error, fmt='o', color='k')
        ax.set_xticks(x, end_prob_labels)
        ax.set_title('ASR cancer incidence, 2020')
        ax.set_xlabel('Lifetime screening coverage, 2020')
        plt.axhline(y=18, color='k', linestyle='--')
        plt.savefig(f"{figfolder}/fig6.png", dpi=100)
        plt.show()

    print('Done.')
