"""
This script compares two distinct sexual networks
"""

import hpvsim as hpv
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt

# %% Run configurations
debug = 1
resfolder = 'results'
figfolder = 'figures'

# Run settings
n_trials    = [3000, 1][debug]  # How many trials to run for calibration
n_workers   = [40, 1][debug]    # How many cores to use
storage     = ["mysql://hpvsim_user@localhost/hpvsim_db", None][debug]  # Storage for calibrations

to_run = [
    'calibrate'
    # 'run_sims',
    # 'plot_sims'
]


# %% Define functions to run
def make_network(location):
    # Set network pars

    if location == 'rwanda':
        layer_probs = dict(
            m=np.array([
                [0, 5,  10,    15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],
                [0, 0, 0.05, 0.25, 0.70, 0.90, 0.95, 0.70, 0.75, 0.65, 0.55, 0.40, 0.40, 0.40, 0.40, 0.40],  # Females
                [0, 0, 0.01, 0.01, 0.10, 0.50, 0.60, 0.70, 0.70, 0.70, 0.70, 0.80, 0.70, 0.60, 0.50, 0.60]]  # Males
            ),
            c=np.array([
                # Share of people of each age in casual partnerships
                [0, 5,   10,   15,   20,   25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75],
                [0, 0, 0.10, 0.60, 0.30, 0.20, 0.20, 0.20, 0.20, 0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # Females
                [0, 0, 0.05, 0.70, 0.80, 0.60, 0.60, 0.50, 0.50, 0.40, 0.30, 0.10, 0.05, 0.01, 0.01, 0.01]],  # Males
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
                [0, 0, 0.10, 0.10, 0.10, 0.10, 0.50, 0.90, 0.90, 0.80, 0.80, 0.50, 0.40, 0.10, 0.01, 0.01],  # Share f
                [0, 0, 0.10, 0.20, 0.50, 0.60, 0.80, 0.90, 0.90, 0.80, 0.80, 0.70, 0.50, 0.30, 0.10, 0.10]],  # Share m
            ),
        )

    else:
        raise ValueError('Unknown network')

    if location == 'rwanda':
        m_partners = dict(
            m=dict(dist='poisson1', par1=0.001),
            c=dict(dist='poisson1', par1=3),
        )
        f_partners = dict(
            m=dict(dist='poisson1', par1=0.001),
            c=dict(dist='poisson1', par1=0.2),
        )

    elif location == 'india':
        m_partners = dict(
            m=dict(dist='poisson1', par1=0.01),
            c=dict(dist='poisson1', par1=0.2),
        )
        f_partners = dict(
            m=dict(dist='poisson1', par1=0.01),
            c=dict(dist='neg_binomial', par1=2, par2=0.025),
        )

    else:
        raise ValueError('Unknown network')

    if location == 'rwanda':
        f_cross_layer = 0.02,
        m_cross_layer = 0.10,

    elif location == 'india':
        f_cross_layer = 0.05,
        m_cross_layer = 0.50,

    else:
        raise ValueError('Unknown network')

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


def make_sim(location, seed=0, debug=0, add_analyzers=False):
    ''' Make a single sim for a given network '''

    # Parameters
    layer_probs, m_partners, f_partners, m_cross_layer, f_cross_layer = make_network(location)

    pars = dict(
        n_agents=[20e3, 5e3][debug],
        dt=[0.25, 1.0][debug],
        start=[1960, 2000][debug],
        burnin=[30, 0][debug],
        end=2020,
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
    )
    analyzers = sc.autolist()
    if add_analyzers:
        analyzers += partner_count()
    sim = hpv.Sim(pars=pars, analyzers=analyzers)

    return sim


def calibrate(location=None, n_trials=None, n_workers=None, do_save=True, filestem=''):

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
        calib_pars['m_cross_layer']=[0.5, 0.3, 0.7, 0.05]
        calib_pars['m_partners']=dict(
            c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        )
        calib_pars['f_cross_layer']=[0.05, 0.01, 0.1, 0.01]
        # calib_pars['f_partners']=dict(
        #     c=dict(par1=[2, 1, 3, 0.2],
        #            par2=[0.025, 0.01, 0.1, 0.005])
        # )
    elif location == 'rwanda':
        calib_pars['m_cross_layer']=[0.1, 0.05, 0.2, 0.01]
        calib_pars['m_partners']=dict(
            c=dict(par1=[3, 1, 5, 0.5])
        )
        calib_pars['f_cross_layer']=[0.02, 0.01, 0.1, 0.01]
        # calib_pars['f_partners']=dict(
        #     c=dict(par1=[0.2, 0.1, 0.6, 0.02])
        # )

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


def run_sim(location, verbose=None, seed=0, debug=0):
    ''' Make and run a single sim '''
    sim = make_sim(seed=seed, debug=debug, location=location)
    sim.run(verbose=verbose)
    sim.shrink()
    return sim


def plot_degree(partner_dict):
    # Create and run the simulation
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout="tight")

    bins = np.concatenate([np.arange(21), [100]])
    rn = 0

    for location, partners in partner_dict.items():
        for cn, slabel in enumerate(['females', 'males']):
            sex = slabel[0]
            counts, bins = np.histogram(partners[sex], bins=bins)
            total = sum(counts)
            counts = counts / total

            axes[rn, cn].bar(bins[:-1], counts)
            axes[rn, cn].set_xlabel(f'Number of lifetime casual partners')
            axes[rn, cn].set_title(f'Casual partners, {slabel}, {location.capitalize()}')
            axes[rn, cn].set_ylim([0, 1])
            stats = f"Mean: {np.mean(partners[sex]):.1f}\n"
            stats += f"Median: {np.median(partners[sex]):.1f}\n"
            stats += f"Std: {np.std(partners[sex]):.1f}\n"
            stats += f"%>20: {np.count_nonzero(partners[sex] >= 20) / total * 100:.2f}\n"
            axes[rn, cn].text(15, 0.5, stats)
        rn += 1

    plt.show()

    return


def make_sims(locations, seed=0, debug=0):
    """ Set up scenarios to compare algorithms """
    sims = sc.autolist()
    for location in locations:
        sims += make_sim(seed=seed, debug=debug, location=location)
    return sims


def run_sims(locations, seed=0, debug=0):
    """ Run the simulations """
    sims = make_sims(locations, seed=seed, debug=debug)
    msim = hpv.parallel(sims)
    partner_dict = dict()
    for sim in msim.sims:
        a = sim.analyzers[0]
        partner_dict[sim['location']] = a.partners
    return msim, partner_dict

# %% Run as a script
if __name__ == '__main__':

    if 'calibrate' in to_run:
        sim, calib = calibrate(location='india', n_trials=n_trials, n_workers=n_workers, do_save=True, filestem='')

    if 'run_sims' in to_run:
        msim, partner_dict = run_sims(debug=debug, locations=['rwanda', 'india'])
        sc.saveobj(f'{resfolder}/partner_dict.obj', partner_dict)

    if 'plot_sims' in to_run:
        partner_dict = sc.loadobj(f'{resfolder}/partner_dict.obj')
        plot_degree(partner_dict)


    print('Done.')
