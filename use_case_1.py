"""
This script runs Use Case 1 from the HPVsim methods manuscript.

*Motivation*
Here we investigate the potential impact of changing from a screen-and-treat algorithm
to a screen-triage-treat algorithm
"""

import hpvsim as hpv
import numpy as np
import sciris as sc
import pandas as pd
import pylab as pl
import matplotlib as mpl

# Imports from this repository
import utils as ut



#%% Run configurations
debug = 0
resfolder = 'results'
figfolder = 'figures'
to_run = [
    'run_scenarios',
    # 'plot_scenarios',
]


#%% Define functions to run
def make_sim(seed=0):
    ''' Make a single sim '''

    # Parameters
    pars = dict(
        n_agents        = [50e3,5e3][debug],
        dt              = [0.5,1.0][debug],
        start           = [1975,2000][debug],
        end             = 2060,
        burnin          = [45,0][debug],
        rand_seed       = seed,
    )
    sim = hpv.Sim(pars=pars)
    return sim


def run_sim(verbose=None, seed=0):
    ''' Make and run a single sim '''
    sim = make_sim(seed=seed, meta=meta)
    sim.run(verbose=verbose)
    sim.shrink()
    return sim


def run_scens(sim=None, seed=0, n_seeds=3, meta=None, verbose=0, debug=debug):
    ''' Run scenarios for all specified settings '''

    if sim is None: sim = make_sim(seed=seed)

    # Shared parameters
    primary_screen_prob = 0.2
    triage_screen_prob = 0.9
    ablate_prob = 0.9
    start_year = 2025

    ####################################################################
    #### Algorithm 2 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV testing, then immediate ablation for anyone eligible
    ####################################################################
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    hpv_primary = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='hpv primary',
    )

    hpv_positive = lambda sim: sim.get_intervention('hpv primary').outcomes['positive']
    ablation2 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hpv_positive,
        label = 'ablation'
    )

    algo2 = [hpv_primary, ablation2]
    for intv in algo2: intv.do_plot=False

    ####################################################################
    #### Algorithm 3 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # Cytology testing, triage ASCUS results with HPV, triage all HPV+ and
    # abnormal cytology results with colposcopy/biopsy, then ablation for all
    # HSILs
    ####################################################################

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    cytology = hpv.routine_screening(
        product='lbc',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='cytology',
    )

    # Triage ASCUS with HPV test
    ascus = lambda sim: sim.get_intervention('cytology').outcomes['ascus']
    hpv_triage = hpv.routine_triage(
        product='hpv',
        prob=triage_screen_prob,
        annual_prob=False,
        eligibility=ascus,
        label='hpv triage'
    )

    # Send abnormal cytology results, plus ASCUS results that were HPV+, for colpo
    to_colpo = lambda sim: list(set(sim.get_intervention('cytology').outcomes['abnormal'].tolist() + sim.get_intervention('hpv triage').outcomes['positive'].tolist()))
    colpo = hpv.routine_triage(
        product='colposcopy',
        prob = triage_screen_prob,
        annual_prob=False,
        eligibility=to_colpo,
        label = 'colposcopy'
    )

    # After colpo, treat HSILs with ablation
    hsils = lambda sim: sim.get_intervention('colposcopy').outcomes['hsil']
    ablation3 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hsils,
        label = 'ablation'
    )

    algo3 = [cytology, hpv_triage, colpo, ablation3]
    for intv in algo3: intv.do_plot=False

    ####################################################################
    #### Set up scenarios to compare algoriths 2 & 3
    ####################################################################
    scenarios = {
        'baseline': {'name': 'No screening','pars': {}},
        'algo2':    {'name': 'Algorithm 2', 'pars': {'interventions': algo2}},
        'algo2':    {'name': 'Algorithm 3', 'pars': {'interventions': algo3}},
    }
    scens = hpv.Scenarios(sim=sim, metapars={'n_runs': 3}, scenarios=scenarios)
    scens.run()

    return scens



#%% Run as a script
if __name__ == '__main__':

    # Run scenarios
    if 'run_scenarios' in to_run:
        scens = run_scens(verbose=0.1)
        sc.saveobj(f'{resfolder}/uc1_scens.obj', scens)

    # Run scenarios
    if 'plot_scenarios' in to_run:
        scens = sc.loadobj(f'{resfolder}/uc1_scens.obj')

        to_plot = {
        'Age standardized cancer incidence': ['asr_cancer_incidence'],
        'Cancer cases': ['cancers'],
        'Cancer deaths': ['cancer_deaths'],
        'Treatments': ['new_cin_treatments'],
        }
        scens.plot(to_plot=to_plot, n_cols=2, fig_path=f'{figfolder}/uc1_plot.png', do_save=True)

    print('Done.')
