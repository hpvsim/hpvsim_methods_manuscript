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
    # 'run_cea',
    # 'plot_scenarios',
]


#%% Define analyzer for computing DALYs
class dalys(hpv.Analyzer):

    def __init__(self, max_age=84, cancer=None, dysplasia=None, **kwargs):
        super().__init__(**kwargs)
        self.max_age = max_age
        self.cancer = cancer if cancer else dict(dur=1, wt=0.16325) # From GBD 2017, calculated as 1 yr @ 0.288 (primary), 4 yrs @ 0.049 (maintenance), 0.5 yrs @ 0.451 (late), 0.5 yrs @ 0.54 (tertiary), scaled up to 12 years
        return

    def __getitem__(self, key):
        return getattr(self, key)

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


#%% Define functions to run
def make_sim(seed=0):
    ''' Make a single sim '''

    # Parameters
    pars = dict(
        n_agents        = [50e3,5e3][debug],
        dt              = [0.5,1.0][debug],
        start           = [1975,2000][debug],
        end             = 2060,
        ms_agent_ratio  = 10,
        burnin          = [45,0][debug],
        rand_seed       = seed,
    )

    sim = hpv.Sim(pars=pars, analyzers=dalys)

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
        'algo3':    {'name': 'Algorithm 3', 'pars': {'interventions': algo3}},
    }
    scens = hpv.Scenarios(sim=sim, metapars={'n_runs': n_seeds}, scenarios=scenarios)
    scens.run()

    return scens


def run_cea():
    scens = sc.loadobj(f'{resfolder}/uc1_scens.obj')

    # Extract number of products used in screening, triage, and treatment from each scenario
    s0 = scens.sims[0][0]
    s2 = scens.sims[1][0]
    s3 = scens.sims[2][0]
    si = sc.findinds(s2.res_yearvec, 2025)[0]
    pop_scale = 1250 # Approximate number of people represented by each agent - assumes pop size of 62.5m in 1975

    products = {
        'algo2': {
            'hpv primary':  s2.get_intervention('hpv primary').n_products_used[si:]*pop_scale,
            'ablation':     s2.get_intervention('ablation').n_products_used[si:]*pop_scale,
        },
        'algo3': {
            'cytology':     s3.get_intervention('cytology').n_products_used[si:]*pop_scale,
            'hpv triage':   s3.get_intervention('hpv triage').n_products_used[si:]*pop_scale,
            'colposcopy':   s3.get_intervention('colposcopy').n_products_used[si:]*pop_scale,
            'ablation':     s3.get_intervention('ablation').n_products_used[si:]*pop_scale,
        }
    }

    # Total products
    total_products = {}
    for algo in ['algo2', 'algo3']:
        total_products[algo] = {}
        for product_name, products_used in products[algo].items():
            total_products[algo][product_name] = sum(products_used)

    # Create a hypothetical dataframe of costs
    screen_cost = 1
    treat_cost = 1*4
    costs_threshold = {'hpv primary': screen_cost, 'cytology': screen_cost, 'hpv triage': screen_cost, 'colposcopy': screen_cost, 'ablation': treat_cost}
    costs = {'hpv primary': 5, 'cytology': 12.5, 'hpv triage': 5, 'colposcopy': 25, 'ablation': 50}

    # Dicsounted costs
    discounted_costs = {}
    len_t = s2.res_tvec[-1]-s2.res_tvec[si]+1
    dr = 0.03 # Discounting rate
    for product_name, cost in costs.items():
        discounted_costs[product_name] = cost/(1+dr)**np.arange(len_t)

    # Calculate the total cost of each
    total_costs = {}
    for algo in ['algo2', 'algo3']:
        total_costs[algo] = {}
        total_costs[algo]['total'] = 0
        for product_name, products_used in products[algo].items():
            total_costs[algo][product_name] = sum(discounted_costs[product_name] * products[algo][product_name])
            total_costs[algo]['total'] += total_costs[algo][product_name]

    # Impact
    discounted_impact = {
        'algo2': sum((s0.results.cancers[si:] - s2.results.cancers[si:])/(1 + dr) ** np.arange(len_t)),
        'algo3': sum((s0.results.cancers[si:] - s3.results.cancers[si:]) / (1 + dr) ** np.arange(len_t))
    }

    # ICERs
    icers = {
        'algo2': total_costs['algo2']['total'] / discounted_impact['algo2'],
        'algo3': total_costs['algo3']['total'] / discounted_impact['algo3']
    }

    # Print results
    print(f"Algorithm 2: {total_costs['algo2']['total']}")
    print(f"Algorithm 3: {total_costs['algo3']['total']}")

    print(f"Algorithm 2: {icers['algo2']}")
    print(f"Algorithm 3: {icers['algo3']}")



#%% Run as a script
if __name__ == '__main__':

    # Run scenarios
    if 'run_scenarios' in to_run:
        scens = run_scens(verbose=0.1)
        sc.saveobj(f'{resfolder}/uc1_scens.obj', scens)

    # Run cost-effectiveness analyses
    if 'run_cea' in to_run:
        run_cea()

    # Run scenarios
    if 'plot_scenarios' in to_run:
        scens = sc.loadobj(f'{resfolder}/uc1_scens.obj')

        to_plot = {
        'Age standardized cancer incidence': ['asr_cancer_incidence'],
        # 'Cancer cases': ['cancers'],
        # 'Cancer deaths': ['cancer_deaths'],
        'Treatments': ['new_cin_treatments'],
        }
        scens.plot(to_plot=to_plot, n_cols=1, fig_path=f'{figfolder}/uc1_plot.png', do_save=True)

    print('Done.')
