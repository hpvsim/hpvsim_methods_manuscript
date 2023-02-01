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
from analyzers import dalys


#%% Run configurations
debug = 0
resfolder = 'results'
figfolder = 'figures'
to_run = [
    'run_scenarios',
    # 'run_cea',
    # 'plot_scenarios',
]


#%% Define functions to run
def make_sim(seed=0):
    ''' Make a single sim '''

    # Parameters
    pars = dict(
        n_agents        = [50e3,5e3][debug],
        dt              = [0.25,1.0][debug],
        start           = [1950,2000][debug],
        burnin          = 70,
        end             = 2060,
        ms_agent_ratio  = 100,
        rand_seed       = seed,
    )

    sim = hpv.Sim(pars=pars, analyzers=[hpv.daly_computation(start=2020)])

    return sim


def run_sim(verbose=None, seed=0):
    ''' Make and run a single sim '''
    sim = make_sim(seed=seed)
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
    #### Algorithm 1 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # VIA, then immediate ablation for anyone eligible
    ####################################################################
    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (
                sim.t > (sim.people.date_screened + 5 / sim['dt']))
    via_primary = hpv.routine_screening(
        product='via',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='via primary',
    )

    via_positive = lambda sim: sim.get_intervention('via primary').outcomes['positive']
    ablation1 = hpv.treat_num(
        prob=ablate_prob,
        product='ablation',
        eligibility=via_positive,
        label='ablation'
    )

    algo1 = [via_primary, ablation1]
    for intv in algo1: intv.do_plot = False

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
    #### Algorithm 4 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV genotype, triage oHR results with VIA, ablate all HPV 16/18+
    ####################################################################

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    hpv_geno_primary = hpv.routine_screening(
        product='hpv_type',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='hpv genotype primary',
    )

    # Send abnormal cytology results, plus ASCUS results that were HPV+, for colpo
    to_via = lambda sim: sim.get_intervention('hpv genotype primary').outcomes['positive_ohr']
    via_triage = hpv.routine_triage(
        product='via',
        prob = triage_screen_prob,
        annual_prob=False,
        eligibility=to_via,
        label = 'via triage'
    )

    # Triage ASCUS with HPV test
    to_ablate = lambda sim: list(set(sim.get_intervention('hpv genotype primary').outcomes['positive_1618'].tolist() + sim.get_intervention('via triage').outcomes['positive'].tolist()))
    ablation4 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = to_ablate,
        label = 'ablation'
    )
    algo4 = [hpv_geno_primary, via_triage, ablation4]
    for intv in algo4: intv.do_plot=False

    ####################################################################
    #### Algorithm 5 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV testing, then VIA triage
    ####################################################################

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (
                sim.t > (sim.people.date_screened + 5 / sim['dt']))
    hpv_primary = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='hpv primary',
    )

    hpv_positive = lambda sim: sim.get_intervention('hpv primary').outcomes['positive']
    via_triage = hpv.routine_triage(
        product='via',
        prob=triage_screen_prob,
        annual_prob=False,
        eligibility=hpv_positive,
        label='via triage'
    )

    to_treat = lambda sim: sim.get_intervention('via triage').outcomes['positive']
    ablation5 = hpv.treat_num(
        prob=ablate_prob,
        product='ablation',
        eligibility=to_treat,
        label='ablation'
    )

    algo5 = [hpv_primary, via_triage, ablation5]
    for intv in algo5: intv.do_plot = False

    ####################################################################
    #### Algorithm 6 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV DNA testing, triage HPV+ with colposcopy/biopsy, then ablation for all
    # HSILs
    ####################################################################

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (sim.t > (sim.people.date_screened + 5 / sim['dt']))
    hpv_primary = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='hpv primary',
    )

    # Triage HPV+ with Colposcopy
    to_colpo = lambda sim: sim.get_intervention('hpv primary').outcomes['positive']
    colpo_triage = hpv.routine_triage(
        product='colposcopy',
        prob=triage_screen_prob,
        annual_prob=False,
        eligibility=to_colpo,
        label='colposcopy'
    )

    # After colpo, treat HSILs with ablation
    hsils = lambda sim: sim.get_intervention('colposcopy').outcomes['hsil']
    ablation6 = hpv.treat_num(
        prob = ablate_prob,
        product = 'ablation',
        eligibility = hsils,
        label = 'ablation'
    )

    algo6 = [hpv_primary, colpo_triage, ablation6]
    for intv in algo6: intv.do_plot=False

    ####################################################################
    #### Algorithm 7 (https://www.ncbi.nlm.nih.gov/books/NBK572308/)
    # HPV DNA testing, triage HPV+ with cytology, colposcopy/biopsy for ASCUS+, then ablation for all
    # HSILs
    ####################################################################

    screen_eligible = lambda sim: np.isnan(sim.people.date_screened) | (
                sim.t > (sim.people.date_screened + 5 / sim['dt']))
    hpv_primary = hpv.routine_screening(
        product='hpv',
        prob=primary_screen_prob,
        eligibility=screen_eligible,
        start_year=start_year,
        label='hpv primary',
    )

    # Triage HPV+ with cytology
    to_cyto = lambda sim: sim.get_intervention('hpv primary').outcomes['positive']
    cytology = hpv.routine_triage(
        product='lbc',
        prob=triage_screen_prob,
        eligibility=to_cyto,
        annual_prob=False,
        label='cytology',
    )

    # Triage ASCUS with HPV test
    to_colpo = lambda sim: list(set(sim.get_intervention('cytology').outcomes['ascus'].tolist() + sim.get_intervention('cytology').outcomes['abnormal'].tolist()))
    colpo_triage = hpv.routine_triage(
        product='colposcopy',
        prob=triage_screen_prob,
        annual_prob=False,
        eligibility=to_colpo,
        label='colposcopy'
    )

    # After colpo, treat HSILs with ablation
    hsils = lambda sim: sim.get_intervention('colposcopy').outcomes['hsil']
    ablation7 = hpv.treat_num(
        prob=ablate_prob,
        product='ablation',
        eligibility=hsils,
        label='ablation'
    )

    algo7 = [hpv_primary, cytology, colpo_triage, ablation7]
    for intv in algo7: intv.do_plot = False

    ####################################################################
    #### Set up scenarios to compare algoriths 1-7
    ####################################################################
    scenarios = {
        'baseline': {'name': 'No screening','pars': {}},
        'algo1':    {'name': 'Algorithm 1', 'pars': {'interventions': algo1}},
        'algo2':    {'name': 'Algorithm 2', 'pars': {'interventions': algo2}},
        'algo3':    {'name': 'Algorithm 3', 'pars': {'interventions': algo3}},
        'algo4':    {'name': 'Algorithm 4', 'pars': {'interventions': algo4}},
        'algo5':    {'name': 'Algorithm 5', 'pars': {'interventions': algo5}},
        'algo6':    {'name': 'Algorithm 6', 'pars': {'interventions': algo6}},
        'algo7':    {'name': 'Algorithm 7', 'pars': {'interventions': algo7}},
    }
    scens = hpv.Scenarios(sim=sim, metapars={'n_runs': n_seeds}, scenarios=scenarios)
    scens.run()

    return scens


def run_cea():
    scens = sc.loadobj(f'{resfolder}/uc1_scens.obj')

    # Extract number of products used in screening, triage, and treatment from each scenario
    s0 = scens.sims[0][0]
    s1 = scens.sims[1][0] # algo1
    s2 = scens.sims[2][0] # algo2
    s3 = scens.sims[3][0] # algo3
    s4 = scens.sims[4][0] # algo4
    s5 = scens.sims[5][0] # algo5
    s6 = scens.sims[6][0] # algo6
    s7 = scens.sims[7][0] # algo7
    si = sc.findinds(s2.res_yearvec, 2025)[0]

    dalys = {}
    for sim, scen in zip([s0, s1, s2, s3, s4, s5, s6, s7],['baseline', 'algo1', 'algo2', 'algo3', 'algo4', 'algo5', 'algo6', 'algo7']):
        a = sim.get_analyzers()[0]
        df = a.df
        discounted_cancers = np.array([i / 1.03 ** t for t, i in enumerate(df['new_cancers'].values)])
        discounted_cancer_deaths = np.array([i / 1.03 ** t for t, i in enumerate(df['new_cancer_deaths'].values)])
        avg_age_ca_death = np.mean(df['av_age_cancer_deaths'])
        avg_age_ca = np.mean(df['av_age_cancers'])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(0.4 * ca_years * discounted_cancers)
        yll = np.sum((84 - avg_age_ca_death) * discounted_cancer_deaths)
        daly = yll + yld
        dalys[scen] = daly


    products = {
        'algo1': {
            'via primary': s1.get_intervention('via primary').n_products_used[si:],# * pop_scale,
            'ablation': s1.get_intervention('ablation').n_products_used[si:],# * pop_scale,
        },
        'algo2': {
            'hpv primary':  s2.get_intervention('hpv primary').n_products_used[si:],#*pop_scale,
            'ablation':     s2.get_intervention('ablation').n_products_used[si:],#*pop_scale,
        },
        'algo3': {
            'cytology':     s3.get_intervention('cytology').n_products_used[si:],#*pop_scale,
            'hpv triage':   s3.get_intervention('hpv triage').n_products_used[si:],#*pop_scale,
            'colposcopy':   s3.get_intervention('colposcopy').n_products_used[si:],#*pop_scale,
            'ablation':     s3.get_intervention('ablation').n_products_used[si:],#*pop_scale,
        },
        'algo4': {
            'hpv genotype primary': s4.get_intervention('hpv genotype primary').n_products_used[si:],  # * pop_scale,
            'via triage': s4.get_intervention('via triage').n_products_used[si:],  # * pop_scale,
            'ablation': s4.get_intervention('ablation').n_products_used[si:],  # * pop_scale,
        },
        'algo5': {
            'hpv primary': s5.get_intervention('hpv primary').n_products_used[si:],# * pop_scale,
            'via triage': s5.get_intervention('via triage').n_products_used[si:],# * pop_scale,
            'ablation': s5.get_intervention('ablation').n_products_used[si:],# * pop_scale,
        },
        'algo6': {
            'hpv primary': s6.get_intervention('hpv primary').n_products_used[si:],  # * pop_scale,
            'colposcopy': s6.get_intervention('colposcopy').n_products_used[si:],  # * pop_scale,
            'ablation': s6.get_intervention('ablation').n_products_used[si:],  # * pop_scale,
        },
        'algo7': {
            'hpv primary': s7.get_intervention('hpv primary').n_products_used[si:],  # * pop_scale,
            'cytology': s7.get_intervention('cytology').n_products_used[si:],
            'colposcopy': s7.get_intervention('colposcopy').n_products_used[si:],  # * pop_scale,
            'ablation': s7.get_intervention('ablation').n_products_used[si:],  # * pop_scale,
        },
    }

    # Total products
    total_products = {}
    for algo in ['algo1', 'algo2', 'algo3', 'algo4', 'algo5', 'algo6', 'algo7']:
        total_products[algo] = {}
        for product_name, products_used in products[algo].items():
            total_products[algo][product_name] = sum(products_used)

    # Create a hypothetical dataframe of costs
    costs = {'hpv primary': 20, 'cytology': 20, 'hpv triage': 20, 'colposcopy': 20, 'ablation': 5, 'hpv genotype primary': 30,
             'via primary': 13, 'via triage': 13}

    # Dicsounted costs
    discounted_costs = {}
    len_t = s2.res_tvec[-1]-s2.res_tvec[si]+1
    dr = 0.03 # Discounting rate
    for product_name, cost in costs.items():
        discounted_costs[product_name] = cost/(1+dr)**np.arange(len_t)

    # Calculate the total cost of each
    total_costs = {}
    total_ablations = {}
    for algo in ['baseline', 'algo1', 'algo2', 'algo3', 'algo4', 'algo5', 'algo6', 'algo7']:
        total_costs[algo] = 0
        if algo != 'baseline':
            for product_name, products_used in products[algo].items():
                total_costs[algo] += sum(discounted_costs[product_name] * products[algo][product_name])
                if product_name == 'ablation':
                    total_ablations[algo] = sum(products[algo][product_name])

    # Impact
    cancer_redux = {
        'algo1': sum((s0.results.cancers[si:] - s1.results.cancers[si:]) ),
        'algo2': sum((s0.results.cancers[si:] - s2.results.cancers[si:]) ),
        'algo3': sum((s0.results.cancers[si:] - s3.results.cancers[si:]) ),
        'algo4': sum((s0.results.cancers[si:] - s4.results.cancers[si:])),
        'algo5': sum((s0.results.cancers[si:] - s5.results.cancers[si:]) ),
        'algo6': sum((s0.results.cancers[si:] - s6.results.cancers[si:])),
        'algo7': sum((s0.results.cancers[si:] - s7.results.cancers[si:])),
    }

    # ICERs
    icers = {
        'algo1': total_costs['algo1'] / (dalys['baseline'] - dalys['algo1']),
        'algo2': total_costs['algo2'] / (dalys['baseline'] - dalys['algo2']),
        'algo3': total_costs['algo3'] / (dalys['baseline'] - dalys['algo3']),
        'algo4': total_costs['algo4'] / (dalys['baseline'] - dalys['algo4']),
        'algo5': total_costs['algo5'] / (dalys['baseline'] - dalys['algo5']),
        'algo6': total_costs['algo6'] / (dalys['baseline'] - dalys['algo6']),
        'algo7': total_costs['algo7'] / (dalys['baseline'] - dalys['algo7']),
    }

    total_df = pd.DataFrame(columns=['Scenario', 'DALYs', 'Costs'])
    total_df['Scenario'] = dalys.keys()
    total_df['DALYs'] = dalys.values()
    total_df['Costs'] = total_costs.values()

    base_DALYs = total_df.iloc[0]['DALYs']
    total_df['DALYs averted'] = base_DALYs - total_df['DALYs']

    data_to_plot = total_df.copy()
    # data_to_plot = total_df[total_df['Scenario'] != 'baseline']
    efficiency_data = data_to_plot.copy().sort_values('Costs').reset_index(drop=True)
    efficient_scenarios = efficiency_data['Scenario'].values
    num_scens = len(efficient_scenarios)-1
    icers = sc.autolist()
    icers += 0
    i = 1
    while i <= num_scens:
        inc_DALYs = efficiency_data.iloc[i]['DALYs averted'] - efficiency_data.iloc[i - 1]['DALYs averted']
        inc_cost = efficiency_data.iloc[i]['Costs'] - efficiency_data.iloc[i - 1]['Costs']
        if inc_DALYs < 0: # if it averts negative DALYs it is dominated by definition
            efficiency_data = efficiency_data.drop(i).reset_index(drop=True)
            efficient_scenarios = np.delete(efficient_scenarios, i)
            num_scens -=1
        else:
            icer = inc_cost / inc_DALYs
            extended_dominance_check = True
            while extended_dominance_check:
                if icer < icers[i - 1]:
                    efficiency_data = efficiency_data.drop(i - 1).reset_index(drop=True)
                    efficient_scenarios = np.delete(efficient_scenarios, i - 1)
                    num_scens -= 1
                    icers = np.delete(icers, i - 1)
                    i -= 1
                    inc_DALYs = efficiency_data.iloc[i]['DALYs averted'] - efficiency_data.iloc[i - 1]['DALYs averted']
                    inc_cost = efficiency_data.iloc[i]['Costs'] - efficiency_data.iloc[i - 1]['Costs']
                    if inc_DALYs < 0:
                        efficiency_data = efficiency_data.drop(i).reset_index(drop=True)
                        efficient_scenarios = np.delete(efficient_scenarios, i)
                        num_scens -= 1
                        i -= 1
                        extended_dominance_check = False
                    else:
                        icer = inc_cost / inc_DALYs
                        icers = np.append(icers,icer)
                else:
                    if icer not in icers:
                        icers = np.append(icers,icer)
                    i += 1
                    extended_dominance_check = False

    scen_labels = {
        'baseline': 'No Screening',
        'algo1': 'Algorithm 1',
        'algo2': 'Algorithm 2',
        'algo3': 'Algorithm 3',
        'algo4': 'Algorithm 4',
        'algo5': 'Algorithm 5',
        'algo6': 'Algorithm 6',
        'algo7': 'Algorithm 7',
    }
    ut.set_font(size=20)
    f, ax = pl.subplots(figsize=(10, 10))

    colors = sc.gridcolors(10)
    efficiency_data.plot(ax=ax, kind='line', x='DALYs averted', y='Costs', color='black',
                         label='Efficiency frontier')

    for i, scen in enumerate(['baseline', 'algo1', 'algo2', 'algo3', 'algo4', 'algo5', 'algo6', 'algo7']):
        group = data_to_plot[data_to_plot['Scenario'] == scen]
        group.plot(ax=ax, kind='scatter', x='DALYs averted', y='Costs', label=scen_labels[scen],
                   color=colors[i], s=200)

    ax.set_xlabel('DALYs averted, 2020-2060')
    ax.set_ylabel('Total costs, $USD 2020-2060')
    ax.legend(fancybox=True)  # , title='Screening method')
    sc.SIticks(ax)
    f.tight_layout()
    fig_name = f'{figfolder}/ICER.png'
    sc.savefig(fig_name, dpi=100)



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
        'Treatments': ['new_cin_treatments'],
        }
        scens.plot(to_plot=to_plot, n_cols=1, fig_path=f'{figfolder}/uc1_plot.png', do_save=True)

    print('Done.')