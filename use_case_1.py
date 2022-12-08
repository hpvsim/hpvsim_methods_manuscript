"""
This script runs Use Case 1 from the HPVsim methods manuscript.
Motivation:

Prophylactic vaccination is one of the most essential and effective pillars of
the current public health response to HPV/cervical cancer. In the majority of
countries where prophylactic vaccination is routinely administered, it is
targeted to girls aged 9-14, with the intention being to vaccinate prior to
them being exposed to HPV (i.e. before they become sexually active).

Here we investigate the potential impact of expanding the age of vaccination
(EAV) in three different country archetypes that vary according to the average
age of first sex for girls (AFS) and the rates of sexual mixing between younger
women and older men, quantified by the average sexual partner age difference (SPAD):
 1. AFS = 18, SPAD = 10
 2. AFS = 18, SPAD = 1
 3. AFS = 16, SPAD = 1

Hypothesis: EAV will have the highest impact in Setting 1, followed by Setting 2,
then Setting 3.
"""

import hpvsim as hpv  # Import HPVsim as a Python module
import numpy as np

#%% Define parameters and functions
debug=0
debut = dict()
mixing = dict()

# Define ASF for all 3 archetypes
debut['s1'] = dict(
    f=dict(dist='normal', par1=18., par2=2.),
    m=dict(dist='normal', par1=20., par2=2.),
)
debut['s2'] = dict(
    f=dict(dist='normal', par1=18., par2=2.),
    m=dict(dist='normal', par1=19., par2=2.),
)
debut['s3'] = dict(
    f=dict(dist='normal', par1=16., par2=2.),
    m=dict(dist='normal', par1=17., par2=2.),
)

# Define SPAD for all 3 archetypes
mixing['s1'] = {
    k:np.array([
        #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
        [0,     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [5,     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [15,    0,  0,  1, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [20,    0,  0,  1, .5, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [25,    0,  0,  0,  1, .5, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [30,    0,  0,  0, .5,  1, .5, .5,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [35,    0,  0,  0, .5, .5,  1, .5, .5,  0,  0,  0,  0,  0,  0,  0,  0],
        [40,    0,  0,  0, .4, .5, .5,  1, .5, .5,  0,  0,  0,  0,  0,  0,  0],
        [45,    0,  0,  0, .3, .4, .5, .5,  1, .5, .5,  0,  0,  0,  0,  0,  0],
        [50,    0,  0,  0, .2, .3, .4, .5, .5,  1, .5, .5,  0,  0,  0,  0,  0],
        [55,    0,  0,  0, .1, .2, .3, .4, .5, .5,  1, .5, .5,  0,  0,  0,  0],
        [60,    0,  0,  0,  0, .1, .2, .3, .4, .5, .5,  1, .5, .5,  0,  0,  0],
        [65,    0,  0,  0,  0,  0, .1, .2, .3, .4, .5, .5,  1, .5, .5,  0,  0],
        [70,    0,  0,  0,  0,  0,  0, .1, .2, .3, .4, .5, .5,  1, .5,  1, .5],
        [75,    0,  0,  0,  0,  0,  0,  0, .1, .2, .3, .4, .5, .5,  1,  1, 1],
    ]) for k in ['m','c','o']
}

mixing['s2'] = {
    k:np.array([
        #       0,  5,  10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75
        [0,     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [5,     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [10,    0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [15,    0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [20,    0,  0,  1, .5,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [25,    0,  0,  0,  0, .5,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [30,    0,  0,  0,  0,  0, .5,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [35,    0,  0,  0,  0,  0,  0, .5,  1,  0,  0,  0,  0,  0,  0,  0,  0],
        [40,    0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
        [45,    0,  0,  0,  0,  0,  0,  0,  0, .5,  1,  0,  0,  0,  0,  0,  0],
        [50,    0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  1,  0,  0,  0,  0,  0],
        [55,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  1,  0,  0,  0,  0],
        [60,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  1,  0,  0,  0],
        [65,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  1,  0,  0],
        [70,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  1,  0],
        [75,    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, .5,  1],
    ]) for k in ['m','c','o']
}

mixing['s3'] = mixing['s2']


def run_scens(setting=None, verbose=0):

    # Parameters
    pars = dict(
        n_agents       = [50e3,5e3][debug],
        dt             = [0.5,1.0][debug],
        start          = [1950,1980][debug],
        end            = 2040,
        network        = 'default',
        debut          = debut[setting],
        mixing         = mixing[setting],
    )

    sim = hpv.Sim(pars)

    # Interventions
    # Routine vaccination
    routine_vx = hpv.routine_vx(
        prob=.5,
        start_year=2015,
        product='bivalent',
        age_range=(9, 10),
        label='Routine'
    )
    # One-off catch-up for people 10-24
    campaign_vx = hpv.campaign_vx(
        prob=.5,
        start_year=2025,
        product='bivalent',
        age_range=(10, 24),
        label='Campaign'
    )

    # Define the scenarios
    scenarios = {
        'baseline': {
            'name': 'Baseline',
            'pars': {
                'interventions': [routine_vx]
            }
        },
        'catchup': {
            'name': 'Catch-up',
            'pars': {
                'interventions': [routine_vx, campaign_vx]
            }
        }
    }

    metapars = {'n_runs': 3}

    scens = hpv.Scenarios(sim=sim, metapars=metapars, scenarios=scenarios)
    scens.run(verbose=verbose, debug=debug)
    scens.plot(do_save=True, fig_path=f'figures/{setting}_uc1.png')

    return scens


    #%% Run as a script
if __name__ == '__main__':

    for setting in ['s1', 's2', 's3']:
        scen = run_scens(setting)

    print('Done.')
