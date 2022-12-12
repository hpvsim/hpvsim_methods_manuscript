"""
This script runs Use Case 1 from the HPVsim methods manuscript.

*Motivation*
Prophylactic vaccination is one of the most essential and effective pillars of
the current public health response to HPV/cervical cancer. In the majority of
countries where prophylactic vaccination is routinely administered, it is
targeted to girls aged 9-14, with the intention being to vaccinate prior to
them being exposed to HPV (i.e. before they become sexually active).

Here we investigate the potential impact of expanding the age of vaccination
(EAV) in three different country archetypes that vary according to the average
age of first sex for girls (AFS) and the rates of sexual mixing between younger
women and older men, quantified by the average sexual partner age difference (SPAD):
 1. AFS = 18, SPAD = 1      --- most young girls (<18) will not have been exposed
 2. AFS = 18, SPAD = 10     --- more girls have been exposed via older partners
 3. AFS = 16, SPAD = 1      --- more girls expoesd via younger AFS

*Hypothesis*
EAV will have the highest impact in Setting 1, followed by Setting 2, then 3.
"""

import hpvsim as hpv
import numpy as np
import sciris as sc
import pandas as pd
import pylab as pl


#%% Run configurations
debug = 0
resfolder = 'results'
figfolder = 'figures'
to_run = [
    # 'run_sim',
    'run_scenarios',
    # 'plot_scenarios',
]

#%% Define parameters
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

mixing['s2'] = {
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

mixing['s3'] = mixing['s1']

#%% Define  functions to run
def make_sim(setting=None, campaign=None, seed=0, meta=None):
    ''' Make a single sim '''

    # Decide what message to print
    if meta is not None:
        msg = f'Making sim {meta.inds} ({meta.count} of {meta.n_sims}) for {setting}'
    else:
        msg = f'Making sim for {setting}'
    if debug: msg += ' IN DEBUG MODE'
    print(msg)

    # Parameters
    pars = dict(
        n_agents        = [50e3,5e3][debug],
        dt              = [0.5,1.0][debug],
        start           = [1975,2000][debug],
        end             = 2060,
        burnin          = [25,0][debug],
        condoms         = dict(m=0, c=0, o=0),
        debut           = debut[setting],
        mixing          = mixing[setting],
        use_multiscale  = False,
        ms_agent_ratio  = 100,
        rand_seed       = seed,
    )

    # Interventions
    interventions = []

    # Routine vaccination
    routine_vx = hpv.routine_vx(
        prob=.5,
        start_year=2015,
        product='bivalent',
        age_range=(9, 10),
        label='Routine'
    )
    interventions.append(routine_vx)

    if campaign is not None:
        # One-off catch-up for people 10-24
        campaign_vx = hpv.campaign_vx(
            prob=.5,
            years=2025,
            product='bivalent',
            age_range=(10, 24),
            label='Campaign'
        )
        interventions.append(campaign_vx)

    sim = hpv.Sim(pars, interventions=interventions)

    # Store metadata
    if meta is not None:
        sim.meta = meta # Copy over meta info
    else:
        sim.meta = sc.objdict()
    vx_label = 'no_campaign' if campaign is None else campaign
    sim.label = f'{setting}-{vx_label}-{seed}' # Set label


    return sim


def run_sim(verbose=None, setting=None, campaign=None, seed=0, meta=None):
    ''' Make and run a single sim '''
    sim = make_sim(setting=setting, campaign=campaign, seed=seed, meta=meta)
    sim.run(verbose=verbose)
    return sim


def make_msims(sims, use_mean=True):
    ''' Utility to take a slice of sims and turn it into a multisim '''

    msim = hpv.MultiSim(sims)
    msim.reduce(use_mean=use_mean)
    i_se, i_vx, i_s = sims[0].meta.inds
    for s, sim in enumerate(sims):  # Check that everything except seed matches
        assert i_se == sim.meta.inds[0]
        assert i_vx == sim.meta.inds[1]
        assert (s == 0) or i_s != sim.meta.inds[2]
    msim.meta = sc.objdict()
    msim.meta.inds = [i_se, i_vx]
    msim.meta.vals = sc.dcp(sims[0].meta.vals)
    msim.meta.vals.pop('seed')
    print(f'Processed multisim {msim.meta.vals.values()}... ')
    return msim


def run_scens(settings=None, vx_scens=None, n_seeds=5, verbose=0, debug=debug):
    ''' Run scenarios for all specified settings '''

    # Set up iteration arguments
    ikw = []
    count = 0
    n_sims = len(settings) * len(vx_scens) * n_seeds
    for i_se, setting in enumerate(settings):
        for i_vx, vx_scen in enumerate(vx_scens):
            for i_s in range(n_seeds):
                count += 1
                meta = sc.objdict()
                meta.count = count
                meta.n_sims = n_sims
                meta.inds = [i_se, i_vx, i_s]
                meta.vals = sc.objdict(setting=setting, campaign=vx_scen, seed=i_s)
                ikw.append(sc.dcp(meta.vals))
                ikw[-1].meta = meta

    # Run sims in parallel
    sc.heading(f'Running {len(ikw)} scenario sims...')
    kwargs = dict(verbose=verbose)
    all_sims = sc.parallelize(run_sim, iterkwargs=ikw, kwargs=kwargs, serial=debug)

    # Rearrange sims
    sims = np.empty((len(settings), len(vx_scens), n_seeds), dtype=object)
    for sim in all_sims:  # Unflatten array
        i_se, i_vx, i_s = sim.meta.inds
        sims[i_se, i_vx, i_s] = sim

    # Calculate cancers averted for each seed
    cancers_averted = sc.objdict()
    for i_se, setting in enumerate(settings):
        cancers_averted[setting] = sc.autolist()
        for i_s in range(n_seeds):
            baseline = sims[i_se, 0, i_s].results['cancers'][:].sum()
            campaign = sims[i_se, 1, i_s].results['cancers'][:].sum()
            cancers_averted[setting] += baseline - campaign
    sc.saveobj(f'{resfolder}/cancers_averted_uc1.obj', cancers_averted)

    # Prepare to convert sims to msims
    all_sims_for_multi = []
    for i_se, setting in enumerate(settings):
        for i_vx, vx_scen in enumerate(vx_scens):
            sim_seeds = sims[i_se, i_vx, :].tolist()
            all_sims_for_multi.append(sim_seeds)

    # Convert sims to msims
    all_msims = sc.parallelize(make_msims, iterarg=all_sims_for_multi)

    # Now strip out all the results and place them in a dataframe
    dfs = sc.autolist()
    msims = np.empty((len(settings), len(vx_scens)), dtype=object)
    for msim in all_msims:
        df = pd.DataFrame()
        i_se, i_vx = msim.meta.inds
        msims[i_se, i_vx] = msim
        df['year']      = msim.results['year']
        df['cancers']   = msim.results['cancers'][:]
        df['cancers_low']   = msim.results['cancers'].low
        df['cancers_high']   = msim.results['cancers'].high
        df['setting']   = settings[i_se]
        vx_scen_label = 'no_campaign' if vx_scens[i_vx] is None else vx_scens[i_vx]
        df['vx_scen'] = vx_scen_label
        dfs += df

    alldf = pd.concat(dfs)
    sc.saveobj(f'{resfolder}/results_uc1.obj', alldf)

    return alldf, msims


#%% Run as a script
if __name__ == '__main__':

    # Run single sim
    if 'run_sim' in to_run:
        sim = run_sim(verbose=0.1, setting='s1')

    # Run scenarios
    if 'run_scenarios' in to_run:
        settings = ['s1', 's2', 's3']
        vx_scens = ['no_campaign', 'campaign']
        n_seeds = [5,1][debug]
        alldf, msims = run_scens(settings=settings, vx_scens=vx_scens, n_seeds=n_seeds, verbose=-1, debug=debug)

    # Plot scenarios
    if 'plot_scenarios' in to_run:
        sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
        sc.options(font='Libertinus Sans', fontsize=24)

        settings = sc.objdict({'s1':'Setting 1', 's2':'Setting 2', 's3':'Setting 3'})
        vx_scens = sc.objdict({'no_campaign': 'No campaign', 'campaign':'Campaign'})
        bigdf = sc.loadobj(f'{resfolder}/results_uc1.obj')
        colors = sc.gridcolors(len(vx_scens))
        start_year = 2000
        intv_year = 2025
        cancers_averted = sc.objdict()

        for skey,sname in settings.items():

            fig, ax = pl.subplots(figsize=(16, 8))

            for vn, vkey, vname in vx_scens.enumitems():
                df = bigdf[(bigdf.setting == skey) & (bigdf.vx_scen == vkey)]
                si = sc.findinds(np.array(df.year), start_year)[0]
                ii = sc.findinds(np.array(df.year), intv_year)[0]
                years = np.array(df.year)[si:]
                best = np.array(df['cancers'])[si:]
                low = np.array(df['cancers_low'])[si:]
                high = np.array(df['cancers_high'])[si:]
                ax.plot(years, best, color=colors[vn], label=vname)
                ax.fill_between(years, low, high, color=colors[vn], alpha=0.3)
            ax.legend(loc='upper left')
            sc.SIticks(ax)
            sc.setylim([0,350])
            fig.tight_layout()
            fig_name = f'{figfolder}/{skey}.png'
            sc.savefig(fig_name, dpi=100)

            cancers_averted[skey] = bigdf[(bigdf.setting == skey) & (bigdf.vx_scen == 'no_campaign')].cancers[ii:].sum() - bigdf[(bigdf.setting == skey) & (bigdf.vx_scen == 'campaign')].cancers[ii:].sum()

    print('Done.')
