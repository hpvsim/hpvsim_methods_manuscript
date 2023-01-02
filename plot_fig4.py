"""
This script produces figure 4 of the HPVsim methods paper, showing the natural history
"""
import hpvsim as hpv
import hpvsim.utils as hpu
import pylab as pl
import pandas as pd
from scipy.stats import lognorm, norm
import numpy as np
import sciris as sc
import utils as ut
import seaborn as sns


#%% Plotting function
def plot_fig4(calib_pars=None):

    # Group genotypes
    genotypes = ['hpv16', 'hpv18', 'hrhpv']
    sim = hpv.Sim(genotypes=genotypes)
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


    # Get parameters
    ng = sim['n_genotypes']
    genotype_map = sim['genotype_map']
    genotype_pars = sim['genotype_pars']

    # Shorten duration names
    dur_precin = [genotype_pars[genotype_map[g]]['dur_precin'] for g in range(ng)]
    dur_dysp = [genotype_pars[genotype_map[g]]['dur_dysp'] for g in range(ng)]
    dysp_rate = [genotype_pars[genotype_map[g]]['dysp_rate'] for g in range(ng)]
    prog_rate = [genotype_pars[genotype_map[g]]['prog_rate'] for g in range(ng)]
    cancer_prob = [genotype_pars[genotype_map[g]]['cancer_prob'] for g in range(ng)]


    ####################
    # Make figure, set fonts and colors
    ####################
    ut.set_font(size=32)
    colors = sc.gridcolors(10)
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])
    fig, ax = pl.subplot_mosaic('AB;CD;EF', figsize=(16, 20))

    ####################
    # Panel A and B
    ####################

    x = np.linspace(0.01, 3, 200) # Make an array of durations 0-3 years
    glabels = [16,18,'OHR']

    # Loop over genotypes, plot each one
    for gi, gtype in enumerate(genotypes):
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_precin']['par1'],
                                         genotype_pars[gtype]['dur_precin']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax['A'].plot(x, rv.pdf(x), color=colors[gi], lw=2, label=glabels[gi])
        ax['B'].plot(x, ut.logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[gi], lw=2, label=gtype.upper())

    # Axis labeling and other settings
    ax['B'].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")
    for axn in ['A', 'B']:
        ax[axn].set_ylabel("")
        ax[axn].grid()
        ax[axn].set_xticks(np.arange(4))
        ax[axn].set_xticklabels([0, 12, 24, 36])
    ax['A'].set_ylabel("Frequency")
    ax['B'].set_ylabel("Probability of developing\ndysplasia")
    ax['A'].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")

    ax['A'].legend(fontsize=28, frameon=False)

    ####################
    # Panel C and D
    ####################

    thisx = np.linspace(0.01, 25, 100)
    n_samples = 10

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_dysp']['par1'],
                                         genotype_pars[gtype]['dur_dysp']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax['C'].plot(thisx, rv.pdf(thisx), color=colors[gi], lw=2, label=gtype.upper())
        ax['D'].plot(thisx, ut.logf1(thisx, genotype_pars[gtype]['prog_rate']), color=colors[gi], lw=2,
                       label=gtype.upper())
        for year in range(1, 26):
            peaks = ut.logf1(year, hpu.sample(dist='normal', par1=genotype_pars[gtype]['prog_rate'],
                                              par2=genotype_pars[gtype]['prog_rate_sd'], size=n_samples))
            ax['D'].plot([year] * n_samples, peaks, color=colors[gi], lw=0, marker='o', alpha=0.5)

    ax['C'].set_ylabel("")
    # ax['C'].legend(fontsize=18)
    ax['C'].grid()
    ax['C'].set_ylabel("Frequency")
    ax['C'].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")

    # Severity
    ax['D'].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")
    ax['D'].set_ylabel("Clinical severity")
    ax['D'].set_ylim([0, 1])
    ax['D'].axhline(y=0.33, ls=':', c='k')
    ax['D'].axhline(y=0.67, ls=':', c='k')
    ax['D'].axhspan(0, 0.33, color=cmap[0], alpha=.4)
    ax['D'].axhspan(0.33, 0.67, color=cmap[1], alpha=.4)
    ax['D'].axhspan(0.67, 1, color=cmap[2], alpha=.4)
    ax['D'].text(-0.3, 0.08, 'CIN1', rotation=90)
    ax['D'].text(-0.3, 0.4, 'CIN2', rotation=90)
    ax['D'].text(-0.3, 0.73, 'CIN3', rotation=90)


    ####################
    # Panel E
    ####################

    # This section calculates the overall share of outcomes for people infected with each genotype
    dysp_shares = [] # Initialize the share of people who develop ANY dysplasia
    gtypes = []      # Initialize genotypes -- TODO, is this necessary?
    noneshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], [] # Initialize share by each outcome
    igi = 0.01 # Define the integration interval
    longx = sc.inclusiverange(0.01,80,igi) # Initialize a LONG array of years

    # Loop over genotypes
    for g in range(ng):

        # Firstly, determine shares of women who develop any dysplasia
        sigma, scale = ut.lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2']) # Calculate parameters in the format expected by scipy
        rv = lognorm(sigma, 0, scale) # Create scipy rv object
        aa = np.diff(rv.cdf(longx))  # Calculate the probability that a woman will have a pre-dysplasia duration in any of the subintervals of time spanning 0-25 years
        bb = ut.logf1(longx, dysp_rate[g])[1:] # Calculate the probablity of her developing dysplasia for a given duration
        dysp_shares.append(np.dot(aa, bb)) # Convolve the two above calculations to determine the probability of her developing dysplasia overall
        gtypes.append(genotype_map[g].replace('hpv', '')) # Store genotype names for labeling

    for g in range(ng):
        # Next, determine the outcomes for women who do develop dysplasia
        sigma, scale = ut.lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2']) # Calculate parameters in the format expected by scipy
        rv = lognorm(sigma, 0, scale) # Create scipy rv object
        peak_dysp = ut.logf1(longx, prog_rate[g]) # Calculate peak dysplasia

        # Find women who only advance to CIN1
        indcin1 = sc.findinds(peak_dysp < .33)[-1]
        cin1_share = rv.cdf(longx[indcin1]) - rv.cdf(longx[0])

        # See if there are women who advance to CIN2 and get their indices if so
        if (peak_dysp > .33).any():
            indcin2 = sc.findinds((peak_dysp > .33) & (peak_dysp < .67))[-1]
        else:
            indcin2 = indcin1
        cin2_share = rv.cdf(longx[indcin2]) - rv.cdf(longx[indcin1])

        # See if there are women who advance to CIN3, and get their indices if so
        cancer_share_of_cin3s = 0 # Initially assume no cancers, update later
        if (peak_dysp > .67).any():
            cin3_cancer_inds = sc.findinds((peak_dysp>0.67)) # This give the combined indices of those whose worst outcome is CIN3 and those whose worst outcome is cancer. We now need to separate these
            indcin3 = cin3_cancer_inds[-1] # Index after which people develop CIN3 (plus possibly cancer)

            # Calculate the share of these women who develop cancer
            years_with_dysp = longx[cin3_cancer_inds] # Figure out the total duration of dysplasia for women who develop CIN3
            years_with_cin3 = years_with_dysp - hpu.invlogf1(0.67, prog_rate[g]) # Figure out how many years they have CIN3 for (i.e., total dysp time minus time they developed CIN3, note this is not dt-dependent)
            cancer_probs = 1 - (1 - cancer_prob[g]) ** years_with_cin3 # Apply the annual probability of them developing cancer to each of the years they have CIN3
            cancer_inds = hpu.true(hpu.n_binomial(cancer_probs, len(cin3_cancer_inds))) # Use binomial probabilities to determine the indices of those who get cancer
            n_cin3_cancer = len(cin3_cancer_inds) # Number who get CIN3 + number who get cancer
            n_cancer = len(cancer_inds) # Number who get cancer
            cancer_share_of_cin3s = n_cancer/n_cin3_cancer # Share of CIN3/cancer women who get cancer

        else:
            indcin3 = indcin2

        cin3_cancer_share = rv.cdf(longx[indcin3]) - rv.cdf(longx[indcin2]) # Share who develop CIN3 as worst outcome + share who develop cancer as worst outcome
        cin3_share = cin3_cancer_share*(1-cancer_share_of_cin3s)
        cancer_share = cin3_cancer_share*cancer_share_of_cin3s

        noneshares.append(1 - dysp_shares[g])
        cin1shares.append(cin1_share * dysp_shares[g])
        cin2shares.append(cin2_share * dysp_shares[g])
        cin3shares.append(cin3_share * dysp_shares[g])
        cancershares.append(cancer_share * dysp_shares[g])


    # Final plot
    bottom = np.zeros(ng)
    all_shares = [noneshares,
                  cin1shares,
                  cin2shares,
                  cin3shares,
                  cancershares,
                  ]

    for gn, grade in enumerate(['No dysplasia', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        #if len(ydata.shape) > 1: ydata = ydata[:, 0]
        color = cmap[gn-1,:] if gn > 0 else 'gray'
        ax['E'].bar(np.arange(1, ng + 1), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata

    ax['E'].set_xticks(np.arange(1,ng + 1))
    ax['E'].set_xticklabels(glabels)
    ax['E'].set_ylabel("")
    ax['E'].set_ylabel("Distribution of outcomes")
    # ax['E'].legend(bbox_to_anchor=(1.1, 1))
    handles, labels = ax['E'].get_legend_handles_labels()

    ax['F'].set_axis_off()
    ax['F'].legend(handles, labels, bbox_to_anchor=(0.5, 1), frameon=False)

    fs=40
    pl.figtext(0.06, 0.975, 'A', fontweight='bold', fontsize=fs)
    pl.figtext(0.52, 0.975, 'B', fontweight='bold', fontsize=fs)
    pl.figtext(0.06, 0.62, 'C', fontweight='bold', fontsize=fs)
    pl.figtext(0.52, 0.62, 'D', fontweight='bold', fontsize=fs)
    pl.figtext(0.06, 0.3, 'E', fontweight='bold', fontsize=fs)
    # pl.figtext(0.69, 0.51, 'F', fontweight='bold', fontsize=30)

    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/fig4.png", dpi=100)

#%% Run as a script
if __name__ == '__main__':

    file = f'nigeria_pars.obj'
    calib_pars = sc.loadobj(file)

    plot_fig4(calib_pars)

    print('Done.')
