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

def lognorm_params(par1, par2):
    """
    Given the mean and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    mean = np.log(par1 ** 2 / np.sqrt(par2 ** 2 + par1 ** 2))  # Computes the mean of the underlying normal distribution
    sigma = np.sqrt(np.log(par2 ** 2 / par1 ** 2 + 1))  # Computes sigma for the underlying normal distribution

    scale = np.exp(mean)
    shape = sigma
    return shape, scale


def plot_fig4():

    # Group genotypes
    genotypes = ['hpv16', 'hpv18', 'hrhpv']
    sim = hpv.Sim(genotypes=genotypes)
    sim.initialize()

    # Get parameters
    ng = sim['n_genotypes']
    genotype_map = sim['genotype_map']

    # Get parameters
    genotype_pars = sim['genotype_pars']
    genotype_pars['hpv16']['dur_dysp']['par1'] = 6
    genotype_pars['hpv16']['dur_dysp']['par2'] = 9

    genotype_pars['hpv18']['dur_dysp']['par1'] = 5
    genotype_pars['hpv18']['dur_dysp']['par2'] = 9

    genotype_pars['hrhpv']['dur_dysp']['par1'] = 7
    genotype_pars['hrhpv']['dur_dysp']['par2'] = 10

    genotype_pars['hpv16']['dysp_infl'] = 15
    genotype_pars['hpv18']['dysp_infl'] = 16
    genotype_pars['hrhpv']['dysp_infl'] = 17

    genotype_pars['hpv16']['transform_prob'] = 0.00012
    genotype_pars['hpv18']['transform_prob'] = 0.0002
    genotype_pars['hrhpv']['transform_prob'] = 0.00032

    # Shorten names
    dur_dysp = [genotype_pars[genotype_map[g]]['dur_dysp'] for g in range(ng)]
    dysp_rate = [genotype_pars[genotype_map[g]]['dysp_rate'] for g in range(ng)]
    dysp_rate_sd = [genotype_pars[genotype_map[g]]['dysp_rate_sd'] for g in range(ng)]
    dysp_infl = [genotype_pars[genotype_map[g]]['dysp_infl'] for g in range(ng)]
    transform_probs = [genotype_pars[genotype_map[g]]['transform_prob'] for g in range(ng)]
    ####################
    # Make figure, set fonts and colors
    ####################
    ut.set_font(size=25)
    colors = sc.gridcolors(10)
    cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])
    fig, ax = pl.subplot_mosaic('AB;CD', figsize=(16, 16))

    ####################
    # Panel A and C
    ####################

    glabels = ['HPV16', 'HPV18', 'HRHPV']
    ####################
    # Make plots
    ####################

    thisx = np.linspace(1, 40, 40)
    n_samples = 10

    def cum_transform_prob(cp, x, dysp):
        dd = np.diff(dysp)
        n = len(x)
        result = [1 - np.product([((1 - cp) ** (100 * dd[i])) ** (j - i) for i in range(j)]) for j in range(n)]
        return result

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        sigma, scale = ut.lognorm_params(dur_dysp[gi]['par1'], dur_dysp[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        ax['A'].plot(thisx, rv.pdf(thisx), color=colors[gi], lw=2, label=glabels[gi])
        ax['C'].plot(thisx, ut.logf2(thisx, dysp_infl[gi], dysp_rate[gi]), color=colors[gi], lw=2, label=gtype.upper())

        for smpl in range(n_samples):
            dr = hpu.sample(dist='normal_pos', par1=dysp_rate[gi], par2=dysp_rate_sd[gi])
            ax['C'].plot(thisx, hpu.logf2(thisx, dysp_infl[gi], dr), color=colors[gi], lw=1, alpha=0.5, label=gtype.upper())

        tp = cum_transform_prob(transform_probs[gi], thisx, hpu.logf2(thisx, dysp_infl[gi], dysp_rate[gi]))
        ax['B'].plot(thisx, tp, color=colors[gi], label=gtype.upper())

    ax['A'].set_ylabel("")
    ax['A'].grid()
    ax['A'].set_xlabel("Duration of infection (years)")
    ax['A'].set_ylabel("Density")
    ax['A'].legend()

    ax['C'].set_ylabel("Degree of dysplasia")
    ax['C'].set_xlabel("Duration of infection (years)")
    ax['C'].set_ylim([0,1])
    ax['C'].grid()

    ax['C'].axhline(y=0.33, ls=':', c='k')
    ax['C'].axhline(y=0.67, ls=':', c='k')
    ax['C'].axhspan(0, 0.33, color=cmap[0], alpha=.4)
    ax['C'].axhspan(0.33, 0.67, color=cmap[1], alpha=.4)
    ax['C'].axhspan(0.67, 1, color=cmap[2], alpha=.4)
    ax['C'].text(-0.3, 0.08, 'CIN1', rotation=90)
    ax['C'].text(-0.3, 0.4, 'CIN2', rotation=90)
    ax['C'].text(-0.3, 0.73, 'CIN3', rotation=90)

    ax['B'].grid()
    ax['B'].set_ylabel("Probability of transformation")
    ax['B'].set_xlabel("Duration of infection (years)")

    ####################
    # Panel D
    ####################

    # This section calculates the overall share of outcomes for people infected with each genotype
    cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [] # Initialize the share of people who get dysplasia vs cancer

    # Loop over genotypes
    for g in range(ng):
        # First, determine the outcomes for women
        sigma, scale = ut.lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2']) # Calculate parameters in the format expected by scipy
        rv = lognorm(sigma, 0, scale) # Create scipy rv object
        tp = cum_transform_prob(transform_probs[g], thisx, hpu.logf2(thisx, dysp_infl[g], dysp_rate[g]))
        peak_dysp = hpu.logf2(thisx, dysp_infl[g], dysp_rate[g])  # Calculate peak dysplasia

        # To start find women who advance to cancer
        cancer_inds = hpu.true(hpu.n_binomial(tp, len(thisx)))  # Use binomial probabilities to determine the indices of those who get cancer

        # Find women who only advance to CIN1
        indcin1 = sc.findinds(peak_dysp < .33)[-1]
        n_cin1 = len(sc.findinds(peak_dysp < .33))
        cin1_share = rv.cdf(thisx[indcin1])

        # See if there are women who advance to CIN2 and get their indices if so
        if (peak_dysp > .33).any():
            n_cin2 = len(sc.findinds((peak_dysp > .33) & (peak_dysp < .67)))
            indcin2 = sc.findinds((peak_dysp > .33) & (peak_dysp < .67))[-1]
        else:
            n_cin2 = 0
            indcin2 = indcin1
        cin2_share = rv.cdf(thisx[indcin2]) - rv.cdf(thisx[indcin1])

        if (peak_dysp > .67).any():
            n_cin3 = len(sc.findinds(peak_dysp > .67))
            indcin3 = sc.findinds((peak_dysp > 0.67))[-1]  # Index after which people develop CIN3 (plus possibly cancer)
        else:
            n_cin3 = 0
            indcin3 = indcin2
        cin3_share = rv.cdf(thisx[indcin3]) - rv.cdf(thisx[indcin2])

        n_cancer_cin1 = len(np.intersect1d(cancer_inds, sc.findinds(peak_dysp < .33)))
        n_cancer_cin2 = len(np.intersect1d(cancer_inds, sc.findinds((peak_dysp > .33) & (peak_dysp < .67))))
        n_cancer_cin3 = len(np.intersect1d(cancer_inds, sc.findinds((peak_dysp > 0.67))))

        cancer_share_of_cin1s = n_cancer_cin1 / n_cin1  # Share of CIN1 women who get cancer
        cancer_share_of_cin2s = n_cancer_cin2 / n_cin2  # Share of CIN2 women who get cancer
        cancer_share_of_cin3s = n_cancer_cin3 / n_cin3  # Share of CIN3 women who get cancer

        cin1_share *= 1 - cancer_share_of_cin1s
        cin2_share *= 1 - cancer_share_of_cin2s
        cin3_share *= 1 - cancer_share_of_cin3s
        cancer_share = 1 - (cin1_share + cin2_share + cin3_share)

        cin1shares.append(cin1_share)
        cin2shares.append(cin2_share)
        cin3shares.append(cin3_share)
        cancershares.append(cancer_share)

    # Final plot
    bottom = np.zeros(ng)
    all_shares = [cin1shares,
                  cin2shares,
                  cin3shares,
                  cancershares
                  ]

    for gn, grade in enumerate(['HPV/CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        color = cmap[gn,:]
        ax['D'].bar(np.arange(1, ng + 1), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata

    ax['D'].set_xticks(np.arange(1,ng + 1))
    ax['D'].set_xticklabels(glabels)
    ax['D'].set_ylabel("")
    ax['D'].set_ylabel("Distribution of outcomes")
    # ax['E'].legend(bbox_to_anchor=(1.1, 1))
    handles, labels = ax['D'].get_legend_handles_labels()
    ax['D'].legend(handles, labels, frameon=True, loc='lower right')

    fs=40
    pl.figtext(0.02, 0.955, 'A', fontweight='bold', fontsize=fs)
    pl.figtext(0.51, 0.955, 'C', fontweight='bold', fontsize=fs)
    pl.figtext(0.02, 0.47, 'B', fontweight='bold', fontsize=fs)
    pl.figtext(0.51, 0.47, 'D', fontweight='bold', fontsize=fs)
    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/fig4.png", dpi=100)


#%% Run as a script
if __name__ == '__main__':


    plot_fig4()

    print('Done.')
