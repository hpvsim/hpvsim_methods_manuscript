"""
This script produces figure 4 of the HPVsim methods paper, showing the natural history
"""

# Standard imports
import numpy as np
from scipy.stats import lognorm
import sciris as sc
import pylab as pl

# HPVsim and local imports
import hpvsim as hpv
import hpvsim.parameters as hppar
import utils as ut


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
    clinical_cutoffs = sim['clinical_cutoffs']
    clinical_cutoffs['precin'] = 0.03

    # Shorten names
    dur_episomal = [genotype_pars[genotype_map[g]]['dur_episomal'] for g in range(ng)]
    # sev_rate = [genotype_pars[genotype_map[g]]['sev_fn']['k'] for g in range(ng)]
    # sev_rate_sd = [genotype_pars[genotype_map[g]]['sev_rate_sd'] for g in range(ng)]
    sev_fn = [genotype_pars[genotype_map[g]]['sev_fn'] for g in range(ng)]
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
        sigma, scale = ut.lognorm_params(dur_episomal[gi]['par1'], dur_episomal[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        ax['A'].plot(thisx, rv.pdf(thisx), color=colors[gi], lw=2, label=glabels[gi])
        ax['C'].plot(thisx, hppar.compute_severity(thisx, pars=sev_fn[gi]), color=colors[gi], lw=2, label=gtype.upper())

        # for smpl in range(n_samples):
        #     dr = hpv.sample(dist='normal_pos', par1=sev_rate[gi], par2=sev_rate_sd[gi])
        #     ax['C'].plot(thisx, ut.logf2(thisx, sev_infl[gi], dr), color=colors[gi], lw=1, alpha=0.5, label=gtype.upper())

        tp = cum_transform_prob(transform_probs[gi], thisx, hppar.compute_severity(thisx, pars=sev_fn[gi]))
        ax['B'].plot(thisx, tp, color=colors[gi], label=gtype.upper())

    ax['A'].set_ylabel("")
    ax['A'].grid()
    ax['A'].set_xlabel("Duration of infection (years)")
    ax['A'].set_ylabel("Density")
    ax['A'].legend()

    ax['C'].set_ylabel("Severity of infection")
    ax['C'].set_xlabel("Duration of infection (years)")
    ax['C'].set_ylim([0,1])
    ax['C'].grid()

    ax['C'].axhline(y=clinical_cutoffs['precin'], ls=':', c='k')
    ax['C'].axhline(y=clinical_cutoffs['cin1'], ls=':', c='k')
    ax['C'].axhline(y=clinical_cutoffs['cin2'], ls=':', c='k')
    ax['C'].axhspan(0, clinical_cutoffs['precin'], color='gray', alpha=.4)
    ax['C'].axhspan(clinical_cutoffs['precin'], clinical_cutoffs['cin1'], color=cmap[0], alpha=.4)
    ax['C'].axhspan(clinical_cutoffs['cin1'], clinical_cutoffs['cin2'], color=cmap[1], alpha=.4)
    ax['C'].axhspan(clinical_cutoffs['cin2'], 1.0, color=cmap[2], alpha=.4)
    ax['C'].text(-0.3, 0.15, 'CIN1', rotation=90)
    ax['C'].text(-0.3, 0.48, 'CIN2', rotation=90)
    ax['C'].text(-0.3, 0.8, 'CIN3', rotation=90)

    ax['B'].grid()
    ax['B'].set_ylabel("Probability of transformation")
    ax['B'].set_xlabel("Duration of infection (years)")

    ####################
    # Panel D
    ####################

    # This section calculates the overall share of outcomes for people infected with each genotype
    precinshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], [] # Initialize the share of people who get dysplasia vs cancer

    # Loop over genotypes
    for g in range(ng):
        # First, determine the outcomes for women
        sigma, scale = ut.lognorm_params(dur_episomal[g]['par1'], dur_episomal[g]['par2']) # Calculate parameters in the format expected by scipy
        rv = lognorm(sigma, 0, scale) # Create scipy rv object
        tp = cum_transform_prob(transform_probs[g], thisx, hppar.compute_severity(thisx, pars=sev_fn[g]))
        peak_dysp = hppar.compute_severity(thisx, pars=sev_fn[g])  # Calculate peak dysplasia

        # To start find women who advance to cancer
        cancer_inds = hpv.true(hpv.n_binomial(tp, len(thisx)))  # Use binomial probabilities to determine the indices of those who get cancer


        # Find women who only advance to PRECIN
        indprecin = sc.findinds(peak_dysp < clinical_cutoffs['precin'])[-1]
        n_precin = len(sc.findinds(peak_dysp < clinical_cutoffs['precin']))
        precin_share = rv.cdf(thisx[indprecin])

        # Find women who only advance to CIN1
        indcin1 = sc.findinds((peak_dysp > clinical_cutoffs['precin']) & (peak_dysp < clinical_cutoffs['cin1']))[-1]
        n_cin1 = len(sc.findinds((peak_dysp > clinical_cutoffs['precin']) & (peak_dysp < clinical_cutoffs['cin1'])))
        cin1_share = rv.cdf(thisx[indcin1]) - rv.cdf(thisx[indprecin])

        # See if there are women who advance to CIN2 and get their indices if so
        if (peak_dysp > clinical_cutoffs['cin1']).any():
            n_cin2 = len(sc.findinds((peak_dysp > clinical_cutoffs['cin1']) & (peak_dysp < clinical_cutoffs['cin2'])))
            indcin2 = sc.findinds((peak_dysp > clinical_cutoffs['cin1']) & (peak_dysp < clinical_cutoffs['cin2']))[-1]
        else:
            n_cin2 = 0
            indcin2 = indcin1
        cin2_share = rv.cdf(thisx[indcin2]) - rv.cdf(thisx[indcin1])

        if (peak_dysp > clinical_cutoffs['cin2']).any():
            n_cin3 = len(sc.findinds(peak_dysp > clinical_cutoffs['cin2']))
            indcin3 = sc.findinds((peak_dysp > clinical_cutoffs['cin2']))[-1]  # Index after which people develop CIN3 (plus possibly cancer)
        else:
            n_cin3 = 0
            indcin3 = indcin2
        cin3_share = rv.cdf(thisx[indcin3]) - rv.cdf(thisx[indcin2])


        n_cancer_precin= len(np.intersect1d(cancer_inds, sc.findinds(peak_dysp < clinical_cutoffs['precin'])))
        n_cancer_cin1 = len(np.intersect1d(cancer_inds, sc.findinds((peak_dysp > clinical_cutoffs['precin']) & (peak_dysp < clinical_cutoffs['cin1']))))
        n_cancer_cin2 = len(np.intersect1d(cancer_inds, sc.findinds((peak_dysp > clinical_cutoffs['cin1']) & (peak_dysp < clinical_cutoffs['cin2']))))
        n_cancer_cin3 = len(np.intersect1d(cancer_inds, sc.findinds((peak_dysp > clinical_cutoffs['cin2']))))

        cancer_share_of_precins = n_cancer_precin/n_precin
        cancer_share_of_cin1s = n_cancer_cin1 / n_cin1  # Share of CIN1 women who get cancer
        cancer_share_of_cin2s = n_cancer_cin2 / n_cin2  # Share of CIN2 women who get cancer
        cancer_share_of_cin3s = n_cancer_cin3 / n_cin3  # Share of CIN3 women who get cancer

        precin_share *= 1 - cancer_share_of_precins
        cin1_share *= 1 - cancer_share_of_cin1s
        cin2_share *= 1 - cancer_share_of_cin2s
        cin3_share *= 1 - cancer_share_of_cin3s
        cancer_share = 1 - (precin_share + cin1_share + cin2_share + cin3_share)

        precinshares.append(precin_share)
        cin1shares.append(cin1_share)
        cin2shares.append(cin2_share)
        cin3shares.append(cin3_share)
        cancershares.append(cancer_share)

    # Final plot
    bottom = np.zeros(ng)
    all_shares = [precinshares,
                  cin1shares,
                  cin2shares,
                  cin3shares,
                  cancershares
                  ]

    for gn, grade in enumerate(['Pre-CIN', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        color = cmap[gn - 1, :] if gn > 0 else 'gray'
        ax['D'].bar(np.arange(1, ng + 1), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata

    ax['D'].set_xticks(np.arange(1,ng + 1))
    ax['D'].set_xticklabels(glabels)
    ax['D'].set_ylabel("")
    ax['D'].set_ylabel("Distribution of outcomes")
    handles, labels = ax['D'].get_legend_handles_labels()
    ax['D'].legend(handles, labels, frameon=True, loc='lower right')

    fs = 40
    pl.figtext(0.02, 0.955, 'A', fontweight='bold', fontsize=fs)
    pl.figtext(0.51, 0.955, 'C', fontweight='bold', fontsize=fs)
    pl.figtext(0.02, 0.47, 'B', fontweight='bold', fontsize=fs)
    pl.figtext(0.51, 0.47, 'D', fontweight='bold', fontsize=fs)
    fig.tight_layout()
    pl.savefig(f"../{ut.figfolder}/fig4.png", dpi=100)
    pl.show()


#%% Run as a script
if __name__ == '__main__':


    plot_fig4()

    print('Done.')
