"""
This script produces figure 1 of the HPVsim methods paper
"""

import hpvsim as hpv
import hpvsim.parameters as hppar
import sciris as sc
import pylab as pl
import utils as ut
import numpy as np
from scipy.stats import lognorm


def plot_nh_simple(sim=None):

    genotypes = ['hpv16', 'hpv18', 'hi5', 'ohr']
    glabels = ['HPV16', 'HPV18', 'Hi5', 'OHR']

    dur_cin = sc.autolist()
    cancer_fns = sc.autolist()
    cin_fns = sc.autolist()
    dur_precin = sc.autolist()
    for gi, genotype in enumerate(genotypes):
        dur_precin += sim['genotype_pars'][genotype]['dur_precin']
        dur_cin += sim['genotype_pars'][genotype]['dur_cin']
        cancer_fns += sim['genotype_pars'][genotype]['cancer_fn']
        cin_fns += sim['genotype_pars'][genotype]['cin_fn']

    ####################
    # Make figure, set fonts and colors
    ####################
    colors = sc.gridcolors(len(genotypes))
    ut.set_font(size=16)
    fig, axes = pl.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    ####################
    # Make plots
    ####################
    dt = 0.25
    this_precinx = np.arange(dt, 15+dt, dt)
    years = np.arange(1,16,1)
    this_cinx = np.arange(dt, 30+dt, dt)

    width = .2
    multiplier = 0

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        offset = width * multiplier

        # Panel A: durations of infection
        # axes[0].set_ylim([0,1])
        sigma, scale = ut.lognorm_params(dur_precin[gi]['par1'], dur_precin[gi]['par2'])
        rv = lognorm(sigma, 0, scale)

        axes[0].bar(years+offset - width/3, rv.pdf(years), color=colors[gi], lw=2, label=glabels[gi], width=width)
        multiplier += 1
        # Panel B: prob of dysplasia
        dysp = hppar.compute_severity(this_precinx[:], pars=cin_fns[gi])
        axes[1].plot(this_precinx, dysp, color=colors[gi], lw=2, label=gtype.upper())

        # Panel C: durations of CIN
        sigma, scale = ut.lognorm_params(dur_cin[gi]['par1'], dur_cin[gi]['par2'])
        rv = lognorm(sigma, 0, scale)
        axes[2].plot(this_cinx, rv.pdf(this_cinx), color=colors[gi], lw=2, label=glabels[gi])

        # Panel D: cancer
        cancer = hppar.compute_severity(this_cinx[:], pars=sc.mergedicts(cin_fns[gi], cancer_fns[gi]))
        axes[3].plot(this_cinx, cancer, color=colors[gi], lw=2, label=gtype.upper())

    axes[0].set_ylabel("")
    axes[0].grid()
    axes[0].set_xlabel("Duration of infection (years)")
    axes[0].set_title("(A) Probability of persistance")
    axes[0].legend(frameon=False)

    axes[1].set_ylabel("Probability of CIN")
    axes[1].set_xlabel("Duration of infection (years)")
    axes[1].set_title("(B) Probability that an infection of at least\nX years will lead to high-grade lesions")
    axes[1].set_ylim([0,1])
    axes[1].grid()

    axes[2].set_ylabel("")
    axes[2].grid()
    axes[2].set_xlabel("Duration of high-grade lesions (years)")
    axes[2].set_title("(C) Distribution of high-grade lesion duration")
    axes[2].legend(frameon=False)

    axes[3].set_ylim([0,1])
    axes[3].grid()
    axes[3].set_xlabel("Duration of CIN (years)")
    axes[3].set_title("(D) Probability that a high-grade lesion of \nat least X years will eventuate in cancer")

    fig.tight_layout()
    fig.savefig('../figures/fig1.png')
    fig.show()
    return


# %% Run as a script
if __name__ == '__main__':

    sim = hpv.Sim(genotypes=[16, 18, 'hi5', 'ohr'])
    sim.initialize()
    plot_nh_simple(sim)

    print('Done.')