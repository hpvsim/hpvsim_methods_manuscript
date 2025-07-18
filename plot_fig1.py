"""
This script produces figure 1 of the HPVsim methods paper
"""

import hpvsim as hpv
# import hpvsim.parameters as hppar
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
        dur_precin += sim.pars[genotype]['dur_precin']
        dur_cin += sim.pars[genotype]['dur_cin']
        cancer_fns += sc.mergedicts(sim.pars[genotype]['cin_fn'], sim.pars[genotype]['cancer_fn'])
        cin_fns += sim.pars[genotype]['cin_fn']

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
    years = np.arange(1, 16, 1)
    this_cinx = np.arange(dt, 30+dt, dt)

    width = .2
    multiplier = 0

    # Durations and severity of dysplasia
    for gi, gtype in enumerate(genotypes):
        offset = width * multiplier

        # Panel A: durations of infection
        # axes[0].set_ylim([0,1])
        mean = sim.pars[genotype]['dur_precin'].pars['mean'].v # mean duration of infection
        std = sim.pars[genotype]['dur_precin'].pars['std'].v # std. dev. of infection duration
        sigma, scale = ut.lognorm_params(mean, std)
        rv = lognorm(sigma, 0, scale)

        axes[0].bar(years+offset - width/3, rv.pdf(years), color=colors[gi], lw=2, label=glabels[gi], width=width)
        multiplier += 1

        # Panel B: prob of dysplasia
        dysp = hpv.logf2(this_precinx[:], **cin_fns[gi])
        axes[1].plot(this_precinx, dysp, color=colors[gi], lw=2, label=gtype.upper())

        # Panel C: durations of CIN
        mean = sim.pars[genotype]['dur_cin'].pars['mean'].v # mean duration of infection
        std = sim.pars[genotype]['dur_cin'].pars['std'].v # std. dev. of infection duration
        sigma, scale = ut.lognorm_params(mean, std)
        rv = lognorm(sigma, 0, scale)
        axes[2].plot(this_cinx, rv.pdf(this_cinx), color=colors[gi], lw=2, label=glabels[gi])

        # Panel D: cancer
        cancer = hpv.compute_cancer_prob(this_cinx[:], pars=cancer_fns[gi])
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
    sc.savefig('figures/fig1.png')

    return


# %% Run as a script
if __name__ == '__main__':

    sim = hpv.Sim()
    sim.init()
    plot_nh_simple(sim)

    print('Done.')