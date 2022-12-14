"""
This script produces figure 4 of the HPVsim methods paper, showing the natural history
"""
import hpvsim as hpv
import pylab as pl
from scipy.stats import lognorm
import numpy as np
import sciris as sc

import utils as ut

#%% Preliminary definitions

# Group genotypes
HR = ['hpv16', 'hpv18']
OHR = ['hpv31', 'hpv33', 'hpv35', 'hpv45', 'hpv51', 'hpv52', 'hpv56', 'hpv58']
LR = ['hpv6', 'hpv11']
alltypes = [HR, OHR, LR]

# Create sim to get baseline prognoses parameters
sim = hpv.Sim(genotypes='all')
sim.initialize()

# Get parameters
ng = sim['n_genotypes']
genotype_pars = sim['genotype_pars']
genotype_map = sim['genotype_map']

# Shorten duration names
dur_precin = [genotype_pars[genotype_map[g]]['dur_precin'] for g in range(ng)]
dur_dysp = [genotype_pars[genotype_map[g]]['dur_dysp'] for g in range(ng)]
dysp_rate = [genotype_pars[genotype_map[g]]['dysp_rate'] for g in range(ng)]
prog_rate = [genotype_pars[genotype_map[g]]['prog_rate'] for g in range(ng)]
prog_rate_sd = [genotype_pars[genotype_map[g]]['prog_rate_sd'] for g in range(ng)]


#%% Plotting function
def plot_fig4():

    ut.set_font(size=20)
    fig, ax = pl.subplots(2, 3, figsize=(16, 10))
    pn = 0
    x = np.linspace(0.01, 2, 200)
    colors = sc.gridcolors(ng)

    # Loop over genotypes, each one on its own plot
    for ai, gtypes in enumerate(alltypes):
        for gtype in gtypes:
            sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_precin']['par1'],
                                             genotype_pars[gtype]['dur_precin']['par2'])
            rv = lognorm(sigma, 0, scale)
            ax[0, ai].plot(x, rv.pdf(x), color=colors[pn], lw=2, label=gtype.upper())
            ax[1, ai].plot(x, ut.logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[pn], lw=2, label=gtype.upper())
            pn += 1

        ax[0, ai].legend(fontsize=16)
        ax[1, ai].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")
        for row in [0, 1]:
            ax[row, ai].set_ylabel("")
            ax[row, ai].grid()
            ax[row, ai].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
            ax[row, ai].set_xticklabels([0, 6, 12, 18, 24])
        ax[1, ai].set_ylim([0, .99])
        ax[0, ai].set_ylim([0, 1.8])
    ax[0, 0].set_ylabel("Frequency")
    ax[1, 0].set_ylabel("Probability of developing\ndysplasia")

    pl.figtext(0.06, 0.94, 'A', fontweight='bold', fontsize=30)
    pl.figtext(0.375, 0.94, 'B', fontweight='bold', fontsize=30)
    pl.figtext(0.69, 0.94, 'C', fontweight='bold', fontsize=30)
    pl.figtext(0.06, 0.51, 'D', fontweight='bold', fontsize=30)
    pl.figtext(0.375, 0.51, 'E', fontweight='bold', fontsize=30)
    pl.figtext(0.69, 0.51, 'F', fontweight='bold', fontsize=30)

    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/fig4.png", dpi=100)


#%% Run as a script
if __name__ == '__main__':

    plot_fig4()


    print('Done.')
