"""
This script produces figure 4 of the HPVsim methods paper, showing the natural history
"""
import hpvsim as hpv
import hpvsim.utils as hpu
import pylab as pl
import pandas as pd
from scipy.stats import lognorm
import numpy as np
import sciris as sc

import utils as ut

#%% Preliminary definitions

# Group genotypes
genotypes = ['hpv16', 'hpv18', 'hrhpv']


# Create sim to get baseline prognoses parameters
sim = hpv.Sim(genotypes=genotypes)
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
cancer_prob = [genotype_pars[genotype_map[g]]['cancer_prob'] for g in range(ng)]

# Set common attributes
colors = sc.gridcolors(ng)
n_samples = 10
cmap = pl.cm.Oranges([0.25, 0.5, 0.75, 1])


#%% Plotting function
def plot_fig4():

    ut.set_font(size=20)
    fig, ax = pl.subplots(2, 3, figsize=(16, 10))
    pn = 0
    x = np.linspace(0.01, 2, 200)

    for gtype in genotypes:
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_precin']['par1'],
                                         genotype_pars[gtype]['dur_precin']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, 0].plot(x, rv.pdf(x), color=colors[pn], lw=2, label=gtype.upper())
        ax[1, 0].plot(x, ut.logf1(x, genotype_pars[gtype]['dysp_rate']), color=colors[pn], lw=2, label=gtype.upper())
        pn += 1

        ax[1, 0].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")
        for row in [0, 1]:
            ax[row, 0].set_ylabel("")
            ax[row, 0].grid()
            ax[row, 0].set_xticks([0, 0.5, 1.0, 1.5, 2.0])
            ax[row, 0].set_xticklabels([0, 6, 12, 18, 24])
        ax[1, 0].set_ylim([0, .99])
        ax[0, 0].set_ylim([0, 1.8])
    ax[0, 0].set_ylabel("Frequency")
    ax[1, 0].set_ylabel("Probability of developing\ndysplasia")
    ax[0, 0].set_xlabel("Duration of infection prior to\ncontrol/clearance/dysplasia (months)")

    pn = 0
    thisx = np.linspace(0.01, 20, 100)

    # Durations and severity of dysplasia
    for gtype in genotypes:
        ai=1
        sigma, scale = ut.lognorm_params(genotype_pars[gtype]['dur_dysp']['par1'],
                                         genotype_pars[gtype]['dur_dysp']['par2'])
        rv = lognorm(sigma, 0, scale)
        ax[0, ai].plot(thisx, rv.pdf(thisx), color=colors[pn], lw=2, label=gtype.upper())
        ax[1, ai].plot(thisx, ut.logf1(thisx, genotype_pars[gtype]['prog_rate']), color=colors[pn], lw=2,
                       label=gtype.upper())
        for year in range(1, 21):
            peaks = ut.logf1(year, hpu.sample(dist='normal', par1=genotype_pars[gtype]['prog_rate'],
                                              par2=genotype_pars[gtype]['prog_rate_sd'], size=n_samples))
            if pn == 1:
                ax[1, ai].plot([year] * n_samples, peaks, color=colors[pn], lw=0, marker='o', alpha=0.5)
        pn += 1

        ax[0, ai].set_ylabel("")
        ax[0, ai].legend(fontsize=18)
        ax[0, ai].grid()
        ax[0, ai].set_ylabel("Frequency")
        ax[0, ai].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")

        # Severity
        ax[1, ai].set_xlabel("Duration of dysplasia prior to\nregression/cancer (years)")
        ax[1, ai].set_ylabel("Clinical severity")
        ax[1, ai].set_ylim([0, 1])
        ax[1, ai].axhline(y=0.33, ls=':', c='k')
        ax[1, ai].axhline(y=0.67, ls=':', c='k')
        ax[1, ai].axhspan(0, 0.33, color=cmap[0], alpha=.4)
        ax[1, ai].axhspan(0.33, 0.67, color=cmap[1], alpha=.4)
        ax[1, ai].axhspan(0.67, 1, color=cmap[2], alpha=.4)
        ax[1, ai].text(-0.3, 0.08, 'CIN1', rotation=90)
        ax[1, ai].text(-0.3, 0.4, 'CIN2', rotation=90)
        ax[1, ai].text(-0.3, 0.73, 'CIN3', rotation=90)

    # pl.figtext(0.06, 0.94, 'A', fontweight='bold', fontsize=30)
    # pl.figtext(0.375, 0.94, 'B', fontweight='bold', fontsize=30)
    # pl.figtext(0.69, 0.94, 'C', fontweight='bold', fontsize=30)
    # pl.figtext(0.06, 0.51, 'D', fontweight='bold', fontsize=30)
    # pl.figtext(0.375, 0.51, 'E', fontweight='bold', fontsize=30)
    # pl.figtext(0.69, 0.51, 'F', fontweight='bold', fontsize=30)

    shares = []
    gtypes = []
    noneshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], []
    longx = np.linspace(0.01, 12, 1000)
    for g in range(ng):
        sigma, scale = ut.lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        aa = np.diff(rv.cdf(longx))
        bb = ut.logf1(longx, dysp_rate[g])[1:]
        shares.append(np.dot(aa, bb))
        gtypes.append(genotype_map[g].replace('hpv', ''))

    for g in range(ng):
        sigma, scale = ut.lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        dd = ut.logf1(longx, prog_rate[g])
        indcin1 = sc.findinds(dd < .33)[-1]
        if (dd > .33).any():
            indcin2 = sc.findinds((dd > .33) & (dd < .67))[-1]
        else:
            indcin2 = indcin1
        if (dd > .67).any():
            indcin3 = sc.findinds((dd > .67) & (dd < 1))[-1]
        else:
            indcin3 = indcin2

        if indcin3:
            indcancer = len(hpu.true(hpu.n_binomial(cancer_prob[g], indcin3)))
        else:
            indcancer = indcin3

        noneshares.append(1 - shares[g])
        cin1shares.append(((rv.cdf(longx[indcin1]) - rv.cdf(longx[0])) * shares[g]))
        cin2shares.append(((rv.cdf(longx[indcin2]) - rv.cdf(longx[indcin1])) * shares[g]))
        cin3shares.append(((rv.cdf(longx[indcin3]) - rv.cdf(longx[indcin2])) * shares[g]))
        cancershares.append(((rv.cdf(longx[indcancer]) - rv.cdf(longx[indcin3])) * shares[g]))

    # create dataframes
    years = np.arange(1, 13)
    cin1_shares, cin2_shares, cin3_shares, cancer_shares = [], [], [], []
    all_years = []
    all_genotypes = []
    for g in range(ng):
        for year in years:
            peaks = ut.logf1(year, hpu.sample(dist='normal', par1=prog_rate[g], par2=prog_rate_sd[g], size=n_samples))
            cin1_shares.append(sum(peaks < 0.33) / n_samples)
            cin2_shares.append(sum((peaks > 0.33) & (peaks < 0.67)) / n_samples)
            cin3_shares.append(sum((peaks > 0.67) & (peaks < 1)) / n_samples)
            cancer_shares.append(0)
            all_years.append(year)
            all_genotypes.append(genotype_map[g].replace('hpv', ''))
    data = {'Year': all_years, 'Genotype': all_genotypes, 'CIN1': cin1_shares, 'CIN2': cin2_shares, 'CIN3': cin3_shares,
            'Cancer': cancer_shares}
    sharesdf = pd.DataFrame(data)

    ai = 2
    ###### Share of women who develop each CIN grade
    loc_array = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
    w = 0.07
    for y in years:
        la = loc_array[y - 1] * w + np.sign(loc_array[y - 1]) * (-1) * w / 2
        bottom = np.zeros(ng)
        for gn, grade in enumerate(['CIN1', 'CIN2', 'CIN3', 'Cancer']):
            ydata = sharesdf[sharesdf['Year'] == y][grade]
            ax[0,ai].bar(np.arange(1, ng + 1) + la, ydata, width=w, color=cmap[gn], bottom=bottom, edgecolor='k', label=grade)
            bottom = bottom + ydata

    ax[0,ai].set_title("Share of women with dysplasia by clinical grade, duration, and genotype")
    ax[0,ai].set_xlabel("")
    ax[0,ai].set_xticks(np.arange(ng) + 1)
    ax[0,ai].set_xticklabels(gtypes)

    ##### Final outcomes for women
    bottom = np.zeros(ng + 1)
    all_shares = [noneshares + [sum([j * 1 / ng for j in noneshares])],
                  cin1shares + [sum([j * 1 / ng for j in cin1shares])],
                  cin2shares + [sum([j * 1 / ng for j in cin2shares])],
                  cin3shares + [sum([j * 1 / ng for j in cin3shares])],
                  cancershares + [sum([j * 1 / ng for j in cancershares])],
                  ]
    for gn, grade in enumerate(['No dysplasia', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        if len(ydata.shape) > 1: ydata = ydata[:, 0]
        color = cmap[gn - 1] if gn > 0 else 'gray'
        ax[1,ai].bar(np.arange(1, ng + 2), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata
    ax[1,ai].set_xticks(np.arange(ng + 1) + 1)
    ax[1,ai].set_xticklabels(gtypes + ['Average'])
    ax[1,ai].set_ylabel("")
    ax[1,ai].set_title("Eventual outcomes for women\n")
    ax[1,ai].legend()

    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/fig4.png", dpi=100)



def plot_fig6():

    shares = []
    gtypes = []
    noneshares, cin1shares, cin2shares, cin3shares, cancershares = [], [], [], [], []
    longx = np.linspace(0.01, 12, 1000)
    for g in range(ng):
        sigma, scale = ut.lognorm_params(dur_precin[g]['par1'], dur_precin[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        aa = np.diff(rv.cdf(longx))
        bb = ut.logf1(longx, dysp_rate[g])[1:]
        shares.append(np.dot(aa, bb))
        gtypes.append(genotype_map[g].replace('hpv', ''))

    for g in range(ng):
        sigma, scale = ut.lognorm_params(dur_dysp[g]['par1'], dur_dysp[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        dd = ut.logf1(longx, prog_rate[g])
        indcin1 = sc.findinds(dd < .33)[-1]
        if (dd > .33).any():
            indcin2 = sc.findinds((dd > .33) & (dd < .67))[-1]
        else:
            indcin2 = indcin1
        if (dd > .67).any():
            indcin3 = sc.findinds((dd > .67) & (dd < 1))[-1]
        else:
            indcin3 = indcin2
        if (dd > 1).any():
            indcancer = sc.findinds(dd > 1)[-1]
        else:
            indcancer = indcin3

        noneshares.append(1 - shares[g])
        cin1shares.append(((rv.cdf(longx[indcin1]) - rv.cdf(longx[0])) * shares[g]))
        cin2shares.append(((rv.cdf(longx[indcin2]) - rv.cdf(longx[indcin1])) * shares[g]))
        cin3shares.append(((rv.cdf(longx[indcin3]) - rv.cdf(longx[indcin2])) * shares[g]))
        cancershares.append(((rv.cdf(longx[indcancer]) - rv.cdf(longx[indcin3])) * shares[g]))

    # create dataframes
    years = np.arange(1, 13)
    cin1_shares, cin2_shares, cin3_shares, cancer_shares = [], [], [], []
    all_years = []
    all_genotypes = []
    for g in range(ng):
        for year in years:
            peaks = ut.logf1(year, hpu.sample(dist='normal', par1=prog_rate[g], par2=prog_rate_sd[g], size=n_samples))
            cin1_shares.append(sum(peaks < 0.33) / n_samples)
            cin2_shares.append(sum((peaks > 0.33) & (peaks < 0.67)) / n_samples)
            cin3_shares.append(sum((peaks > 0.67) & (peaks < 1)) / n_samples)
            cancer_shares.append(0)
            all_years.append(year)
            all_genotypes.append(genotype_map[g].replace('hpv', ''))
    data = {'Year': all_years, 'Genotype': all_genotypes, 'CIN1': cin1_shares, 'CIN2': cin2_shares, 'CIN3': cin3_shares,
            'Cancer': cancer_shares}
    sharesdf = pd.DataFrame(data)

    ut.set_font(size=20)
    fig, ax = pl.subplots(2, 1, figsize=(16, 8))

    ###### Share of women who develop each CIN grade
    loc_array = np.array([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6])
    w = 0.07
    for y in years:
        la = loc_array[y - 1] * w + np.sign(loc_array[y - 1])*(-1)*w/2
        bottom = np.zeros(ng)
        for gn, grade in enumerate(['CIN1', 'CIN2', 'CIN3', 'Cancer']):
            ydata = sharesdf[sharesdf['Year']==y][grade]
            ax[0].bar(np.arange(1,ng+1)+la, ydata, width=w, color=cmap[gn], bottom=bottom, edgecolor='k', label=grade);
            bottom = bottom + ydata

    # ax[1,1].legend()
    ax[0].set_title("Share of women with dysplasia by clinical grade, duration, and genotype")
    ax[0].set_xlabel("")
    ax[0].set_xticks(np.arange(ng) + 1)
    ax[0].set_xticklabels(gtypes)

    ##### Final outcomes for women
    bottom = np.zeros(ng+1)
    all_shares = [noneshares+[sum([j*1/ng for j in noneshares])],
                  cin1shares+[sum([j*1/ng for j in cin1shares])],
                  cin2shares+[sum([j*1/ng for j in cin2shares])],
                  cin3shares+[sum([j*1/ng for j in cin3shares])],
                  cancershares+[sum([j*1/ng for j in cancershares])],
                  ]
    for gn,grade in enumerate(['None', 'CIN1', 'CIN2', 'CIN3', 'Cancer']):
        ydata = np.array(all_shares[gn])
        if len(ydata.shape)>1: ydata = ydata[:,0]
        color = cmap[gn-1] if gn>0 else 'gray'
        ax[1].bar(np.arange(1,ng+2), ydata, color=color, bottom=bottom, label=grade)
        bottom = bottom + ydata
    ax[1].set_xticks(np.arange(ng+1) + 1)
    ax[1].set_xticklabels(gtypes+['Average'])
    ax[1].set_ylabel("")
    ax[1].set_title("Eventual outcomes for women\n")
    ax[1].legend(bbox_to_anchor =(0.5, 1.2),loc='upper center',ncol=5,frameon=False)

    #
    # plt.figtext(0.04, 0.85, 'A', fontweight='bold', fontsize=30)
    # plt.figtext(0.51, 0.85, 'B', fontweight='bold', fontsize=30)

    fig.tight_layout()
    pl.savefig(f"{ut.figfolder}/fig6.png", dpi=100)


#%% Run as a script
if __name__ == '__main__':

    plot_fig4()
    # plot_fig5()
    # plot_fig6()


    print('Done.')
