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
def make_tabS2():

    sim = hpv.Sim(genotypes=[16,18,31,33,35,45,51,52,56,58,'hrhpv'])
    sim.initialize()

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

    igi = 0.01 # Define the integration interval
    x = sc.inclusiverange(0.01,11,igi) # Initialize array of years
    shares = sc.objdict()
    shares['6m'] = sc.autolist()
    shares['12m'] = sc.autolist()
    shares['24m'] = sc.autolist()
    shares['120m'] = sc.autolist()
    output = []

    # Output values for Table S2
    for g,gname in enumerate(sim['genotypes']):
        sigma, scale = ut.lognorm_params(dur_precin[g]['par1'],
                                         dur_precin[g]['par2'])
        rv = lognorm(sigma, 0, scale)
        aa = np.diff(rv.cdf(x))  # Calculate the probability that a woman will have a pre-dysplasia duration in any of the subintervals of time spanning 0-25 years
        bb = ut.logf1(x, dysp_rate[g])[1:] # Calculate the probablity of her developing dysplasia for a given duration

        # Get indices of timepoints
        ind_6m = sc.findinds(x > .5)[0]
        ind_12m = sc.findinds(x > 1)[0]
        ind_24m = sc.findinds(x > 2)[0]
        ind_120m = sc.findinds(x > 10)[0]

        # Do calculations
        shares_6m = np.dot(aa[ind_6m:], 1-bb[ind_6m:])
        shares_12m = np.dot(aa[ind_12m:], 1-bb[ind_12m:])
        shares_24m = np.dot(aa[ind_24m:], 1-bb[ind_24m:])
        shares_120m = np.dot(aa[ind_120m:], 1-bb[ind_120m:])
        shares['6m'] += shares_6m
        shares['12m'] += shares_12m
        shares['24m'] += shares_24m
        shares['120m'] += shares_120m

        # Write to file
        output += [[gname, f'{shares_6m:.2f} / {shares_12m:.2f} / {shares_24m:.2f} \n']]

    return shares, output


#%% Run as a script
if __name__ == '__main__':

    shares, output = make_tabS2()

    import csv
    with open('tables2.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerows(output)

    print('Done.')
