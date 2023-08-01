"""
Script to simulate cervical outcome using the CCNSW transition probabilities
"""
import numpy as np
from numpy.linalg import matrix_power as mp

# Create transition probability matrices
# Take from appendix to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6508597/#R25
# see Table S25
tpm16 = {
    25: np.array([
        [1.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # From well
        [0.591, 0.133, 0.209, 0.067, 0.000, 0.000],  # From HPV
        [0.177, 0.022, 0.731, 0.035, 0.035, 0.000],  # From CIN1
        [0.245, 0.026, 0.092, 0.505, 0.132, 0.000],  # From CIN2
        [0.000, 0.000, 0.053, 0.037, 0.909, 0.001],  # From CIN3
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # From cancer
    ]),
    50: np.array([
        [1.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # From well
        [0.436, 0.419, 0.119, 0.020, 0.000, 0.000],  # From HPV
        [0.177, 0.022, 0.714, 0.040, 0.041, 0.000],  # From CIN1
        [0.245, 0.026, 0.092, 0.416, 0.215, 0.000],  # From CIN2
        [0.000, 0.000, 0.037, 0.032, 0.911, 0.014],  # From CIN3
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # From cancer
    ]),
    70: np.array([
        [1.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # From well
        [0.264, 0.622, 0.093, 0.007, 0.000, 0.000],  # From HPV
        [0.177, 0.022, 0.689, 0.053, 0.044, 0.000],  # From CIN1
        [0.245, 0.026, 0.092, 0.339, 0.284, 0.000],  # From CIN2
        [0.000, 0.000, 0.015, 0.009, 0.929, 0.032],  # From CIN3
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # From cancer
    ]),
    80: np.array([
        [1.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # From well
        [0.233, 0.638, 0.093, 0.007, 0.000, 0.000],  # From HPV
        [0.177, 0.022, 0.674, 0.053, 0.044, 0.000],  # From CIN1
        [0.245, 0.026, 0.092, 0.282, 0.325, 0.000],  # From CIN2
        [0.000, 0.000, 0.007, 0.005, 0.920, 0.038],  # From CIN3
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # From cancer
    ]),
}

TPM = {}
for key, tpm in tpm16.items():
    row_sums = tpm[1:, 1:].sum(axis=1)
    TPM[key] = tpm[1:, 1:] / row_sums[:, np.newaxis]

# Infection at age A for X years
def years_in_age(age, x):
    """
    For a given age, calculate how long an infection of x years will be in each age bucket (0-24, 25-49, 50-69, 70-79)
    """
    n_25 = max(0, 25 - age)
    n_50 = max(0, 50 - age - n_25)
    n_70 = max(0, 70 - age - n_25 - n_50)
    n_80 = max(0, 80 - age - n_25 - n_50 - n_70)
    yia = [n_25, n_50, n_70, n_80]

    aa = []
    for i, a in enumerate(yia):
        if a < x: aa.append(a)
        else: aa.append(max(0,x))
        x -= aa[-1]

    return aa


def outcomes(age, x):
    """
    Calculate probability of a person infected at a given age having each possible outcome after X years
    e.g. age = 23, x = 3 or x = 30
    """
    p0 = np.array([1, 0, 0, 0, 0])  # probs for HPV, CIN1, CIN2, CIN3, CC
    n_25, n_50, n_70, n_80 = years_in_age(age, x)

    tpm_final = np.eye(5)
    if n_25 > 0: tpm_final = np.matmul(tpm_final, mp(TPM[25], n_25))
    if n_50 > 0: tpm_final = np.matmul(tpm_final, mp(TPM[50], n_50))
    if n_70 > 0: tpm_final = np.matmul(tpm_final, mp(TPM[70], n_70))
    if n_80 > 0: tpm_final = np.matmul(tpm_final, mp(TPM[80], n_80))

    return np.matmul(p0, tpm_final)

import pylab as pl
import sciris as sc
fig, axes = pl.subplots(2, 2, figsize=(14,10))
axes = axes.flatten()
xvals = np.arange(1,31)
age_range = [15, 20, 25, 30, 35, 40, 45, 50]
colors = sc.vectocolor(age_range)
sc.options(fontsize=15)
for i, age in enumerate(age_range):
    cin1_probs =\
        []
    cin2_probs = []
    cin3_probs = []
    cancer_probs = []
    for x in xvals:
        cin1_probs.append(outcomes(age, x)[1])
        cin2_probs.append(outcomes(age, x)[2])
        cin3_probs.append(outcomes(age, x)[3])
        cancer_probs.append(outcomes(age, x)[-1])
    axes[0].plot(xvals, cin1_probs, color=colors[i], label=f'{age=}', lw=2)
    axes[1].plot(xvals, cin2_probs, color=colors[i], label=f'{age=}', lw=2)
    axes[2].plot(xvals, cin3_probs, color=colors[i], label=f'{age=}', lw=2)
    axes[3].plot(xvals, cancer_probs, color=colors[i], label=f'{age=}', lw=2)

axes[3].legend()
# for ax in axes:
#     ax.set_xlabel('Duration of infection')
axes[0].set_title('Probability of CIN1')
axes[1].set_title('Probability of CIN2')
axes[2].set_title('Probability of CIN3')
axes[3].set_title('Probability of cancer')
fig.tight_layout()
pl.show()


