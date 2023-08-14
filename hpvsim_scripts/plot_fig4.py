import sciris as sc
import pylab as pl
import hpvsim as hpv
import utils as ut

rerun = True
filename = '../results/multiscale_test.df'

T = sc.timer()

if rerun:

    # Define the parameters
    pars = dict(
       # total_pop=10e3,  # Population size
        start=1980,  # Starting year
        n_years=50,  # Number of years to simulate
        genotypes=[16, 18],  # Include the two genotypes of greatest general interest
        verbose=0,
        rel_init_prev=4.0,
    )

    debug = 0
    repeats = [10, 3][debug]
    ms_agent_ratios = [[1, 3, 10, 30, 100], [1, 3, 10, 30]][debug]
    n_agents = [[100, 200, 500, 1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3], [100, 200, 500, 1000]][debug]

    # Run the sims -- not parallelized to collect timings
    data = []
    count = 0
    for n in n_agents:
        for ms in ms_agent_ratios:
            for r in range(repeats):
                count += 1
                label = f'n={n} ms={ms} r={r}'
                sc.heading(f'Running {count} of {len(n_agents) * len(ms_agent_ratios) * repeats}: {label}')
                sim = hpv.Sim(pars, rand_seed=r, n_agents=n, ms_agent_ratio=ms, label=label)
                T.tic()
                sim.run()
                sim.time = T.tocout()
                row = dict(
                    n=n,
                    ms=ms,
                    seed=r,
                    time=sim.time,
                    n_agents=len(sim.people),
                    infs=sim.results.infections.values.sum(),
                    cancers=sim.results.cancers.values.sum(),
                    deaths=sim.results.cancer_deaths.values.sum()
                )
                data.append(row)
                print(f'Time: {sim.time:0.2f} s')

    df = sc.dataframe(data)
    sc.save(filename, df)

else:
    df = sc.load(filename)

# %% Analysis

g = df.groupby(['n', 'ms'])
gm = g.mean()
gs = g.std()
gc = g.std() / g.mean()

sc.heading('Means')
print(gm)
sc.heading('STDs')
print(gs)
sc.heading('CoVs')
print(gc)

quantity = ['infs', 'cancers', 'deaths'][1]


def set_font(size=None, font='Libertinus Sans'):
    """ Set a custom font """
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


ut.set_font(size=18)
fig, ax = pl.subplots(figsize=(11, 6))
index = gc.reset_index()
colors = sc.vectocolor(pl.log(index['ms'].values), cmap='parula')

# Hack to get color legend to appear
# From https://github.com/matplotlib/matplotlib/issues/16616
import numpy as np
from matplotlib.colors import from_levels_and_colors
c = index['ms'].values
colors = sc.vectocolor(pl.log(index['ms'].values), cmap='parula')
cmap, norm = from_levels_and_colors(np.append(index['ms'].values[:5], 101), colors[:5])

# Transform sizes
sizes = 2 * (index['n'].values) ** (1 / 2)
scatter = pl.scatter(gm['time'].values, gc[quantity].values, c=c, cmap=cmap, norm=norm, s=sizes, lw=0, marker='o', alpha=0.7)
pl.gca().set_xscale('log')
pl.gca().set_yscale('log')
legend1 = pl.legend(*scatter.legend_elements(), bbox_to_anchor=(0.85,0.99), loc='upper right', title="MS ratio", frameon=False)
ax.add_artist(legend1)

# Another hack to get the marker size legend correct
# From https://stackoverflow.com/questions/24164797/creating-a-matplotlib-scatter-legend-size-related
msizes = [sizes[0], sizes[15], sizes[30], sizes[45]]
labels = ['100', '1k', '10k', '100k']
markers = []
for sn, size in enumerate(msizes):
    markers.append(pl.scatter([], [], s=size, c='k', alpha=0.6, label=labels[sn]))
pl.legend(handles=markers, ncols=1, frameon=False, title="# agents", bbox_to_anchor=(0.85,0.99), loc='upper left')
ax.add_artist(legend1)

pl.xlabel('Time per simulation (s)')
pl.ylabel(f'Coefficient of variation in {quantity}')


fig.tight_layout()
pl.savefig(f"../{ut.figfolder}/fig4.png", dpi=100)
pl.show()

total = T.timings[:].sum()
print(f'Done: {total:0.0f} s')
