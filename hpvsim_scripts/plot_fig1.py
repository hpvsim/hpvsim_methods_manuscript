"""
This script produces figure 1 of the HPVsim methods paper -- the standard simulation
output
"""

import hpvsim as hpv
import utils as ut


def plot_fig1(simkw=None, plotkw=None):
    if simkw is None: simkw = dict()
    if plotkw is None: plotkw = dict()
    sim = hpv.Sim(**simkw)  # Create a simulation with default settings
    sim.run()			    # Run the simulation
    sim.plot(**plotkw)      # Plot the simulation & optionally save

    return sim


#%% Run as a script
if __name__ == '__main__':

    sim = plot_fig1(
        simkw=dict(location='india', n_agents=50e3, start=1950, end=2020, ms_agent_ratio=100),
        plotkw=dict(do_save=True, fig_path=f'../{ut.figfolder}/fig1.png')
    )

    print('Done.')
