# HPVsim methods manuscript

This repository includes the code for reproducing the figures and analyses of the HPVsim methods manuscript. The original scientific paper describing HPVsim is available at http://paper.hpvsim.org. The recommended citation is:

> **HPVsim: An agent-based model of HPV transmission and cervical disease**. Stuart RM, Cohen JC, Kerr CC, Abeysuriya RG, Zimmermann M, Rao DW, Boudreau MC, Klein DJ (2023).

This repository contains scripts for creating Figures 1, 4, 5, and 6. The code for Figure 2 is in a separate repository located here: https://github.com/hpvsim/hpvsim_india, and Figure 3 is a schematic.

**Results in this repository were produced with HPVsim v2.2.6.** Plot-ready baselines live in [`results/v2.2.6_baseline/`](results/v2.2.6_baseline/); the original published v1.0.0 values (extracted from committed `.obj` artifacts) live in [`results/v1.0.0_published/`](results/v1.0.0_published/). Run `python compare_baselines.py` to generate side-by-side comparison plots.

## Organization

The repository is organized as follows:

### Running scripts

There are separate scripts for running and plotting each figure and the use cases. Specifically:

#### `plot_fig1.py`
 - This script can be used to reproduce Figure 1.

#### `plot_fig4.py` 
- This script can be used to reproduce Figure 4, the first figure in the results section.

#### `plot_fig56.py`
This script is used for running and plotting the results for the remaining two case studies in the results section.
- The sections `run_simple`, `calibrate` and `plot_calibrate` are legacy code that was used for validating the exercises - these sections can safely be ignored.
- The section `run_sims` runs the simulations and creates the plotting objects used for section 3.2 of the methods paper.
- The section `plot_fig5_sims` can be used to reproduce Figure 5.
- The section `run_screening` runs the simulations and creates the plotting objects used for section 3.3 of the methods paper.
- The section `plot_fig6_screening` can be used to reproduce Figure 6.


### Baseline-freezing and comparison scripts
- `save_fig1_baseline.py` freezes Fig 1 plot-ready arrays to a versioned baseline dir.
- `save_fig56_baseline.py` freezes Fig 5/6 plot-ready values, extracting from the committed `.obj` files or (for newer pandas-incompatible pickles) a CSV dump of `dwelltime_df`.
- `compare_baselines.py` produces side-by-side comparison figures across baseline dirs (v1.0.0 vs v2.2.6, extensible to v2.3 / v3.0).

### Additional utility scripts
- `utils.py` contains utilities for numerical calculations and creating plots.


## Installation

If HPVsim is already installed (`pip install hpvsim`), the only other required dependency is ``seaborn``. Alternatively, to install this code as a Python library, run `pip install -e .` (including the dot). This will install the latest version of each library (including HPVsim). You can also install with `pip install -e .[frozen]` to use the versions of libraries (including HPVsim) that were originally used for these analyses. 


## Usage

Run the desired analyses by running one of the scripts described above.


## Further information

Further information on HPVsim is available [here](http://docs.hpvsim.org). If you have further questions or would like technical assistance, please reach out to us at info@hpvsim.org.
