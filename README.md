# HPVsim methods manuscript

This repository includes the code for reproducing the figures and analyses of the HPVsim methods manuscript. The original scientific paper describing HPVsim is available at http://paper.hpvsim.org. The recommended citation is:

> **HPVsim: An agent-based model of HPV transmission and cervical disease**. Stuart RM, Cohen JC, Kerr CC, Abeysuriya RG, Zimmermann M, Rao DW, Boudreau MC, Klein DJ (2023).

## Organization

The repository is organized as follows:

### Running scripts

There are separate scripts for running and plotting each figure and the use cases. Specifically:

#### `plot_fig1.py`
 - This script can be used to reproduce Figure 1.

#### `plot_fig4.py` 
- This script can be used to reproduce Figure 4.

#### `use_case_1.py`
This script is used for running and plotting the results for use case 1.
- The `run_scenarios` section is computationally intensive and typically run on virtual machines.
 - `plot_scenarios` and `run_cea` creates plots of the use case 1 outputs.

#### `use_case_2.py`
This script is used for running and plotting the results for use case 2.
- The `run_scenarios` section is computationally intensive and typically run on virtual machines.
 - `plot_scenarios` creates plots of the use case 2 results.


### Additional utility scripts
- `analyzers.py` defines two custom analyzers for extracting additional data from simulations.
- `utils.py` contains utilities used to preprocess data, post-process results, and create plots.


## Installation

If HPVsim is already installed (`pip install hpvsim`), no other dependencies are required. Alternatively, to install this code as a Python library, run `pip install -e .` (including the dot). This will install the latest version of each library (including HPVsim). You can also install with `pip install -e .[frozen]` to use the versions of libraries (including HPVsim) that were originally used for these analyses. 


## Usage

Run the desired analyses by running one of the scripts described above.


## Further information

Further information on HPVsim is available [here](http://docs.hpvsim.org). If you have further questions or would like technical assistance, please reach out to us at info@hpvsim.org.
