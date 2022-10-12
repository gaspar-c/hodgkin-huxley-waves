# hodgkin-huxley-waves

Code to reproduce the results of the manuscript "Action potential solitons and waves in axons", Cano and Dilão (2022).

## How to run

The results are divided into 4 separate simulation sets. Each set can be run with a python script file:

 - `run_set1.py` generates results shown in Figures 2, 3, and 4;
 - `run_set2.py` generates results shown in Figure 5;
 - `run_set3.py` generates results shown in Figures 6 and 8;
 - `run_set4.py` generates results shown in Figure 7.
 
These scripts output `.png` and `.txt` files. These files replicate all the results shown in the manuscript.

The code has been tested using the following libraries:

    python=3.10.6 numpy=1.23.3 scipy=1.9.1 matplotlib=3.6.0

## Content

The python scripts mentioned above make use of the code in the `python_code/` directory.

### `python_code/`

 - `hh_model.py` contains the Hodgkin-Huxley neuron model and its numerical solver;
 - `simulate.py` contains code needed to run a simulation of the HH model;
 - `post_simulation.py` contains operations to be performed after each simulation;
 - `set_operations.py` contains operations to be performed after a set of simulations with varying input currents. Used in simulation sets 3 and 4.

### `mathematica_code/`

The figures included in the manuscript were generated with Mathematica scripts. Running these will create new figure files but won't generate new results.

### `simulation_results/`

The python output files and Mathematica figure files are saved in this folder.
