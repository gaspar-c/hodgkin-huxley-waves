"""

Script to run simulation set 3, replicating results shown in Figures 6 and 8

The simulations span the entire oscillatory region, which is divided into 'n_sims' steps
Each of the 'n_sims' simulations injects a different current at x=0

Simulations end once 31 spikes have been produced
Speeds are recorded from spikes 1 and 31

The result shown in Figure 6 can be seen in the output figure 'current_vs_final_speeds.png'
The result shown in Figure 8 can be seen in the output figure 'dispersion_relation.png'

In the manuscript we used 'n_sims' = 200

"""

import time
import os
import multiprocessing
import numpy as np
from python_code.hh_model import HHNeuron, SimulateHH
from python_code.simulate import CurrentI, SpikeAnalyser, run_simulation
from python_code.set_operations import run_set_operations


if __name__ == '__main__':

    # string with current date and time
    time_stamp = time.strftime("%Y%m%d_%H%M%S")

    # number of cores to use in parallel
    n_cores = 4
    p = multiprocessing.Pool(n_cores)

    # name of simulation set, results will be stored under this directory
    set3_name = 'set3_' + time_stamp
    os.mkdir('simulation_results/' + set3_name)

    # number of simulations to run
    n_sims = 10

    # create array with currents within [I1, I5]
    current_start = 1.087
    current_stop = 6.533
    if n_sims > 1:
        current_step = (current_stop - current_start) / (n_sims - 1)
        current_array = np.arange(current_start, current_stop + current_step, step=current_step)
    else:
        current_array = [current_start]

    # create simulation set
    simulation_set3 = []
    for current_value in current_array:
        simulation_set3 += [
            SimulateHH(set3_name + '/current%f' % current_value,
                       HHNeuron(length=250),
                       2000,
                       CurrentI(current_value),
                       spike_analyser=SpikeAnalyser(
                           final_spike=31,
                           rec_speeds=np.array([1, 31])  # record speed for spikes 1 and 31
                       ),
                       live_plot=False
                       )
        ]

    print('Running %d simulations...' % len(simulation_set3))
    p.map(run_simulation, simulation_set3)

    # run set operations once simulations are complete
    run_set_operations('simulation_results/' + set3_name + '/')
