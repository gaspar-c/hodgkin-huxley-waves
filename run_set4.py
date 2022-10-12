"""

Script to run simulation set 4, replicating results shown in Figure 7

The simulations span the transition from type-I intermittency to the oscillatory region
Each of the 'n_sims' simulations injects a different current at x=0

Simulations end once a maximum of 31 spikes have been produced
All spike speeds are recorded

The result shown in Figure 7 can be seen in the output figure 'all_speeds.png'

In the manuscript we used 'n_sims' = 500

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
    set4_name = 'set4_' + time_stamp
    os.mkdir('simulation_results/' + set4_name)

    # number of simulations to run
    n_sims = 50

    # create array with currents in the vicinity of I1
    current_start = 1.025
    current_stop = 1.100
    if n_sims > 1:
        current_step = (current_stop - current_start) / (n_sims - 1)
        current_array = np.arange(current_start, current_stop + current_step, step=current_step)
    else:
        current_array = [current_start]

    # create simulation set
    simulation_set4 = []
    for current_value in current_array:
        simulation_set4 += [
            SimulateHH(set4_name + '/current%f' % current_value,
                       HHNeuron(length=250),
                       2000,
                       CurrentI(current_value),
                       spike_analyser=SpikeAnalyser(
                           final_spike=31,
                           rec_speeds=np.arange(1, 31)  # record speed for all spikes
                       ),
                       live_plot=False
                       )
        ]

    print('Running %d simulations...' % len(simulation_set4))
    p.map(run_simulation, simulation_set4)

    # run set operations once simulations are complete
    run_set_operations('simulation_results/' + set4_name + '/')
