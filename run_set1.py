"""

Script to run simulation set 1, replicating results shown in Figures 2, 3, and 4

The set contains three simulations
    -   '1spike'    (Figure 2)  1.028muA injected at x=0, 1 spike is generated
    -   '3spikes'   (Figure 3)  1.073muA injected at x=0, 3 spikes are generated
    -   'collision' (Figure 4)  1.065muA injected at x=[L/3, 2L/3], 1 spike generated at each injection point

"""

import time
import os
import multiprocessing
import numpy as np
from python_code.hh_model import HHNeuron, SimulateHH
from python_code.simulate import CurrentI, SpikeAnalyser, run_simulation


if __name__ == '__main__':

    # string with current date and time
    time_stamp = time.strftime("%Y%m%d_%H%M%S")

    # number of cores to use in parallel
    n_cores = 4
    p = multiprocessing.Pool(n_cores)

    # name of simulation set, results will be stored under this directory
    set1_name = 'set1_' + time_stamp
    os.mkdir('simulation_results/' + set1_name)

    # neuron and mesh for set 1
    neuron_set1 = HHNeuron(length=100)
    mesh_set1 = 800

    simulation_set1 = [
        # results shown in Figure 2
        SimulateHH(set1_name + '/1spike',
                   neuron_set1,
                   mesh_set1,
                   CurrentI(1.028),
                   spike_analyser=SpikeAnalyser(rec_peaks=1.0),
                   end_time=110,
                   snapshots=[5, 30, 55, 80, 105],
                   ),

        # results shown in Figure 3
        SimulateHH(set1_name + '/3spikes',
                   neuron_set1,
                   mesh_set1,
                   CurrentI(1.073),
                   spike_analyser=SpikeAnalyser(rec_peaks=1.0),
                   end_time=140,
                   snapshots=[5, 30, 55, 80, 105, 130],
                   ),

        # results shown in Figure 4
        SimulateHH(set1_name + '/collision',
                   neuron_set1,
                   mesh_set1,
                   CurrentI(1.065, x_inj=np.array([int(mesh_set1 / 3), int(2 * mesh_set1 / 3)])),
                   spike_analyser=SpikeAnalyser(rec_peaks=1.0, sort_peaks=False),
                   end_time=100,
                   snapshots=[4, 6, 8, 10, 12, 14, 16, 18, 20],
                   ),
    ]

    # run simulation set 1
    p.map(run_simulation, simulation_set1)
