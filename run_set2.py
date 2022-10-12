"""

Script to run simulation set 2, replicating results shown in Figure 5

The set contains three simulations
    -   'oscillation1'  (Figure 5a)  3.500muA injected at x=0
    -   'oscillation2'  (Figure 5b)  5.570muA injected at x=0
    -   'oscillation3'  (Figure 5c)  6.355muA injected at x=0

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
    set2_name = 'set2_' + time_stamp
    os.mkdir('simulation_results/' + set2_name)

    # neuron and mesh for set 2
    neuron_set2 = HHNeuron(length=250)
    mesh_set2 = 2000

    simulation_set2 = [
        SimulateHH(set2_name + '/oscillation1',
                   neuron_set2,
                   mesh_set2,
                   CurrentI(3.500),
                   spike_analyser=SpikeAnalyser(final_spike=40,
                                                rec_speeds=np.arange(1, 41),  # record speed for all spikes
                                                rec_peaks=1.0),
                   snapshots=[5, 45, 85, 125, 165, 205, 245],
                   ),
        SimulateHH(set2_name + '/oscillation2',
                   neuron_set2,
                   mesh_set2,
                   CurrentI(5.570),
                   spike_analyser=SpikeAnalyser(final_spike=40,
                                                rec_speeds=np.arange(1, 41),  # record speed for all spikes
                                                rec_peaks=1.0),
                   snapshots=[5, 45, 85, 125, 165, 205, 245],
                   ),
        SimulateHH(set2_name + '/oscillation3',
                   neuron_set2,
                   mesh_set2,
                   CurrentI(6.535),
                   spike_analyser=SpikeAnalyser(final_spike=40,
                                                rec_speeds=np.arange(1, 41),  # record speed for all spikes
                                                rec_peaks=1.0),
                   snapshots=[5, 45, 85, 125, 165, 205, 245],
                   )
    ]

    # run simulation set 2
    p.map(run_simulation, simulation_set2)
