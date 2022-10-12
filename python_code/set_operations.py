"""

This module contains operations to be performed after a set of simulations
with varying input currents has ended. Used in simulation sets 3 and 4

    function get_spike_counts:          get spike counts vs injected currents

    function merge_arrays:              merge result files for all simulations in set

    function merge_speeds:              merge speed files for all simulations in set

    function get_dispersion_relation:   get dispersion relation from loaded periods and wavelengths

    function run_set_operations:        run all operations after simulation set has ended

"""


import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# get spike counts for each simulation from 'log.txt' files
def get_spike_counts(filenames):

    # number of simulation files
    n_files = len(filenames)

    # initialise array to store spike counts
    spike_counts = np.zeros((n_files, 2))

    # iterate through each simulation file
    for i in range(n_files):

        # open and read the file
        log_file = open(filenames[i], 'r')
        lines = log_file.readlines()
        for line in lines:

            # read current
            if 'I0' in line:
                spike_counts[i, 0] = float(line[line.find('=') + 1: line.find('muA')])

            # read number of spikes
            if 'spiked' in line:
                spike_counts[i, 1] = float(line[line.find('spiked') + len('spiked'): line.find('times')])

        # close file
        log_file.close()

    # get array with currents, in the order that was loaded by python
    # the order will be important for loading other files
    disordered_currents = spike_counts[:, 0]

    # array with currents vs spike counts, ordered by current
    spike_counts = spike_counts[spike_counts[:, 0].argsort()]

    return spike_counts, disordered_currents


# merge result files across all simulations in set
def merge_arrays(disordered_currents, filenames):

    # number of simulation files
    n_files = len(filenames)

    # initialise merged array
    merged_arrays = np.empty((0, 3), dtype=float)

    # iterate through each file
    for i in range(n_files):

        # load file contents into a numpy array
        file_array = np.array([np.loadtxt(filenames[i])])

        # prevent dimension error if array has size 1
        if len(file_array.T) == 1:
            file_array = np.array([file_array])

        # append a column to the data with the current for the simulation file being read
        array_i = np.append([[disordered_currents[i], j + 1] for j in range(len(file_array.T))],
                            file_array.T, axis=1)

        # merge with the other simulation files
        merged_arrays = np.append(merged_arrays, array_i, axis=0)

    # sort merged array 1) by current and 2) by spike index
    merged_arrays = merged_arrays[np.lexsort((merged_arrays[:, 1], merged_arrays[:, 0]))]

    return merged_arrays


# merge speed files for all simulations in set
def merge_speeds(disordered_currents, filenames):

    # number of simulation files
    n_files = len(filenames)

    # initialise arrays to store initial and final speeds
    init_speeds = np.empty((0, 4), dtype=float)
    final_speeds = np.empty((0, 4), dtype=float)

    # iterate through each simulation file
    for i in range(n_files):

        # load file into numpy array
        file_array = np.array(np.loadtxt(filenames[i]))

        # iterate through each spike in the file
        for spike in np.unique(file_array[:, 0]).astype(int):

            # get first speed for this spike
            spike_init_speed = np.array([
                np.insert(file_array[file_array[:, 0] == spike][0], 1, disordered_currents[i])]
            )
            init_speeds = np.append(init_speeds, spike_init_speed, axis=0)

            # get last speed for this spike
            spike_final_speed = np.array([
                np.insert(file_array[file_array[:, 0] == spike][-1], 1, disordered_currents[i])]
            )
            final_speeds = np.append(final_speeds, spike_final_speed, axis=0)

    # sort speeds 1) by current and 2) by spike index
    init_speeds = init_speeds[np.lexsort((init_speeds[:, 1], init_speeds[:, 0]))]
    final_speeds = final_speeds[np.lexsort((final_speeds[:, 1], final_speeds[:, 0]))]

    return init_speeds, final_speeds


# get dispersion relation from periods and wavelengths
def get_dispersion_relation(merged_periods, merged_wavelengths):

    # calculate frequencies omega
    frequencies = 1 / merged_periods[:, 2]

    # calculate wavenumbers k
    wavenumbers = 2 * np.pi / merged_wavelengths[:, 2]

    # stack currents, spike indices, frequencies, and wavenumbers
    output = None
    if (merged_periods[:, 0] == merged_wavelengths[:, 0]).all() and \
            (merged_periods[:, 1] == merged_wavelengths[:, 1]).all():
        output = np.column_stack((merged_periods[:, 0],
                                  merged_periods[:, 1],
                                  frequencies * 1000,   # scale from ms to second
                                  wavenumbers * 100     # scale from cm to meter
                                  ))

    return output


# run all operations after simulation set has ended (used by sets 3 and 4)
def run_set_operations(output_dir):

    # get the path of all log files within the set directory
    log_filenames = glob.glob(output_dir + '*/log.txt')

    # if no files are found, return 0
    if len(log_filenames) == 0:
        print('Nothing found in %s' % output_dir)
        return 0

    # get filenames for speeds, periods, and wavelengths
    speed_filenames = []
    period_filenames = []
    wavelength_filenames = []
    for name in log_filenames:
        period_name = name.replace('log.txt', 'periods.txt')
        if os.path.exists(period_name):
            period_filenames += [period_name]

        wavelength_name = name.replace('log.txt', 'wavelengths.txt')
        if os.path.exists(wavelength_name):
            wavelength_filenames += [wavelength_name]

        speed_name = name.replace('log.txt', 'speeds.txt')
        if os.path.exists(speed_name):
            speed_filenames += [speed_name]

    # get spike counts vs injected current
    # 'currents' output come in the order that the files were loaded, used to load other files
    counts, currents = get_spike_counts(log_filenames)
    np.savetxt(output_dir + 'current_vs_spikes.txt', counts)
    plt.plot(counts[:, 0], counts[:, 1])
    plt.xlabel('I0 (muA)')
    plt.ylabel('# Spikes')
    plt.savefig(output_dir + 'current_vs_spikes.png')
    plt.close()

    # get periods vs injected current
    periods = []
    if len(period_filenames) > 0:
        periods = merge_arrays(currents, period_filenames)
        np.savetxt(output_dir + 'current_vs_periods.txt', periods)

        later_periods = periods[periods[:, 1] >= 30]
        if len(later_periods) > 0:
            plt.scatter(later_periods[:, 0], later_periods[:, 2], s=0.5, c='black')
            plt.xlabel('I0')
            plt.ylabel('periods (ms)')
            plt.savefig(output_dir + 'current_vs_later_periods.png')
            plt.close()

    # get wavelengths vs injected current
    wavelengths = []
    if len(wavelength_filenames) > 0:
        wavelengths = merge_arrays(currents, wavelength_filenames)
        np.savetxt(output_dir + 'current_vs_wavelengths.txt', periods)

        later_wavelengths = wavelengths[wavelengths[:, 1] >= 30]
        if len(later_wavelengths) > 0:
            plt.scatter(later_wavelengths[:, 0], later_wavelengths[:, 2], s=0.5, c='black')
            plt.xlabel('I0')
            plt.ylabel('wavelengths (cm)')
            plt.savefig(output_dir + 'current_vs_later_wavelengths.png')
            plt.close()

    # if wavelenghts and periods are loaded, get dispersion relation for spikes 1 and 31
    if len(wavelengths) > 0 and len(periods) > 0:
        if len(wavelengths) == len(periods):
            kw = get_dispersion_relation(periods, wavelengths)
            np.savetxt(output_dir + 'current_vs_kw.txt', kw)

            kw1 = kw[kw[:, 1] == 1]
            kw31 = kw[kw[:, 1] == 31]
            if len(kw1) > 0 and len(kw31) > 0:
                plt.scatter(kw31[:, 2], kw31[:, 3], c='black')
                plt.scatter(kw1[:, 2], kw1[:, 3], c='red')
                plt.xlabel('w (1/s)')
                plt.ylabel('k (1/m)')
                plt.savefig(output_dir + 'dispersion_relation.png')
                plt.close()
        else:
            print('detection ERROR: wavelengths and periods have different counts')

    # get spike speeds vs injected current
    if len(speed_filenames) > 0:
        init_speeds_array, final_speeds_array = merge_speeds(currents, speed_filenames)
        np.savetxt(output_dir + 'init_speeds.txt', init_speeds_array)
        np.savetxt(output_dir + 'final_speeds.txt', final_speeds_array)

        # plot speeds as in Figure 7
        fig_width = 15 / 2.54
        fig_height = 20 / 2.54
        fig, ax = plt.subplots(2, 1, figsize=(fig_width, fig_height))
        ax[0].scatter(init_speeds_array[:, 1], init_speeds_array[:, 3] * 10, c='black', s=0.5)
        ax[1].scatter(final_speeds_array[:, 1], final_speeds_array[:, 3] * 10, c='black', s=0.5)
        ax[1].set_xlabel('I0 (muA)')
        ax[0].set_ylabel('initial speed (m/s)')
        ax[1].set_ylabel('final speed (m/s)')
        plt.savefig(output_dir + 'all_speeds.png', dpi=300)
        plt.close()

        # plot speeds as in Figure 6
        last_spike = np.max(final_speeds_array[:, 0])
        final_speeds1 = final_speeds_array[final_speeds_array[:, 0] == 1]
        final_speeds_last = final_speeds_array[final_speeds_array[:, 0] == last_spike]
        if len(final_speeds1) > 0 and len(final_speeds_last) > 0:
            plt.plot(final_speeds1[:, 1], final_speeds1[:, 3] * 10, c='black', ls='--')
            plt.plot(final_speeds_last[:, 1], final_speeds_last[:, 3] * 10, c='black')
            plt.xlabel('I0 (muA)')
            plt.ylabel('final speed (m/s)')
            plt.savefig(output_dir + 'current_vs_final_speeds.png')
            plt.close()
