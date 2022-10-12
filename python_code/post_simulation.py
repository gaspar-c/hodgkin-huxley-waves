"""

This module contains operations to be performed after each simulation

    function store_peak_records:    store recorded peaks of generated spikes (for characteristic curves)

    function linear_func:           return y = m*x + b

    function calc_spike_speeds:     calculate spike speeds along the axon

    function store_speed_records:   store recorded speeds of generated spikes

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# sort recorded peak trajectories into individual spikes and store them
def store_peak_records(simulation):

    print('%s: recording peak positions...' % simulation.name)

    # scale recorded peaks with simulation dimensions
    peaks = simulation.spike_analyser.track_peaks
    peaks[:, 0] *= simulation.dx
    peaks[:, 1] *= simulation.dt

    # if sort_peaks is False
    if not simulation.spike_analyser.sort_peaks:

        # store all peaks as belonging to a single spike, sorted by space location
        peak_dict = {1: peaks[peaks[:, 0].argsort()]}

    # if sort_peaks is True
    else:
        # sort all peaks by time dimension
        peaks = peaks[peaks[:, 1].argsort()]

        # store first two peaks in dictionary, as belonging to spike 1
        # dictionary key "i" will have the peaks for i-th spike
        peak_dict = {1: np.array([peaks[0], peaks[1]])}

        # distance between first two peaks
        max_dist = np.sqrt((peaks[1][0] - peaks[0][0]) ** 2 + (peaks[1][1] - peaks[0][1]) ** 2)

        # count detected spikes
        count_spikes = 1

        # iterate through every remaining peak
        for j in range(2, len(peaks)):

            # arrays to store distance between j-th peak and last peak for each spike
            dx_array = np.zeros(count_spikes)
            dt_array = np.zeros(count_spikes)

            # iterate through each spike
            for spike_idx in range(count_spikes):
                dx_array[spike_idx] = peaks[j][0] - peak_dict[spike_idx + 1][-1, 0]
                dt_array[spike_idx] = peaks[j][1] - peak_dict[spike_idx + 1][-1, 1]

            # negative distances are exaggerated, if it appears to go backwards it's a different spike
            dx_array[dx_array < 0] = np.inf

            # distance between j-th peak and last peak for each spike
            dist_array = np.sqrt(dx_array ** 2 + dt_array ** 2)

            # if shortest distance is less than 10*max_dist, add peak j to the nearest spike
            if np.min(dist_array) < 10 * max_dist:
                spike_idx = np.argmin(dist_array) + 1
                max_dist = np.max([dist_array[spike_idx - 1], max_dist])

                peak_dict[spike_idx] = np.append(peak_dict[spike_idx], [peaks[j]], axis=0)

            # otherwise, consider it's a new spike
            else:
                count_spikes += 1
                peak_dict[count_spikes] = np.array([peaks[j]])

        # check that detected spikes match number of recorded spikes:
        if count_spikes != simulation.spike_analyser.n_spikes_left:
            print('%s: Found %d branches for %d spikes' %
                  (simulation.name, count_spikes + 1, simulation.spike_analyser.n_spikes_left))

    # directory to store outputs
    output_dir = 'simulation_results/' + simulation.name + '/'

    # plot peak coordinates for each spike
    plt.scatter(peaks[:, 0], peaks[:, 1], c='gray', alpha=0.5, s=2)
    for peak in range(1, len(peak_dict) + 1):
        plt.plot(peak_dict[peak][:, 0], peak_dict[peak][:, 1], lw=1)
    plt.xlabel('x (cm)')
    plt.ylabel('time (ms)')
    plt.savefig(output_dir + 'peak_coordinates.png', dpi=300)
    plt.close()

    # convert dictionary to numpy array and store it
    peaks_array = np.empty((0, 4), float)
    for spike_idx in range(1, len(peak_dict) + 1):
        branch_array = np.insert(peak_dict[spike_idx], 0, spike_idx, axis=1)
        peaks_array = np.append(peaks_array, branch_array, axis=0)
    np.savetxt(output_dir + 'peaks.txt', peaks_array)


# line function
def linear_func(x, m, b):
    return m*x + b


# calculate spike speeds from spike front coordinates
def calc_spike_speeds(spike_coord, sample_size):
    """
    calculate spike speeds along the axon, by fitting samples
    of spike front coordinates to a linear function

    :param spike_coord:     array containing coordinates of spike fronts for all spikes
    :param sample_size:     size of samples from which to measure propagation speed
    :return: speed_dict:    dictionary with speeds along axon for each spike
    """

    # get recorded spike indices
    spike_indices = np.unique(spike_coord[:, 0]).astype(int)

    # dictionary where spike speeds will be recorded
    speed_dict = {}

    # iterate through every recorded spike
    for spike_idx in spike_indices:

        # get recordings corresponding to spike_idx
        spike = spike_coord[spike_coord[:, 0] == spike_idx]

        # calculate speeds along axon for spike spike_idx:
        speed = np.empty((0, 2))

        # iterate through all recordings for this spike
        for idx in range(len(spike)):

            # get one sample
            sample = spike[idx:idx + sample_size, 1:]

            # if sample is complete
            if len(sample) == sample_size:

                # find slope of sample, v = dx/dt
                (slope, _), _ = curve_fit(linear_func, sample[:, 1], sample[:, 0])

                # get mean location of the sample
                sample_x = np.mean(spike[idx:idx + sample_size, 1])

                # store speed for that location
                speed = np.append(speed, [[sample_x, slope]], axis=0)

        # store speeds for current spike in dictionary
        if len(speed) > 0:
            speed_dict[spike_idx] = speed

    return speed_dict


# calculate and store spike speeds
def store_speed_records(simulation):
    print('%s: recording spike speeds...' % simulation.name)

    # scale spike front coordinates with simulation dimensions
    spike_coord = simulation.spike_analyser.spike_front_coord
    spike_coord[:, 1] *= simulation.dx
    spike_coord[:, 2] *= simulation.dt

    # axon sample size to measure speeds
    sample_size = int(simulation.mesh_size * 0.10)
    speed_dict = calc_spike_speeds(spike_coord, sample_size)

    # convert dictionary to numpy array
    speeds_array = np.empty((0, 3), float)
    for spike_idx in speed_dict.keys():
        branch_array = np.insert(speed_dict[spike_idx], 0, spike_idx, axis=1)
        speeds_array = np.append(speeds_array, branch_array, axis=0)

    if len(speeds_array) > 0:

        # store numpy array with all speeds
        output_dir = 'simulation_results/' + simulation.name + '/'
        np.savetxt(output_dir + 'speeds.txt', speeds_array)

        # get array with initial speeds for each spike
        speed_init = np.empty(0)
        for spike_idx in speed_dict.keys():
            speed_init = np.append(speed_init, speed_dict[spike_idx][0, 1])

        # create figure for plotting speeds
        fig_width = 15 / 2.54
        fig_height = 15 / 2.54
        fig, ax = plt.subplots(2, 1, figsize=(fig_width, fig_height))

        # plot initial speeds for all spikes
        ax[0].plot(speed_dict.keys(), speed_init * 10, marker='.', c='black', lw=1)
        ax[0].set_ylim([10, 12.5])
        ax[0].set_xlabel('N')
        ax[0].set_ylabel('initial speed (m/s)')

        # plot speed along axon for all spikes
        for spike_idx in speed_dict.keys():
            if spike_idx <= 29:
                ax[1].plot(speed_dict[spike_idx][:, 0], speed_dict[spike_idx][:, 1] * 10, c='black', lw=1, ls='--')
            else:
                ax[1].plot(speed_dict[spike_idx][:, 0], speed_dict[spike_idx][:, 1] * 10, c='black', lw=1)
        ax[1].set_ylim([10, 12.5])
        ax[1].set_xlim([0, simulation.neuron.length])
        ax[1].set_xlabel('x (cm)')
        ax[1].set_ylabel('speed (m/s)')

        # adjust plot
        for idx in range(2):
            for spine in ['top', 'right']:
                ax[idx].spines[spine].set_visible(False)
        plt.subplots_adjust(hspace=0.3)

        # store figure
        plt.savefig(output_dir + 'speeds.png', dpi=300)
        plt.close()
