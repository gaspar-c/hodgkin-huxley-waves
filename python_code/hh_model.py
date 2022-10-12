"""

This module contains the Hodgkin-Huxley (HH) neuron model and its numerical solver

    class HHNeuron:     creates an object with all model equations and parameters

    class SimulateHH:   receives an instance of HHNeuron and simulates it with zero flux
                        boundary conditions, using a numerical method that minimises
                        the global error of the solution

"""

import os
import numpy as np
from scipy.optimize import fsolve


class HHNeuron:
    def __init__(self,
                 length,         # cm,      axon length
                 c_m=1.0,        # muF/cm², membrane capacitance
                 g_k=36.0,       # mS/cm²   potassium conductance
                 e_k=12.0,       # mV,      potassium Nernst potential
                 g_na=120.0,     # mS/cm²,  sodium conductance
                 e_na=-115.0,    # mv,      sodium Nernst potential
                 g_l=0.3,        # mS/cm²,  leak conductance
                 e_l=-10.613,    # mV,      leak Nernst potential
                 q10=3.0,        # -,       q10 parameter
                 temp=6.3,       # ºC,      temperature
                 radius=0.0238,  # cm,      axon radius
                 r2=0.0354,      # kOhm.cm, axon specific resistivity
                 ):

        # initialise HH neuron parameters
        self.length = length
        self.c_m = c_m
        self.g_k = g_k
        self.e_k = e_k
        self.g_na = g_na
        self.e_na = e_na
        self.g_l = g_l
        self.e_l = e_l
        self.q10 = q10
        self.temp = temp
        self.phi = q10 ** ((temp - 6.3) / 10)
        self.radius = radius
        self.r2 = r2

        # get equilibrium values for i=0
        self.v0, self.n0, self.m0, self.h0 = \
            fsolve(self.system_of_equations, np.array([0, 0, 0, 0]), (0,))

    # HH dv/dt equation
    def hh_dv(self, v, n, m, h, i0):
        i_k = self.g_k * (n ** 4) * (v - self.e_k)
        i_na = self.g_na * (m ** 3) * h * (v - self.e_na)
        dv = - (i0 + self.g_l * (v - self.e_l) + i_k + i_na) / self.c_m
        return dv

    # HH dn/dt equation
    def hh_dn(self, v, n):
        dn = -0.125 * np.exp(v / 80) * n * self.phi + \
             0.01 * (1 - n) * self.phi * (v + 10) / (np.exp((v + 10) / 10) - 1)
        return dn

    # HH dm/dt equation
    def hh_dm(self, v, m):
        dm = -4 * np.exp(v / 18) * m * self.phi + \
             0.1 * (1 - m) * self.phi * (v + 25) / (np.exp((v + 25) / 10) - 1)
        return dm

    # HH dh/dt equation
    def hh_dh(self, v, h):
        dh = 0.07 * np.exp(v / 20) * (1 - h) * self.phi - \
             h * self.phi / (1 + np.exp((v + 30) / 10))
        return dh

    # combine the 4 HH equations into one function
    def system_of_equations(self, variables, i0=0):
        v, n, m, h = variables
        return self.hh_dv(v, n, m, h, i0), self.hh_dn(v, n), self.hh_dm(v, m), self.hh_dh(v, h)


class SimulateHH:
    """
    the reaction-diffusion HH system is solved with zero flux boundary conditions

    for gamma = 1/6, this numerical method minimises the global error of the solution,
    as shown in 'Dilão and Sainhas, 1998'

    the resulting error is of the order of (dx)^6,
    where dx = neuron.length / mesh_size
    """

    def __init__(self,
                 name,                  # -,        name of simulation object
                 neuron,                # -,        Hodgkin-Huxley neuron object
                 mesh_size,             # -,        size of spatial mesh
                 current,               # -,        external current
                 diff=None,             # cm^2/ms,  diffusion coefficient; if None, D = (a/2*r2)/c_m
                 end_time=np.inf,       # ms,       simulation end time
                 gamma=(1/6),           # -,        numerical integration parameter (1/6 minimizes the error)
                 dt=None,               # -,        fixed time step (for other integration methods)
                 spike_analyser=None,   # -,        object that analyses the generated spikes
                 snapshots=None,        # ms,       array with times at which to record snapshots
                 live_plot=True,        # -,        plot results while simulation runs
                 ):

        # initialise simulation parameters
        self.name = name
        self.neuron = neuron
        self.mesh_size = mesh_size
        self.dx = neuron.length / self.mesh_size
        self.current = current
        self.gamma = gamma
        self.end_time = end_time
        self.spike_analyser = spike_analyser
        self.live_plot = live_plot

        # if diffusion is not specified, use the HH standard (a/2*r2)/c_m
        if diff is None:
            self.diff = (neuron.radius / (2 * neuron.r2)) / self.neuron.c_m
        else:
            self.diff = diff

        # if dt is not specified, use the error minimisation condition in Dilao-Sainhas (1998)
        if dt is None:
            self.dt = (self.dx ** 2) * self.gamma / self.diff
        else:
            self.dt = dt

        # initialise snapshots as a numpy array
        if snapshots is not None:
            self.snapshots = np.array(snapshots)
        else:
            self.snapshots = np.array([])

        # create a folder on which to record simulation outputs
        os.mkdir('simulation_results/' + self.name)

        # declare and initialise HH variables
        self.sim_time = 0
        self.v_array = None
        self.n_array = None
        self.m_array = None
        self.h_array = None
        self.initialise()

    # initialise HH model variables at the equilibrium for i = 0
    def initialise(self):
        self.v_array = np.ones(self.mesh_size) * self.neuron.v0
        self.n_array = np.ones(self.mesh_size) * self.neuron.n0
        self.m_array = np.ones(self.mesh_size) * self.neuron.m0
        self.h_array = np.ones(self.mesh_size) * self.neuron.h0

    # calculate current density for each point in space at the simulation time
    def get_current_density(self, current):

        # initialise current density at each point in space
        current_x_array = np.zeros(self.mesh_size)

        # get area of cable segment of length dx
        dx_area = 2 * np.pi * self.neuron.radius * self.dx

        # check if simulation time is within specified current injection times
        if (self.sim_time >= current.start_time) and (self.sim_time <= current.stop_time):

            # inject current density at specified locations
            current_x_array[current.x_inj] = current.value / dx_area

        return current_x_array

    # solve HH system with zero flux boundary conditions
    def solver(self):

        # get current density for this simulation time
        i0 = self.get_current_density(self.current)

        # forward euler integration
        self.sim_time += self.dt
        v_next = self.v_array + self.dt * self.neuron.hh_dv(self.v_array, self.n_array, self.m_array, self.h_array, i0)
        n_next = self.n_array + self.dt * self.neuron.hh_dn(self.v_array, self.n_array)
        m_next = self.m_array + self.dt * self.neuron.hh_dm(self.v_array, self.m_array)
        h_next = self.h_array + self.dt * self.neuron.hh_dh(self.v_array, self.h_array)

        # inner mesh
        v_next[1:-1] += self.gamma * (self.v_array[2:] + self.v_array[:-2] - 2 * self.v_array[1:-1])

        # left boundary
        v_next[0] += self.gamma * (self.v_array[1] - self.v_array[0])

        # right boundary
        v_next[-1] += self.gamma * (self.v_array[-2] - self.v_array[-1])

        return v_next, n_next, m_next, h_next

    # solve the equation and perform all necessary operations
    def step_solve(self):

        # flag to check if simulation should end
        end_flag = False

        # numerical solver
        self.v_array, self.n_array, self.m_array, self.h_array = self.solver()

        # perform all spike analyser operations
        if self.spike_analyser is not None:
            end_flag = self.spike_analyser.analyser_checks(self.sim_time, self.v_array)

        # check if simulation should end
        if self.sim_time >= self.end_time:
            end_flag = True

        # print snapshots of the solution at the required simulation times
        for snapshot in self.snapshots:
            if (snapshot - self.sim_time) < self.dt and (snapshot >= self.sim_time):
                np.savetxt('simulation_results/' + self.name + '/%fms.txt' % self.sim_time, -self.v_array)

        return end_flag

    # return string with simulation data
    def get_text(self):
        text_params = 'Simulation Parameters:\n\n' + \
                      ' mesh size = %d\n' % self.mesh_size + \
                      ' dx = %f\n' % self.dx + \
                      ' gamma = 1/%d\n' % int(1/self.gamma) + \
                      ' dt = %f ms\n\n' % self.dt + \
                      ' D = %f cm2/ms\n' % self.diff + \
                      ' I0 = %f muA [%f:%f]\n\n' % (self.current.value,
                                                    self.current.start_time,
                                                    self.current.stop_time) + \
                      ' t = %f ms / %f ms' % (self.sim_time, self.end_time)

        return text_params
