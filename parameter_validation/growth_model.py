import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path

# model class
class GrowthModel:
    def __init__(self, p, vol_concent):
        self.p = p  # parameters
        self.vol_concent = vol_concent

    def __call__(self, t, y):
        y = np.maximum(y, 0) 
        S, R, N = y # diff eq functions
        
        # unpack parameters
        r, beta, lambda_, nu, rho, alpha, theta, E_max, MIC_S, MIC_R, mu, gamma = self.p
        V_i, A_i = self.vol_concent

        # model functions
        N = S + R # total bacteria population
        V = V_i + lambda_*t # slurry tank volume
        A = (A_i - (theta / gamma)) * np.exp(-1.0 * gamma * t) + (theta / gamma) # antibiotic concentration
        N_max =  mu * V # max carrying capacity
        H = 2 # Hill coefficient
        E_S = 1 - (E_max * (A / V)**H ) / (MIC_S**H + (A / V)**H) # antibiotic effect on sensitive
        E_R = 1 - (E_max * (A / V)**H) / (MIC_R**H + (A / V)**H) # antibiotic effect on resistant

        # differential equations
        dSdt = r * (1 - N/N_max) * E_S * S - (beta * S * R) / N + lambda_ * (1 - rho) * nu
        dRdt = r * (1 - alpha) * (1 - N/N_max) * E_R * R + (beta * S * R)/N + lambda_ * rho * nu
        dNdt = dSdt + dRdt

        return [dSdt, dRdt, dNdt]


# simulation class
class Simulation:
    def __init__(self, p, S0, R0, N0, V0, A0):
        self.p = p  # parameters
        self.S0 = S0  # inital sentitive pop
        self.R0 = R0  # initial resistant pop
        self.N0 = S0 + R0 # total pop

        self.V0 = V0  # initial volume in slurry tank
        self.A0 = A0  # initial antibiotic concentration

        self.y0 = np.array([S0, R0, N0])  # initial bacteria state
        self.vol_concent0 = np.array([V0, A0]) # initial env state

    def run(self, t_span=np.linspace(0, 24), num_time_points=500):
        t_eval = np.linspace(t_span[0], t_span[1], num_time_points)
        model = GrowthModel(self.p, self.vol_concent0)
        sol = solve_ivp(model, t_span, self.y0, t_eval=t_eval)
        return sol.t, sol.y*1e-15
    

# sample from a truncated normal distribution
def sample_truncated_normal(mean, sd, low, upp, size=None):
    if sd == 0: return mean
    else:
        a, b = (low - mean) / sd, (upp - mean) / sd
        return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size)


# parameter sampling class
class ParameterSampler:
    def __init__(self, param_specs, seed=None):
        self.param_specs = param_specs
        if seed is not None:
            np.random.seed(seed)

    def sample_parameters(self):
        p = []
        for key in ['r', 'beta', 'lambda_', 'nu', 'rho', 'alpha', 'theta', 'E_max', 'MIC_S', 'MIC_R', 'mu', 'gamma']:
            spec = self.param_specs[key]
            value = sample_truncated_normal(spec['mean'], spec['sd'], spec['low'], spec['high'])
            p.append(value)
        return p


# simulation class with sampling
class SimulationRunner:
    def __init__(self, num_simulations, conditions, t_eval, param_specs, seed=None):
        self.num_simulations = num_simulations
        self.conditions = conditions
        self.t_eval = t_eval
        self.param_sampler = ParameterSampler(param_specs, seed=seed)
        self.parameters = np.zeros((num_simulations, 12))
        self.curves = np.zeros((num_simulations, len(conditions), 3, len(t_eval)))

    def simulate(self, p, S0, R0, V0, A0):
        N0 = S0 + R0
        sim = Simulation(p, S0, R0, N0, V0, A0)
        _, results = sim.run(t_span=(0, 24), num_time_points=len(self.t_eval))
        return results

    def run_all_simulations(self):
        for i in range(self.num_simulations):

            # sample parameters for run
            p = self.param_sampler.sample_parameters()
            self.parameters[i] = p

            # run simulations under the specified conditions
            for j, (S0, R0, V0, A0) in enumerate(self.conditions):
                simulated = self.simulate(p, S0, R0, V0, A0)
                self.curves[i, j] = simulated

            if (i + 1) % 400 == 0:
                print(f'Simulations completed: {i + 1}/{self.num_simulations}')

        # save results to files
        np.savez_compressed('simulated_data/curves1k.npz', curves=self.curves, t_eval=self.t_eval)
        np.save('simulated_data/parameters.npy', self.parameters)