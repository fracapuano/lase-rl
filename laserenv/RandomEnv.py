"""
This code is mostly derived from https://github.com/gabrieletiboni/doraemon/blob/master/dr_envs/random_env.py
"""
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import pdb

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.beta import Beta

class RandomEnv(gym.Env):
    """Superclass for all environments
        supporting Domain Randomization
        of dynamics parameters
    """

    def __init__(self):
        super().__init__()

        self.nominal_values = None
        self.sampling = None
        self.dr_training = False
        self.preferred_lr = None
        self.reward_threshold = None
        self.dyn_ind_to_name = None

        self.expose_dr_sampler = False
        self.dr_sampler = []

        self.overrides = {}

    
    # Methods to override in child envs:
    # ----------------------------
    def get_search_bounds_mean(self, index):
        """Get search space for current randomized parameter at index `index`"""
        raise NotImplementedError

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index `index`"""
        return -np.inf

    def get_task_upper_bound(self, index):
        """Returns highest feasible value for current randomized parameter at index `index`"""
        return np.inf

    def get_task(self):
        """Get current dynamics parameters"""
        raise NotImplementedError

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        raise NotImplementedError
    # ----------------------------

    def set_random_task(self):
        """Sample and set random parameters
            
            Optionally keeps track of the sampled
            random parameters.
        """
        # Potentially override this method with a gym wrapper
        if 'set_random_task' in self.overrides:
            self.overrides['set_random_task']()
            return

        task = self.sample_task()
        if self.expose_dr_sampler:
            self.dr_sampler.append(task)
        self.set_task(*task)

    def set_dr_training(self, flag):
        """
            If True, new dynamics parameters
            are sampled and set at each .reset() call
        """
        self.dr_training = flag

    def get_dr_training(self):
        return self.dr_training

    def set_expose_dr_sampler(self, flag):
        assert isinstance(flag, bool), f'Flag was {flag} instead of boolean'
        self.expose_dr_sampler = flag

    def reset_dr_sampler(self):
        self.dr_sampler = []

    def get_dr_sampler(self):
        return self.dr_sampler

    def get_reward_threshold(self):
        return self.reward_threshold

    def dyn_index_to_name(self, index):
        assert self.dyn_ind_to_name is not None
        return self.dyn_ind_to_name[index]

    def set_dr_distribution(self, dr_type, distr):
        """
            Set a DR distribution

            dr_type : str
                      {uniform, beta}

            distr : list of distr parameters, or dict for beta
        """
        self.sampling = dr_type
        if dr_type == 'uniform':
            self._set_udr_distribution(distr)
        elif dr_type == 'beta':
            self._set_beta_distribution(distr)
        else:
            raise Exception('Unknown dr_type:'+str(dr_type))

    def get_dr_distribution(self):
        if self.sampling == 'uniform':
            return self.min_task, self.max_task
        elif self.sampling == 'beta':
            return self.distr
        else:
            return None

    def _set_udr_distribution(self, bounds):
        for i in range(len(bounds)//2):
            self.min_task[i] = bounds[i*2]
            self.max_task[i] = bounds[i*2 + 1]
        return

    def _set_beta_distribution(self, distr):
        """
            distr: list of dict
                       list of independent beta distributions.
                       4 keys per dimensions are expected:
                        m=min, M=max, a, b

                    Y ~ Beta(a,b,m,M)
                    y = x(M-m) + m
                    f(y) = f_x((y-m)/(M-m))/(M-m)
        """
        self.distr = distr.copy()
        self.to_distr = []
        for i in range(len(self.distr)):
            a = torch.as_tensor(self.distr[i]['a'], dtype=torch.float32)
            b = torch.as_tensor(self.distr[i]['b'], dtype=torch.float32)
            self.to_distr.append(Beta(a, b))

    def set_task_search_bounds(self):
        """Sets the parameter search bounds based on how they are specified in get_search_bounds_mean"""
        dim_task = len(self.get_task())
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            self.min_task[i], self.max_task[i] = b[0], b[1]

    def get_task_search_bounds(self):
        dim_task = len(self.get_task())
        min_task = np.empty(dim_task)
        max_task = np.empty(dim_task)
        for i in range(dim_task):
            b = self.get_search_bounds_mean(i)
            min_task[i], max_task[i] = b[0], b[1]
        return min_task, max_task

    def sample_tasks(self, num_tasks=1):
        return np.stack([self.sample_task() for _ in range(num_tasks)])

    def sample_task(self):
        """Sample random dynamics parameters"""
        if self.sampling == 'uniform':
            return np.random.uniform(self.min_task, self.max_task, self.min_task.shape)
        elif self.sampling == 'beta':
            sample = []
            for i in range(len(self.distr)):
                m, M = self.distr[i]['m'], self.distr[i]['M']
                value = self.to_distr[i].sample()*(M - m) + m
                sample.append(value.item())
            return np.array(sample)
        else:
            raise ValueError('sampling value of random env needs to be set before using sample_task() or set_random_task(). Set it by uploading a DR distr.')

    def _denormalize_parameters_multivariateGaussian(self, sample):
        if self.distr_low_bound is not None and self.distr_high_bound is not None:
            return sample * (self.distr_high_bound - self.distr_low_bound) + self.distr_low_bound

    def denormalize_parameters(self, parameters):
        """Denormalize parameters back to their original space
        
            Parameters are assumed to be normalized in
            a space of [0, 4]
        """
        assert parameters.shape[0] == self.task_dim

        min_task, max_task = self.get_task_search_bounds()
        parameter_bounds = np.empty((self.task_dim, 2), float)
        parameter_bounds[:,0] = min_task
        parameter_bounds[:,1] = max_task

        orig_parameters = (parameters * (parameter_bounds[:,1]-parameter_bounds[:,0]))/4 + parameter_bounds[:,0]

        return np.array(orig_parameters)

    def get_uniform_dr_by_percentage(self,
                                     percentage: float,
                                     nominal_values: List[float] = None,
                                     lower_bounds: List[float] = None,
                                     dyn_mask: List[int] = None):
        """Returns uniform DR distribution centered in the
        nominal values, and half-width = percentage *
        * (nominal_values - lower_bounds).
        nominal values should be set in between the lower and
        upper bounds, but this is not checked.

            :param percentage: uniform half-width as % of (nominal - lower_bounds)
            :param nominal_values: custom nominal values instead of default ones
            :param lower_bounds: custom lower bounds instead of default ones
            :param dyn_mask: randomize some parameters only
        """
        assert percentage >= 0.0 and percentage <= 1.0

        nominal_values = np.array(self.nominal_values if nominal_values is None else nominal_values)
        task_dim = nominal_values.shape[0]

        if dyn_mask is None:
            dyn_mask = list(range(task_dim))
        else:
            assert np.all(np.array(dyn_mask) < task_dim) and np.all(np.array(dyn_mask) >= 0), f'Specify the index of the dynamics parameter to randomize, starting from 0. Number of current parameters is {task_dim}'
        
        if lower_bounds is None:
            lower_bounds = np.array([self.get_search_bounds_mean(i)[0] for i in range(task_dim)])

        dr_percentage_per_dim = np.zeros(task_dim)
        dr_percentage_per_dim[dyn_mask] = percentage

        deviation = np.multiply(nominal_values-lower_bounds, dr_percentage_per_dim)  # element-wise mult: [ N nominal values ] * [ N DR percentages ]
        bounds_low = nominal_values - deviation
        bounds_high = nominal_values + deviation
        bounds = np.vstack((bounds_low,bounds_high)).reshape((-1,),order='F')  # alternating bounds from the low and high bounds
        
        return bounds

