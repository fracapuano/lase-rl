"""General Gym Wrapper (https://www.gymlibrary.dev/api/wrappers/#general-wrappers)
    for keeping track of dynamics and associated returns when training with DORAEMON.
"""
import pdb

import gymnasium as gym
import numpy as np
import torch

class ReturnDynamicsTrackerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.overrides = {}

        # Override wrapped env set_random_task method
        self.overrides['set_random_task'] = self.doraemon_custom_set_random_task
        self.buffer = []  # keep track of (dynamics, return) tuples
        self.succ_metric_buffer = []  # buffer with metric used for measuring success

        self.cum_reward = 0
        self.curr_episode_dynamics = None
        self.ready_to_update_buffer = False
        self.expose_episode_stats = False

        """Certain environments may define a success metric different from return"""
        if hasattr(self.env.unwrapped, "success_metric"):
            self.success_metric = self.env.unwrapped.success_metric

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated
        if self.expose_episode_stats:
            # Compute return over time
            self.cum_reward += reward

            if done and self.ready_to_update_buffer:
                self._update_buffer(
                    episode_dynamics=self.get_wrapper_attr("get_task")(),
                    episode_return=self.cum_reward
                )

                if hasattr(self, "success_metric"):
                    self._update_succ_metric_buffer(
                        succ_metric=getattr(
                            self.env.unwrapped, 
                            self.success_metric
                        )
                    )

                self.cum_reward = 0
                self.ready_to_update_buffer = False

        return next_state, reward, terminated, truncated, info

    def doraemon_custom_set_random_task(self):
        """Keep track of sampled task and its return"""
        task = self.sample_task()
        self.set_task(*task)

        if self.expose_episode_stats:
            self.cum_reward = 0
            self.curr_episode_dynamics = np.array(task, dtype=np.float64)
            self.ready_to_update_buffer = True

    def _update_buffer(self, episode_dynamics, episode_return):
        self.buffer.append({'dynamics': episode_dynamics, 'return': episode_return})

    def _update_succ_metric_buffer(self, succ_metric):
        self.succ_metric_buffer.append(succ_metric)

    def reset_buffer(self):
        self.buffer = []
        self.succ_metric_buffer = []

    def get_buffer(self):
        return self.buffer

    def get_succ_metric_buffer(self):
        return self.succ_metric_buffer

    def set_expose_episode_stats(self, flag):
        self.expose_episode_stats = flag

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.expose_episode_stats:
            self.cum_reward = 0
            self.ready_to_update_buffer = True
            
        if hasattr(self.env, 'dr_training') and self.env.dr_training:
            self.set_random_task()
            
        return obs, info

    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return seed

    def set_dr_distribution(self, *args, **kwargs):
        """Pass through set_dr_distribution to the underlying environment"""
        if hasattr(self.env, 'set_dr_distribution'):
            return self.env.set_dr_distribution(*args, **kwargs)
        return None

    def get_dr_distribution(self):
        """Pass through get_dr_distribution to the underlying environment"""
        if hasattr(self.env, 'get_dr_distribution'):
            return self.env.get_dr_distribution()
        return None

    def set_dr_training(self, flag):
        """Pass through set_dr_training to the underlying environment"""
        if hasattr(self.env, 'set_dr_training'):
            return self.env.set_dr_training(flag)
        return None

    def get_dr_training(self):
        """Pass through get_dr_training to the underlying environment"""
        if hasattr(self.env, 'get_dr_training'):
            return self.env.get_dr_training()
        return None

    def get_reward_threshold(self):
        """Pass through get_reward_threshold to the underlying environment"""
        if hasattr(self.env, 'get_reward_threshold'):
            return self.env.get_reward_threshold()
        return None

