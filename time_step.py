import numpy as np
import time
import gymnasium as gym

"""Comparing the step method fo LaserEnv to an (unrealistically) fast Hopper env, aiming at improving speed"""
env = gym.make("Hopper-v5")

obs, info = env.reset()

times = []
for _ in range(10):
    action = env.action_space.sample()
    start_time = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    end_time = time.time()
    times.append(end_time - start_time)
    
print(f"Time taken (hopper): {np.mean(times)} seconds")

import laserenv
env = gym.make("LaserEnv")

obs, info = env.reset()

times = []
for _ in range(10):
    action = env.action_space.sample()
    start_time = time.time()
    obs, reward, terminated, truncated, info = env.step(action)
    end_time = time.time()
    times.append(end_time - start_time)
    
print(f"Time taken (laserenv): {np.mean(times)} seconds")