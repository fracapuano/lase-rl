import gymnasium as gym
from stable_baselines3 import PPO, SAC
import laserenv

import numpy as np

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from utils import make_wrapped_environment
from laserenv.RandomVecEnv import RandomDummyVecEnv
from argparse import Namespace
from laserenv.env_utils import EnvParametrization

def main():
    render = True
    dr = False  # whether to randomize or not at eval time
    env_name = "RandomLaserEnv"

    env = make_vec_env(
        env_name,
        n_envs=1,
        seed=0,
        vec_env_cls=RandomDummyVecEnv,
        wrapper_class=make_wrapped_environment,
        wrapper_kwargs={'args': Namespace(stack_history=5)},
        env_kwargs={'render_mode': 'human' if render else 'rgb_array'}
    )
    
    gt_task = gym.make(env_name).unwrapped.get_task()
    bounds_low, bounds_high = EnvParametrization().get_bounds(parameters=["B", "GDD"])

    env.set_dr_distribution(
        dr_type='uniform', 
        distr=[x for i in range(len(gt_task)) for x in (bounds_low[i], bounds_high[i])]
    )
    env.env_method("set_dr_training", dr)

    model = SAC.load("best_model.zip")
    
    # Evaluation loop
    average_reward = 0
    for _ in range(10):
        obs = env.reset()
        
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()
    
        average_reward += episode_reward

    print(f"Average Reward: {average_reward / 10}")
    env.close()

if __name__ == "__main__": 
    main()
