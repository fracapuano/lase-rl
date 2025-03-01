import torch
import gymnasium as gym
import numpy as np
import wandb
from policy.policy import Policy
import laserenv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from callbacks import FROGWhileTrainingCallback
from wandb.integration.sb3 import WandbCallback

# First, define which observations should be available to actor and critic
actor_obs_keys = ["frog_trace", "psi", "action"]  # Limited info for actor
critic_obs_keys = ["frog_trace", "psi", "action", "B_integral", "compressor_GDD"]  # Full info for critic

# Create environment (assuming FROGLaserEnv or similar)
def make_env():
    env = gym.make("LaserEnv", render_mode="human", udr=True)
    return env

env = DummyVecEnv([make_env for _ in range(4)])
env = VecFrameStack(env, n_stack=5)

# Create the policy with asymmetric information
policy = Policy(
    algo='sac',
    env=env,
    lr=3e-4,
    gamma=0.99,
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    # These masks define which observations are passed to actor/critic
    actor_obs_mask=actor_obs_keys,
    critic_obs_mask=critic_obs_keys,
    gradient_steps=-1  # Use as many as number of envs
)

# extracting the model
model = policy.model
timesteps = 200_000

    
run = wandb.init(
    project="RLC-Laser",
    sync_tensorboard=True,
    monitor_gym=True,
    config={
        "algorithm": "sac",
        "timesteps": timesteps,
        "learning_rate": 3e-4,
        "seed": 42,
        "frame_stack": 5,
        "n_envs": 4,
        "udr": True
    },
    notes="Asymmetric SAC with UDR",
)

# Setup the Wandb callback to log training progress, including gradient information.
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    verbose=2
)

frog_callback = FROGWhileTrainingCallback(
    env=env,
    n_eval_episodes=10,
    best_model_path="./"
)

frog_callback = EveryNTimesteps(
    n_steps=5000,
    callback=frog_callback
)

callback = CallbackList([
    wandb_callback,
    frog_callback
])

# Begin training using total_timesteps specified in wandb config
model.learn(
    total_timesteps=timesteps,
    callback=callback, 
    progress_bar=True
)

wandb.finish()