import wandb
from stable_baselines3 import SAC, PPO

from laserenv.LaserEnv import FROGLaserEnv
from laserenv.env_utils import EnvParametrization

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
    VecFrameStack,
)
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from callbacks import FROGWhileTrainingCallback
from wandb.integration.sb3 import WandbCallback

import os
import torch
import argparse

def get_device(return_cpu:bool=False):
    if return_cpu:
        return "cpu"
    else:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="SAC", choices=["PPO", "SAC"],
                      help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=50_000,
                      help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--eval-every", type=int, default=5_000,
                      help="Evaluate and record video every N steps")
    return parser.parse_args()

def main():
    args = parse_args()
    n_envs = 4
    device = get_device(return_cpu=False)
    
    run = wandb.init(
        project="RLC-Laser",
        sync_tensorboard=True,
        monitor_gym=True,
        config={
            "policy": "CnnPolicy",
            "algorithm": args.algo,
            "timesteps": args.timesteps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "frame_stack": 5,
            "n_envs": n_envs
        },
    )
    
    # Create run directory for all assets
    run_dir = f"runs/{run.name}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Use default environment parameters from EnvParametrization
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()

    def make_env():
        env = FROGLaserEnv(
            bounds=bounds,
            compressor_params=compressor_params,
            B_integral=B_integral,
            device=device
        )

        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=5)

    # Initialize the model with selected algorithm
    algo_class = PPO if args.algo == "PPO" else SAC
    model = algo_class(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        tensorboard_log=f"{run_dir}/tensorboard",
        seed=args.seed,
        verbose=0,
        device=device,
    )

    # Setup the Wandb callback to log training progress, including gradient information.
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )

    frog_callback = FROGWhileTrainingCallback(
        env=env,
        n_eval_episodes=10,
        best_model_path=run_dir
    )

    frog_callback = EveryNTimesteps(
        n_steps=args.eval_every, 
        callback=frog_callback
    )

    callback = CallbackList([
        wandb_callback,
        frog_callback
    ])

    # Begin training using total_timesteps specified in wandb config
    model.learn(
        total_timesteps=args.timesteps, 
        callback=callback, 
        progress_bar=True
    )

    # Save the trained model and normalization statistics
    model.save(f"{run_dir}/model.zip")

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()

