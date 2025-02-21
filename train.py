import wandb
from stable_baselines3 import SAC, PPO
from wandb.integration.sb3 import WandbCallback

from laserenv.LaserEnv import FROGLaserEnv
from laserenv.env_utils import EnvParametrization

from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder, VecFrameStack
from stable_baselines3.common.monitor import Monitor

import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"],
                      help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                      help="Total timesteps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                      help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--eval-every", type=int, default=10000,
                      help="Evaluate and record video every N steps")
    return parser.parse_args()

def main():
    args = parse_args()
    
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
        },
    )
    
    # Create run directory for all assets
    run_dir = f"runs/{run.name}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Use default environment parameters from EnvParametrization
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()

    def make_env():
        # Create the environment (without rendering)
        env = FROGLaserEnv(
            bounds=bounds,
            compressor_params=compressor_params,
            B_integral=B_integral,
            device="mps"
        )

        env = Monitor(env)
        return env

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=5)
    env = VecVideoRecorder(
        env,
        f"{run_dir}/videos",
        record_video_trigger=lambda x: x % args.eval_every == 0,
        video_length=20
    )


    # Initialize the model with selected algorithm
    algo_class = PPO if args.algo == "PPO" else SAC
    model = algo_class(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        tensorboard_log=f"{run_dir}/tensorboard",
        seed=args.seed,
        verbose=0,
        device="mps"
    )

    # Setup the Wandb callback to log training progress, including gradient information.
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )

    # Begin training using total_timesteps specified in wandb config
    model.learn(
        total_timesteps=args.timesteps, 
        callback=[wandb_callback], 
        progress_bar=True
    )

    # Save the trained model for later use
    model.save(f"{run_dir}/model.zip")

    # Finish the wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
