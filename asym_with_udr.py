import torch
import gymnasium as gym
import wandb
from policy.policy import Policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps
from callbacks import FROGWhileTrainingCallback
from wandb.integration.sb3 import WandbCallback


def main():
    # First, define which observations should be available to actor and critic
    actor_obs_keys = ["frog_trace", "psi", "action"]  # Limited info for actor
    critic_obs_keys = ["frog_trace", "psi", "action", "B_integral", "compressor_GDD"]  # Full info for critic

    # Number of vectorized environments to run
    n_envs = 4
    # How many frames to stack when forming an observation
    frame_stack = 5

    # Bounds for UDR
    udr_low = 1.5
    udr_high = 2.5

    # Create environment (assuming FROGLaserEnv or similar)
    def make_env():
        env = gym.make("LaserEnv", render_mode="human", udr=True, udr_low=udr_low, udr_high=udr_high)
        return env

    enable_frog_callback = True

    # First, define which observations should be available to actor and critic
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=frame_stack)

    # Create the policy with asymmetric information
    policy = Policy(
        algo="sac",
        env=env,
        lr=3e-4,
        gamma=0.9,
        device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        # These masks define which observations are passed to actor/critic
        actor_obs_mask=actor_obs_keys,
        critic_obs_mask=critic_obs_keys
    )

    # extracting the model
    model = policy.model
    timesteps = 200_000

    run = wandb.init(
        project="RLC-Laser",
        sync_tensorboard=True,
        monitor_gym=True,
        config={
            "algorithm": "asym-sac",
            "timesteps": timesteps,
            "learning_rate": 3e-4,
            "frame_stack": frame_stack,
            "n_envs": n_envs,
            "udr": True,
            "udr_low": udr_low,
            "udr_high": udr_high
        },
        notes="Asymmetric SAC with UDR",
    )

    # Setup the Wandb callback to log training progress, including gradient information.
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        verbose=2
    )

    callback = CallbackList([
        wandb_callback,
    ])

    if enable_frog_callback:
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


    # Begin training
    model.learn(
        total_timesteps=timesteps,
        callback=callback, 
        progress_bar=True
    )

    # save model
    model.save(f"asym_with_udr_{udr_low}_{udr_high}.zip")

    wandb.finish()

if __name__ == "__main__":
    main()