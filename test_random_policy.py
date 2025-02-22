import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

from laserenv.LaserEnv import FROGLaserEnv
from laserenv.env_utils import EnvParametrization

def create_gif_from_frames(frames, episode_num):
    """Create and save a GIF from a list of observation frames"""
    # Convert frames to PIL Images if they're numpy arrays
    pil_frames = []
    for frame in frames:
        pil_frames.append(Image.fromarray(frame))
    
    # Create output directory if it doesn't exist
    os.makedirs('episode_gifs', exist_ok=True)
    
    # Save the GIF
    pil_frames[0].save(
        f'episode_gifs/episode_{episode_num}.gif',
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # Duration between frames in milliseconds
        loop=0
    )

def collect_reward_data(n_timesteps=100):
    # Initialize environment
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()
    
    env = FROGLaserEnv(
        bounds=bounds,
        compressor_params=compressor_params,
        B_integral=B_integral,
        render_mode=None,
        device="mps"
    )
    
    # Data structures to store reward components
    reward_history = defaultdict(list)
    total_rewards = []
    
    obs, info = env.reset()
    
    for _ in tqdm(range(n_timesteps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store reward components
        for component_name, value in info.items():
            if 'component' in component_name:
                reward_history[component_name].append(value)
        total_rewards.append(reward)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    return reward_history, total_rewards

def plot_reward_analysis(reward_history, total_rewards):
    components = list(reward_history.keys())
    n_components = len(components)

    fig, axs = plt.subplots(n_components + 1, 1, figsize=(12, 3*(n_components + 1)))
    fig.suptitle('Reward Components Analysis', fontsize=16)
    
    # Plot individual components
    for idx, component in enumerate(components):
        values = reward_history[component]
        axs[idx].plot(values, label=component, alpha=0.7)
        axs[idx].set_title(f'{component} over time')
        axs[idx].set_xlabel('Timesteps')
        axs[idx].set_ylabel('Value')
        axs[idx].grid(True)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        axs[idx].axhline(y=mean_val, color='r', linestyle='--', alpha=0.5)
        axs[idx].text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}', 
                     transform=axs[idx].transAxes, 
                     bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot total reward
    axs[-1].plot(total_rewards, label='Total Reward', color='black', alpha=0.7)
    axs[-1].set_title('Total Reward over time')
    axs[-1].set_xlabel('Timesteps')
    axs[-1].set_ylabel('Value')
    axs[-1].grid(True)
    
    # Add statistics for total reward
    mean_total = np.mean(total_rewards)
    std_total = np.std(total_rewards)
    axs[-1].axhline(y=mean_total, color='r', linestyle='--', alpha=0.5)
    axs[-1].text(0.02, 0.95, f'Mean: {mean_total:.3f}\nStd: {std_total:.3f}', 
                 transform=axs[-1].transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    plt.show()

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Test random policy for laser environment')
    parser.add_argument('--record-frames', action='store_true', help='Record frames as GIFs')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    args = parser.parse_args()

    # Use default environment parameters from EnvParametrization
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()

    # Create the environment (without rendering)
    env = FROGLaserEnv(
        bounds=bounds,
        compressor_params=compressor_params,
        B_integral=B_integral,
        render_mode="human" if args.render else "rgb_array"
    )

    # Reset the environment to obtain the initial observation and info
    obs, info = env.reset()
    
    done = False
    start_time = time.time()
    episode_count = 0
    frames = []

    for _ in tqdm(range(100)):
        # Sample an action at random from the environment's action space.
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Store the observation frame
        obs_or_frame = env.render() if not args.render else obs["frog_trace"]
        frames.append(obs_or_frame)

        done = terminated or truncated

        if args.render:
            env.render()
        
        if done:
            # Create GIF for the completed episode
            if args.record_frames:
                create_gif_from_frames(frames, episode_count)
            
            episode_count += 1
            # Reset for next episode
            obs, info = env.reset()
            obs_or_frame = env.render() if not args.render else obs["frog_trace"]
            frames = [obs_or_frame]


    # Save the last episode if it's not done
    if frames and args.record_frames:
        create_gif_from_frames(frames, episode_count)

    total_time = time.time() - start_time
    print(f"Rollout complete. Total time: {total_time:.6f} seconds")
    if args.record_frames:
        print(f"Created {episode_count + 1} episode GIFs in the 'episode_gifs' directory")

if __name__ == "__main__":
    # reward_history, total_rewards = collect_reward_data(100)
    # plot_reward_analysis(reward_history, total_rewards)

    main()