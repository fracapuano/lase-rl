import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import os

from laserenv.LaserEnv import FROGLaserEnv
from laserenv.env_utils import EnvParametrization

record_frames = False
render = True

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

def main():
    # Use default environment parameters from EnvParametrization
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()

    # Create the environment (without rendering)
    env = FROGLaserEnv(
        bounds=bounds,
        compressor_params=compressor_params,
        B_integral=B_integral,
        render_mode="human"
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
        frames.append(obs)

        done = terminated or truncated

        env.render()
        
        if done:
            # Create GIF for the completed episode
            if record_frames:
                create_gif_from_frames(frames, episode_count)
            
            episode_count += 1
            # Reset for next episode
            obs, info = env.reset()
            frames = [obs]  # Initialize with first frame of new episode

    # Save the last episode if it's not done
    if frames and record_frames:
        create_gif_from_frames(frames, episode_count)

    total_time = time.time() - start_time
    print(f"Rollout complete. Total time: {total_time:.6f} seconds")
    if record_frames:
        print(f"Created {episode_count + 1} episode GIFs in the 'episode_gifs' directory")

if __name__ == "__main__":
    main()
