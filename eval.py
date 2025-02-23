import gymnasium as gym
from stable_baselines3 import PPO, SAC
import laserenv

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def main():
    render = True
    
    def make_env():
        env = gym.make("LaserEnv", render_mode="human" if render else "rgb_array")
        return env
    
    env = VecFrameStack(DummyVecEnv([make_env]), n_stack=5)
    model = SAC.load("runs/vivid-sponge-110/model.zip")
    
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
