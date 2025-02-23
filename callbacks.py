import numpy as np
import torch
import wandb
from typing import Tuple
from laserenv.BaseLaser import AbstractBaseLaser

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

class StatsAggregator:
    def __init__(self): 
        self.flush()

    def lasertraining_callback(self, locals_dict:dict, globals_dict:dict):
        """
        Uses access to locals() to elaborate information. 
        Intended to be used inside of stable_baselines3 `evaluate_policy`
        """
        if locals_dict["done"]:
            self.stats["controls"].append(locals_dict["info"].get("current_control (picoseconds)"))
            self.stats["peak_intensity"].append(locals_dict["info"].get("current Peak Intensity (TW/m^2)"))
            self.stats["fwhm"].append(locals_dict["info"].get("current FWHM (ps)"))
            self.stats["tl_regret"].append(locals_dict["info"].get("TL-L1Loss"))
            self.stats["x_t(perc)"].append(locals_dict["info"].get("x_t(perc)"))
    
    def flush(self):
        self.stats = {
            "controls": [],
            "peak_intensity": [],
            "fwhm": [],
            "tl_regret": [],
            "x_t(perc)": []
        }

class FROGWhileTrainingCallback(BaseCallback): 
    """Custom callback inheriting from `BaseCallback`.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug.

    Performs various actions when triggered (intended to be a child of EventCallback): 
        1. Evaluates current policy (for n_eval_episodes)
        2. Updates a current best_policy variable
        3. Logs stuff on wandb. More details on what is logged in :meth:_on_step.
    """
    def __init__(
            self, 
            env:AbstractBaseLaser|VecEnv,
            verbose:int=0,
            n_eval_episodes:int=50,
            best_model_path:str="models/"
        ):
        super().__init__(verbose)

        self.episode_stats = StatsAggregator()

        self._envs = env
        self.n_eval_episodes = n_eval_episodes

        # resets environment
        self._envs.reset()
        
        # current best model and best model's return in test trajectories
        self.best_model_path = best_model_path
        self.best_model = None
        self.best_model_mean_reward = -np.inf
        self.bests_found = 0

    def get_rollout_frames(self):
        obs = self._envs.reset()
        max_steps = self.model.get_env().envs[0].unwrapped.MAX_STEPS
        frames = []
        
        for _ in range(max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self._envs.step(action)
            frame = self._envs.render()
            frames.append(frame)
        
        return frames

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # flush past statistics
        self.episode_stats.flush()

        # obtain mean and std of cumulative reward over n_eval_episodes
        mean_cum_reward, std_cum_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            n_eval_episodes=self.n_eval_episodes,
            callback=self.episode_stats.lasertraining_callback,
        )
        
        std_cum_reward = std_cum_reward if std_cum_reward > 0 else 1e-6
        
        frog_traces = torch.stack(
            [
                self.model.get_env().env_method("frog_trace", control)[0]
                for control in self.episode_stats.stats["controls"]
            ],
            dim=0
        )

        # rolling out the env for one episode to capture a video
        frames_rollout = self.get_rollout_frames()

        wandb.log({
            "episode": wandb.Video(np.stack(frames_rollout).transpose(0, 3, 1, 2), fps=5),
            "final_frog_trace": wandb.Image(torch.mean(frog_traces, dim=0)),
            "final_intensity_avg": np.mean(self.episode_stats.stats["peak_intensity"]),
            "final_fwhm_avg": np.mean(self.episode_stats.stats["fwhm"]),
            "final_tl_regret_avg": np.mean(self.episode_stats.stats["tl_regret"]),
            "final_intensity_x_t": np.mean(self.episode_stats.stats["x_t(perc)"]),
            "final_intensity_std": np.std(self.episode_stats.stats["peak_intensity"]),
            "final_fwhm_std": np.std(self.episode_stats.stats["fwhm"]),
            "final_tl_regret_std": np.std(self.episode_stats.stats["tl_regret"]),
            "final_intensity_x_t_std": np.std(self.episode_stats.stats["x_t(perc)"]),
        })
        
        # checks if this model is better than current best. If so, update current best
        if mean_cum_reward >= self.best_model_mean_reward:
            self.best_model = self.model
            self.best_model_mean_reward = mean_cum_reward
            # save best model locally
            model_path = f"{self.best_model_path}/best_model.zip"
            self.best_model.save(path=model_path)
            # save to wandb
            wandb.save(model_path)
            self.bests_found += 1

        wandb.log({
             "Mean Cumulative Reward": mean_cum_reward,
             "Std of Cumulative Reward": std_cum_reward,
        })
        return True
    
    def get_best_model(self, return_reward:bool=True): 
        if return_reward:
            return self.best_model, self.best_model_mean_reward
        else: 
            return self.best_model
