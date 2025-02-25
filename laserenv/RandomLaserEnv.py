import line_profiler

import torch

import numpy as np
from typing import Tuple, List
from collections import deque
from gymnasium.spaces import Box, Dict
from torch.distributions.multivariate_normal import MultivariateNormal

import pygame
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional
from laserenv.RandomBaseLaser import RandomBaseLaser
from laserenv.env_utils import ControlUtils
from laserenv.utils import physics
from laserenv.utils.render import (
    visualize_pulses, 
    visualize_controls,
    visualize_frog,
    visualize_reward
)
from laserenv.env_utils import extract_central_window

# this way, figures are not automatically shown
plt.ioff()

class RandomFROGLaserEnv(RandomBaseLaser):
    metadata = {
        "render_fps": 5,
        "render_modes": ["rgb_array", "human"]
    }
    """Instances a physics-informed version of the L1 Pump laser.
    
    In this version, observations are FROG traces obtained from applying phase control psi on the system.
    Actions are deltas on said configurations (changes made on psi), bounded for machine safety.
    
    This environment implements priviledged information, and the reward accesses the temporal 
    profile of the controlled pulse. Episode termination is declared once the pulse's FWHM goes 
    over a predefinite threshold. We also use an alive bonus, hoping to incentivize the agent to stay alive.
    """
    def __init__(
        self,
        bounds:torch.TensorType,
        compressor_params:torch.TensorType,
        B_integral:float,
        render_mode:str="rgb_array",
        action_bounds:Tuple[float, List[float]]=0.1,
        init_variance:float=.1,
        device:str="cpu",
        window_size:int=64,
        env_kwargs:dict={}
    ) -> None:
        # env parametrization init - chagepoint for different xi's.
        super().__init__(
            # the \xi parameters define the dynamics of the laser system
            compressor_params=compressor_params,
            B_integral=B_integral,
            bounds=bounds,
            render_mode=render_mode, 
            device=device
        )

        self.dyn_ind_to_name = {
            0: "B_integral",
            1: "compressor_GDD"
        }
        self.success_metric = "peak_intensity_ratio"
        
        # Recording the actual dynamics bounds
        self.min_task = [1, 2.47]
        self.max_task = [3.5, 2.87]

        """Specifying observation space"""
        self.psi_dim = 3
        self.window_size = window_size

        # actions are deltas on the control parameters, \delta \psi
        self.action_dim = 3

        # specifiying obs space, as Dict containing:
        # - (self.window_size x self.window_size) B&W FROG traces (Box space)
        # - current control parameters psi (Box space)
        # - action, to guide system identification (Box space)
        self.observation_space = Dict({
            "frog_trace": Box(
                low=0, high=255, shape=(1, self.window_size, self.window_size), dtype=np.uint8
            ),
            "psi": Box(
                low=0.0, high=1.0, shape=(self.psi_dim,), dtype=np.float32
            ),
            "action": Box(
                low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
            ),
            "B_integral": Box(
                low=1, high=3.5, shape=(1,), dtype=np.float32
            ),
            # storing the compressor parameters as GDD(ps^2)/100 for numerical stability
            "compressor_GDD": Box(
                low=2.47, high=2.87, shape=(1,), dtype=np.float32
            )
        })

        """Parsing action-dependant parameters"""
        # custom bounds for env
        if isinstance(action_bounds, list): 
            self.action_lower_bound, self.action_upper_bound = action_bounds
        else:
            self.action_lower_bound, self.action_upper_bound = -action_bounds, +action_bounds
        
        # action range
        self.action_range = self.action_upper_bound - self.action_lower_bound
        
        # actions are defined as deltas, with bounded updates
        self.action_space = Box(
            low = -1*np.ones(self.action_dim, dtype=np.float32), 
            high= +1*np.ones(self.action_dim, dtype=np.float32)
        )

        """Parsing reward-dependant parameters"""
        # laser characteristic specifics
        self.transform_limited = self.laser.transform_limited()
        # control utils - suite to handle with ease normalization of control params
        self.control_utils = ControlUtils()  # initialized with default parameters 

        # defining maximal number of steps and duration (ps) for the pulse
        self.MAX_DURATION=env_kwargs.get("max_duration", 20)
        self.MAX_STEPS=env_kwargs.get("max_steps", 20)
        # physically, the Transform-Limited (TL) pulse maximizes the intensity
        self.TL_intensity = physics.peak_intensity(
            pulse_intensity=self.transform_limited[1]
        )

        """Setting the distribution over initial control conditions."""
        self.rho_zero = MultivariateNormal(
            # loc is compressor params (in the 0-1 range)
            loc=self.control_utils.demagnify_scale(-1*self.compressor_params).float(), 
            # homoscedastic distribution
            covariance_matrix=torch.diag(init_variance * torch.ones(self.action_dim))
        )

        # buffer to store controls
        self.controls_buffer = deque(maxlen=5)
        
        # setting the simulator in empty state
        self.reset()

    def get_default_task(self):
        return np.array([2, 2.67])  # nominal values as per system specification
    
    def get_search_bounds_mean(self, index):
        """Get search space for current randomized parameter at index `index`"""
        search_bounds_mean = {
               "B_integral": (1, 3.5),
               "compressor_GDD": (2.47, 2.87)
        }
        return search_bounds_mean[self.dyn_ind_to_name[index]]

    def get_task_lower_bound(self, index):
        """Returns lowest feasible value for current randomized parameter at index `index`"""
        return self.get_search_bounds_mean(index)[0]

    def get_task_upper_bound(self, index):
        """Returns highest feasible value for current randomized parameter at index `index`"""
        return self.get_search_bounds_mean(index)[1]

    def get_task(self):
        """Get current dynamics parameters"""
        return np.array([self.laser.B, self.dispersion_coefficient_for_dynamics(index=0)])

    def set_task(self, *task):
        """Set dynamics parameters to <task>"""
        B_integral, GDD = task
        self.laser.overwrite_B_integral(B_integral)        

        self.laser.overwrite_single_compressor_params(
            index=0,
            new_compressor_param = -100*GDD*1e-24  # -(s^-12)/100 -> s^2
        )
    
    @property
    def psi(self):
        """Returns control parameters in the 0-1 range."""
        return self._psi
    
    @psi.setter
    def psi(self, value):
        """Sets control parameters ensuring they are in NumPy array format."""
        if isinstance(value, torch.Tensor):
            self._psi = value.type(torch.float32).detach().cpu().numpy()
        else:
            self._psi = value
        
    @property
    def psi_picoseconds(self):
        """
        Returns control parameters in bounds, expressed in picoseconds^k.
        Expressing control parameters in picoseconds allows to express them in float32,
        allowing for MPS compatibility in acceleration.
        """
        # Convert numpy array to tensor, process, and return as tensor
        psi_tensor = torch.from_numpy(self._psi).to(self.device)
        return self.control_utils.descale_control(psi_tensor).type(torch.float32)

    @property
    def pulse(self):
        """Returns the temporal profile of the pulse that derives from the current observation"""
        time, control_shape = self.laser.control_to_temporal(self.psi_picoseconds)
        return (time, control_shape)
    
    @property
    def frog(self):
        """Returns the FROG trace of the current control parameter."""
        return self.laser.control_to_frog(self.psi_picoseconds)
    
    @property
    def pulse_FWHM(self):
        """Returns pulse full-width half-maximum. FWHM is given in picoseconds."""
        time, control_shape = self.pulse
        return physics.FWHM(x=time, y=control_shape) * 1e12  # seconds -> picoseconds

    @property
    def peak_intensity(self): 
        """Returns peak intensity of the controlled shape un-doing intensity normalization."""
        return physics.peak_intensity(pulse_intensity=self.pulse[-1])
    
    def frog_trace(self, control_ps: torch.TensorType) -> torch.TensorType:
        """Returns the FROG trace of the given control parameters. Mostly used for logging."""
        return self.laser.control_to_frog(control_ps)

    def transform_limited_regret(self):
        """Computes aligned-L1 loss between current pulse and transform limited"""
        # obtain the pulse shape corresponding to given set of control params
        time, control_shape = self.pulse
        target_time, target_shape = self.transform_limited
        
        # move target and controlled pulse peak on peak
        pulse1, pulse2 = physics.peak_on_peak(
            temporal_profile=[time.cpu(), control_shape.cpu()], 
            other=[target_time.cpu(), target_shape.cpu()]
            )
        
        # compute sum(L1 loss)
        return (pulse1[1] - pulse2[1]).abs().sum().item()

    def dispersion_coefficient_for_dynamics(self, index: int):
        """Returns the dispersion coefficient for the dynamics at index `index` in the dispersion vector."""
        return -0.01 * self.control_utils.controls_demagnify(
            self.laser.compressor_params).type(torch.float32
        )[index]

    def _get_obs(self): 
        """Return observation."""
        frog_trace = self.frog.cpu().numpy()  # Move to CPU only at the end
        central_window = extract_central_window(frog_trace, window_size=self.window_size)
        
        compressor_GDD = self.dispersion_coefficient_for_dynamics(index=0)
        
        obs = {
            "frog_trace": 255 * central_window.reshape(1, *central_window.shape).astype(np.uint8),
            "psi": self.psi,
            "action": self.action,
            "B_integral": np.array([self.laser.B], dtype=np.float32),
            "compressor_GDD": np.array([compressor_GDD], dtype=np.float32),
        }

        return obs

    def _get_info(
            self, 
            terminated:Optional[bool]=None, 
            truncated:Optional[bool]=None, 
            reward_components:Optional[dict]=None
        ) -> dict:
        """Return state-related info."""
        info = {
            "timesteps": self.n_steps,
            "current_control": self.psi,
            "current_control (picoseconds)": self.psi_picoseconds,
            "current FWHM (ps)": self.pulse_FWHM,
            "current Peak Intensity (TW/m^2)": self.peak_intensity * 1e-12,
            "x_t(perc)": 100 * self.peak_intensity / self.TL_intensity,
            "TL-L1Loss": self.transform_limited_regret(),
            "FWHM-failure": terminated if terminated is not None else False,
            "Timesteps-failure": truncated if truncated is not None else False,
        }
        if reward_components is not None:
            info.update(reward_components)
        
        self._info = info
        
        return info
    
    def remap_action(self, action:np.ndarray)->np.ndarray:
        """
        Remaps the action from [-1, +1] to [lower_bound, upper_bound] range.
        Args: 
            action (np.array): Action sampled from self.action_space (in the [-1, +1] range).
        Returns: 
            np.array: Action in the [self.action_lower_bound, self.action_upper_bound] range.
        """
        # First normalize from [-1,1] to [0,1], then scale to target range
        normalized = (action + 1) / 2
        return self.action_lower_bound + normalized * self.action_range

    def reset(self, seed:int=None, options=None) -> Tuple[np.ndarray, dict]:
        """Resets the environment to initial observations"""
        # Set the seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # number of steps
        self.n_steps = 0
        
        # starting in a random state - sample from torch distribution but convert to numpy
        sampled_tensor = self.rho_zero.sample()
        clipped_tensor = torch.clip(
            sampled_tensor, 
            torch.zeros(self.action_dim), 
            torch.ones(self.action_dim)
        )
        self.psi = clipped_tensor.type(torch.float32).detach().cpu().numpy()  # Store as numpy

        self.action = torch.zeros(self.action_dim).cpu().numpy()
        
        # Clear the buffer and add initial control
        self.controls_buffer.clear()
        self.controls_buffer.append(self.psi.copy())

        self.get_reward()

        return self._get_obs(), self._get_info()

    def is_terminated(self) -> bool:
        """
        Checks whether the episode should be terminated due to failure conditions,
        such as the pulse full-width half maximum (FWHM) exceeding the maximum duration.
        
        Returns:
            bool: True if pulse_FWHM is equal to or exceeds MAX_DURATION.
        """
        terminated = self.pulse_FWHM >= self.MAX_DURATION
        return bool(terminated) and False

    def is_truncated(self) -> bool:
        """
        Checks whether the episode should be truncated due to the maximum number of timesteps being exceeded.
        
        Returns:
            bool: True if the number of steps n_steps is equal or exceeds MAX_STEPS.
        """
        truncated = self.n_steps >= self.MAX_STEPS
        return bool(truncated)

    def get_reward(self)->float:
        """
        This function computes the reward associated with the (state, action) pair. 
        This reward function is made up of several different components and derives from fundamental assumptions
        made on the values that each term can take on.

        Returns: 
            float: Value of reward. Sum of two different components. Namely: Alive Bonus and Intensity (either gain or pure value)
        """
        healthy_reward = 2  # small constant, reward for having not failed yet.
        x = self.peak_intensity / self.TL_intensity

        """This peak intensity ratio is used to assess whether the episode concluded successfully"""
        if hasattr(self, "success_metric"):
            setattr(self, self.success_metric, x)

        a, x = 0.2, min(1, x)
        intensity_reward = min(x+(a / (1-x)) - a, 7)
        
        alive_component = healthy_reward
        intensity_component = intensity_reward
        duration_component = -1 * self.pulse_FWHM

        final_reward = (alive_component + duration_component) + intensity_component
        final_reward *= 1e-1  # scaling down the reward
        
        final_reward = intensity_component - 0.1
        components = {
            "alive_component": alive_component,
            "intensity_component": intensity_component,
            "duration_component": duration_component,
            "final_reward": final_reward
        }

        self.reward_components = components

        return final_reward, components

    def step(self, action:torch.TensorType)->Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Applies given action on laser env. Returns observation, reward, terminated, truncated, info
        """
        # increment number of steps
        self.n_steps += 1
        # overwriting currently stored action with incoming action
        self.action = action
        # scaling the action to the actual range

        rescaled_action = self.remap_action(action=action)
        # applying (rescaled) action, clipping between 0 and 1
        self.psi = np.clip(
            self.psi + rescaled_action, 
            np.zeros(self.action_dim), 
            np.ones(self.action_dim),
            dtype=np.float32
        )
        
        self.controls_buffer.append(self.psi)
        reward, components = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self._get_info(
            terminated=terminated, 
            truncated=truncated, 
            reward_components=components
        )

        if terminated:
            reward = -5  # penalty for terminating the episode
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _render_pulse(self)->np.array: 
        """Renders pulse shape against target.
        
        Returns:
            np.array: PIL image as rgb array. 
        """
        # retrieving control and target pulse and time axes
        time, control_pulse = self.pulse
        target_time, target_pulse = self.transform_limited
        
        # using rendering functions to show off pulses
        fig, ax = visualize_pulses(
            [time, control_pulse], 
            [target_time, target_pulse]
        )
        
        # specializing the plots to showcase control trajectories
        title_string = f"Timestep {self.n_steps}/{self.MAX_STEPS}"
        if self.n_steps == 0:  # episode start
            title_string = title_string if self.n_steps != 0 else "*** START *** " + title_string
            ax.get_lines()[0].set_color("red")
        
        # text box displays info on current control and transform-limited regret
        knobs = self.psi_picoseconds.tolist()
        compressor_params = self.laser.compressor_params.tolist()
        control_info = 'GDD: {:2.2e}\n'.format(knobs[0], compressor_params[0])+\
                       'TOD: {:2.2e}\n'.format(knobs[1], compressor_params[1])+\
                       'FOD: {:2.2e}\n'.format(knobs[2], compressor_params[2])+\
                       'B-integral: {:.4f}'.format(self.laser.B)
        
        compressor_info = 'comp_GDD: {:2.2e}\n'.format(compressor_params[0])+\
                          'comp_TOD: {:2.2e}\n'.format(compressor_params[1])+\
                          'comp_FOD: {:2.2e}'.format(compressor_params[2])
        
        energy_info = 'FWHM (ps): {:2.2f}\n'.format(self._get_info()["current FWHM (ps)"])+\
                      'x: {:2.2f}'.format(100 * self._get_info()["current Peak Intensity (TW/m^2)"] / (self.TL_intensity * 1e-12))
        
        props = dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.5)
        ax.text(0.7, 0.95, control_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.text(0.025, 0.8, energy_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        ax.text(0.025, 0.7, compressor_info, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        ax.legend(loc="upper left", fontsize=12)
        ax.set_title(title_string, fontsize=12)

        # Create high-res image and resize to target dimensions
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(X)
        # Maintain aspect ratio while resizing
        target_height = 240
        aspect_ratio = img.size[0] / img.size[1]
        target_width = int(target_height * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        pulses_rgb_array = np.array(img.convert('RGB'))
        plt.close(fig)

        return pulses_rgb_array

    def _render_controls(self)->np.array:
        """
        Renders the evolution of control parameters in feasible space with respect to a size `n` deque.
        """
        fig, ax = visualize_controls(self.controls_buffer)
        # specializing the plots for showcasing trajectories
        title_string = f"Timestep {self.n_steps}/{self.MAX_STEPS}"
        if self.n_steps == 0:  # episode start
            title_string = title_string if self.n_steps != 0 else "*** START *** " + title_string
            ax.get_lines()[0].set_color("red")
        
        ax.set_title(title_string, fontsize=12)

        # creating and coloring the canvas
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(X)
        target_height = 240
        aspect_ratio = img.size[0] / img.size[1]
        target_width = int(target_height * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        controls_rgb_array = np.array(img.convert('RGB'))
        plt.close(fig)

        return controls_rgb_array
    
    def _render_frog(self)->np.array:
        """Renders FROG trace."""
        fig, ax = visualize_frog(self.frog.cpu().numpy(), window_size=self.window_size)

        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(X)
        target_height = 240
        aspect_ratio = img.size[0] / img.size[1]
        target_width = int(target_height * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        frog_rgb_array = np.array(img.convert('RGB'))
        plt.close(fig)

        return frog_rgb_array
    
    def _render_reward(self)->np.array:
        """Renders reward components."""
        fig, ax = visualize_reward(self.reward_components)
        
        fig.canvas.draw()
        X = np.array(fig.canvas.renderer.buffer_rgba())
        img = Image.fromarray(X)
        target_height = 240
        aspect_ratio = img.size[0] / img.size[1]
        target_width = int(target_height * aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        reward_rgb_array = np.array(img.convert('RGB'))
        plt.close(fig)
        return reward_rgb_array

    def _render_frame(self):
        """
        Renders one frame using Pygame or returns RGB arrays depending on render_mode.
        """
        # Get the visualization arrays - adjust transposition to match desired orientation
        pulse_rgb_array = np.transpose(self._render_pulse(), axes=(1, 0, 2))
        # controls_rgb_array = np.transpose(self._render_controls(), axes=(1, 0, 2))
        controls_rgb_array = np.transpose(self._render_reward(), axes=(1, 0, 2))
        frog_rgb_array = np.transpose(self._render_frog(), axes=(1, 0, 2))

        if self.render_mode == "rgb_array":
            # For rgb_array mode, combine the three arrays horizontally
            combined = np.concatenate([pulse_rgb_array, controls_rgb_array, frog_rgb_array], axis=0)
            # Final transpose to get the correct orientation
            return np.transpose(combined, axes=(1, 0, 2))

        elif self.render_mode == "human":
            screen_size = (960, 240)  # Three panels: each 320 x 240

            # Initialize Pygame window and clock if necessary
            if getattr(self, "window", None) is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(screen_size)

            if getattr(self, "clock", None) is None:
                self.clock = pygame.time.Clock()

            # Process events and check for the QUIT event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            # Create and scale surfaces for each panel
            panel_width = screen_size[0] // 3
            pulses_surf = pygame.transform.scale(
                pygame.surfarray.make_surface(pulse_rgb_array), 
                (panel_width, screen_size[1])
            )
            controls_surf = pygame.transform.scale(
                pygame.surfarray.make_surface(controls_rgb_array), 
                (panel_width, screen_size[1])
            )
            frog_surf = pygame.transform.scale(
                pygame.surfarray.make_surface(frog_rgb_array), 
                (panel_width, screen_size[1])
            )

            # Blit (draw) the surfaces onto the window
            self.window.blit(pulses_surf, (0, 0))
            self.window.blit(controls_surf, (panel_width, 0))
            self.window.blit(frog_surf, (2 * panel_width, 0))

            # Update the display
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

            # Return the same format as rgb_array mode for consistency
            # return np.transpose(pygame.surfarray.array3d(self.window), (1, 0, 2))
            return None

    def close(self):
        if getattr(self, "window", None) is not None:
            pygame.display.quit()
            pygame.quit()


    def render(self):
        """Calls the render frame method."""
        return self._render_frame()
