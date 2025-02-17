import torch
import gymnasium as gym
from laserenv.env_utils import instantiate_laser
from laserenv.utils import *


class AbstractBaseLaser(gym.Env):
    """
    Custom gymnasium env for L1 Laser Pump. 
    This class abstracts actions and observation space.
    """
    def __init__(self, 
                 bounds:torch.TensorType, 
                 compressor_params:torch.TensorType,
                 B_integral:float, 
                 render_mode:str=None, 
                 seed:int=None):
        """Init function. Here laser-oriented characteristics are defined.
        Args: 
            bounds (torch.tensor): GDD, TOD and FOD upper and lower bounds. Shape must be (3x2). Values must be
                                   expressed in SI units (i.e., s^2, s^3 and s^4).
            compressor_params (torch.tensor): $\alpha_{GDD}$, $\alpha_{TOD}$, $\alpha_{FOD}$ of laser compressor. If no 
                                              non-linear effects were to take place, one would have that optimal control
                                              parameters would be exactly equal to -compressor_params.
            B_integral (float): B_integral value. This parameter models non-linear phase accumulation. The larger, 
                                          the higher the non-linearity introduced in the model.
            render_mode (str, optional): Render mode. Defaults to None.
        """
        self._bounds = bounds
        
        """
        xi parameters, parametrizing the dynamics of the laser system
        """
        self._compressor_params = compressor_params
        self._B = B_integral
        
        # render mode
        self.render_mode = render_mode
        # instantiate the considered laser model
        self.laser = instantiate_laser(
            compressor_params=self._compressor_params, 
            B_integral=self._B
        )
        # abstracts observation and action space
        self.observation_space = None
        self.action_space = None
        # initial set of control parameters applied is None
        self._psi = None
        self._seed = seed

    def seed(self, seed:int):
        self._seed = seed
    
    @property
    def B(self)->float: 
        """Returns private value of B integral"""
        return self._B
    
    @property
    def compressor_params(self)->torch.TensorType: 
        """Returns compressor params for laser"""
        return self._compressor_params

    @B.setter
    def update_B(self, new_B:float)->None: 
        """Updates the value of B integral. When updating, also updates the laser changing the value
        of laser's B."""

        if not new_B > 0:
            raise ValueError(f"B integral must be > 0! Prompted {new_B}")
        # updates env
        self._B = new_B
        # updates the simulator
        self._laser.overwrite_B_integral(new_B)

    @compressor_params.setter
    def update_compressor(self, new_params:torch.TensorType)->None: 
        """Updates compressor parameters value with new set of values."""
        # updates env
        self._compressor_params = new_params
        # updates the simulator
        self._laser.overwrite_compressor_params(new_params)
    
    def _get_obs(self)->dict: 
        """Returns observation"""
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def _get_info(self)->dict:
        """Returns info dicionary. Info's should be drawned from current observation."""
        raise NotImplementedError("This method should be implemented by the subclass.")

    def reset(self)->None: 
        """Resets to initial conditions."""
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def step(self, action:torch.TensorType):
        """Updates the observation based on action."""
        raise NotImplementedError("This method should be implemented by the subclass.")
    
    def render(self):
        """Renders current state."""
        raise NotImplementedError("This method should be implemented by the subclass.")

