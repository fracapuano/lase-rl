import os
import json
import random
import string
import numpy as np
import torch
from datetime import datetime
import socket
import pickle
from dr.ReturnDynamicsTrackerWrapper import ReturnDynamicsTrackerWrapper
from dr.ReturnTrackerWrapper import ReturnTrackerWrapper

from gymnasium.wrappers import FrameStackObservation


def save_object(obj, save_dir, filename):
    with open(os.path.join(save_dir, f'{filename}.pkl'), 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def create_dir(path):
	try:
		os.mkdir(os.path.join(path))
	except OSError as error:
		# print('Dir already exists')
		pass

def make_wrapped_environment(env, args, wrapper=None):
    """Wrap env
        
        :param args.stack_history: int
                             number of previous obs and actions 
        :param args.rand_only: List[int]
                               dyn param indices mask
        :param args.dyn_in_obs: bool
                                condition the policy on the true dyn params
    """
    if args.stack_history > 1:
        env = FrameStackObservation(env, stack_size=args.stack_history)
        
    if wrapper is not None:
        if wrapper == 'doraemon':
            env = ReturnDynamicsTrackerWrapper(env)
        elif wrapper == 'returnTracker':
            env = ReturnTrackerWrapper(env)

    return env

def get_actor_critic_obs_keys(args):
    """Dummy method to define the keys to keep in the observation when using asymmetric feature extraction."""
    actor_keys = ["frog_trace", "psi", "action"]
    critic_keys = ["frog_trace", "psi", "action", "B_integral", "compressor_GDD"]
    
    return actor_keys, critic_keys

def get_random_string(n: int=5) -> str:
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))

def set_seed(seed):
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

def get_run_name(args):
    current_date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    return (
        f"{current_date}_"
        f"{args.env}_"
        f"{args.algo}_"
        f"t{args.timesteps}_"
        f"seed{args.seed}_"
        f"{socket.gethostname()}"
    )


def create_dirs(path):
	try:
		os.makedirs(os.path.join(path))
	except OSError as error:
		pass

def save_config(config, path):
    with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as file:
        # pprint(vars(config), stream=file)
        json.dump(config, file)
    return

def load_config(config_path):
    """Load configuration from yaml file"""
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def type_convert(value):
    """Convert string value to appropriate type"""
    # Try converting to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try converting to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Try converting to bool
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Return as string if no other type matches
    return value
