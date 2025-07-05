"""RL-Laser: Open-source physics-informed laser Gymnasium environments.

After `pip install rl-laser`, simply::

    import gymnasium as gym
    import rl_laser
    env = gym.make("LaserEnv-v0", render_mode="rgb_array")

This initializer keeps registration lightweight by importing the heavy
numerical stack only on demand.
"""
from importlib import import_module
# type: ignore
from gymnasium.envs.registration import register, registry

# Lazy import of core implementation to avoid importing torch/numpy when only
# discovering entry-points (e.g. during package metadata inspection).


def _lazy_env_import():
    """Import the actual environment modules exactly once and register them."""
    # Only execute at first call – subsequent calls are no-ops
    if "LaserEnv-v0" in registry and "RandomLaserEnv-v0" in registry:
        return
    # Import implementation package (pulls in torch, numpy, etc.)
    import_module("laserenv")

    # After import, laserenv.__init__ has already registered base ids (if any).
    # Re-register with explicit versioned names for clarity.
    # We purposefully allow duplicate entry-points as Gymnasium will overwrite
    # with the same callable.
    register(
        id="LaserEnv-v0",
        entry_point="laserenv.LaserEnv:FROGLaserEnv",
        max_episode_steps=20,
    )
    register(
        id="RandomLaserEnv-v0",
        entry_point="laserenv.RandomLaserEnv:RandomFROGLaserEnv",
        max_episode_steps=20,
    )


# Execute import at module import time so that `gym.make` works out-of-the-box.
_lazy_env_import()

def make(id: str = "LaserEnv-v0", **kwargs):
    """Convenience wrapper around ``gymnasium.make``.

    Example
    -------
    >>> import rl_laser as rll
    >>> env = rll.make(render_mode="rgb_array")
    """
    import gymnasium as gym  # type: ignore

    return gym.make(id, **kwargs)

__all__ = ["make"]