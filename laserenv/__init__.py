from gymnasium.envs.registration import register
from laserenv.env_utils import EnvParametrization

default_dynamics = EnvParametrization().get_parametrization_dict()

register(
    id="LaserEnv",
    entry_point="laserenv.LaserEnv:FROGLaserEnv",
    max_episode_steps=20,
    kwargs=default_dynamics
)
