import sys, types  # added for stub
# --- Line profiler stub to avoid runtime overhead when not profiling ---
if "line_profiler" not in sys.modules:
    line_profiler = types.ModuleType("line_profiler")
    # A no-op decorator that simply returns the original function untouched
    setattr(line_profiler, "profile", lambda func=None, *args, **kwargs: func if callable(func) else (lambda x: x))
    sys.modules["line_profiler"] = line_profiler
else:
    import line_profiler as _lp
    if not hasattr(_lp, "profile"):
        setattr(_lp, "profile", lambda func=None, *args, **kwargs: func if callable(func) else (lambda x: x))
# ----------------------------------------------------------------------
from gymnasium.envs.registration import register
from laserenv.env_utils import EnvParametrization

default_dynamics = EnvParametrization().get_parametrization_dict()

register(
    id="LaserEnv",
    entry_point="laserenv.LaserEnv:FROGLaserEnv",
    max_episode_steps=20,
    kwargs=default_dynamics
)

register(
    id="RandomLaserEnv",
    entry_point="laserenv.RandomLaserEnv:RandomFROGLaserEnv",
    max_episode_steps=20,
    kwargs=default_dynamics
)