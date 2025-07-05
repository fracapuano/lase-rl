import line_profiler  # type: ignore

from typing import Tuple, List, Sequence, Union, Optional, Dict, Any, TYPE_CHECKING

from rl_laser.core._optional import opt

np = opt("numpy", "base")
torch = opt("torch", "torch")

from rl_laser.core.utils import physics  # type: ignore

class LaserEnv:
    def __init__(
        self,
        # setting the simulator in empty state
        self.reset()

        # runtime type for mypy / IDEs
        from rl_laser.core.laser_numpy import ComputationalLaserNumpy  # type: ignore
        self.laser: ComputationalLaserNumpy  # type: ignore

        # caches
        self._cached_psi = None  # cache for last psi that generated expensive computations
        self._cached_pulse = None
        self._cached_frog = None

    def get_reward(self) -> Tuple[float, Dict[str, float]]:
        pass

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass