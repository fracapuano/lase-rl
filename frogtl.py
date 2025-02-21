from laserenv.utils.frogtrace import compute_frog_trace
import matplotlib.pyplot as plt
from laserenv.env_utils import EnvParametrization
from laserenv.LaserEnv import FROGLaserEnv

# Use default environment parameters from EnvParametrization
params = EnvParametrization()
compressor_params, bounds, B_integral = params.get_parametrization()

env = FROGLaserEnv(
    bounds=bounds,
    compressor_params=compressor_params,
    B_integral=B_integral
)

frog_trace = compute_frog_trace(
    env.transform_limited[0],
    env.laser.frequency,
    trim_window=1000,
    pad_width=10000
)

fig, ax = plt.subplots()
ax.imshow(frog_trace)
plt.show()

