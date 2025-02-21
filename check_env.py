from stable_baselines3.common.env_checker import check_env
from laserenv.env_utils import EnvParametrization
from laserenv.LaserEnv import FROGLaserEnv

if __name__ == "__main__":
    # Use default environment parameters from EnvParametrization
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()

    # Create the environment (without rendering)
    env = FROGLaserEnv(
        bounds=bounds,
        compressor_params=compressor_params,
        B_integral=B_integral, 
        device="mps"
    )

    check_env(env)
