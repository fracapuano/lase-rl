# RL-Laser  
[![CI](https://github.com/yourname/rl-laser/actions/workflows/ci.yml/badge.svg)](https://github.com/yourname/rl-laser/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rl-laser)](https://pypi.org/project/rl-laser/)

Physics-informed **laser control** environments for _Reinforcement Learning_ built on
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

---

## Installation
```bash
pip install rl-laser            # latest release from PyPI
# or for development
pip install -e .[dev,visualize]
```

## Quick start
```python
import gymnasium as gym
import rl_laser

# Standard deterministic dynamics
env = gym.make("LaserEnv-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)
...  # RL loop

# Domain-randomised variant
dr_env = rl_laser.make("RandomLaserEnv-v0")
```

The observation is a dict containing a 2-D **FROG** trace and the current
(normalised) dispersion controls `psi`.

## Environments
| ID | Description |
|----|-------------|
| `LaserEnv-v0` | Deterministic L1-pump simulator |
| `RandomLaserEnv-v0` | Same dynamics with randomised physical parameters per episode |

## Release process  
The repository ships with **GitHub Actions** automation under
`.github/workflows/ci.yml`.

1. Push or open a PR – CI installs the package on Python 3.9 & 3.10 and runs a
   smoke-test import.
2. Create a GitHub **release** with a tag that starts with `v`, e.g. `v0.2.0`.  
   The `publish` job builds the wheel/sdist and uploads them to PyPI using the
   `PYPI_TOKEN` repository secret.
3. Announce the release, attach the Zenodo DOI if desired – your environment is
   now `pip install rl-laser`-able!

---

### Citation
If you use RL-Laser in academic work, please cite our Reinforcement Learning
Conference 2025 paper:
```bibtex
@inproceedings{capuano2025laser,
  title     = {Physics-informed Reinforcement Learning for Ultra-Fast Lasers},
  author    = {Francesco Capuano and others},
  booktitle = {Proceedings of the RLC '25},
  year      = {2025}
}
```
