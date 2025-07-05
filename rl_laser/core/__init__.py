"""Core implementation of RL-Laser environments.

At the moment original source files still live in the legacy *laserenv/*
directory.  To avoid a giant code-move diff we simply re-export them here.
A future PR can physically relocate the files without changing user-facing
imports.
"""

from importlib import import_module as _imp
import sys as _sys

_origin_pkg = _imp("laserenv")

# expose all public names so that `from rl_laser.core import LaserEnv` works
for _name in dir(_origin_pkg):
    if not _name.startswith("__"):
        setattr(_sys.modules[__name__], _name, getattr(_origin_pkg, _name))

# register submodules under the new namespace
for _mod_name, _mod in list(_sys.modules.items()):
    if _mod_name.startswith("laserenv"):
        _sys.modules[_mod_name.replace("laserenv", __name__, 1)] = _mod

del _imp, _sys, _origin_pkg, _name, _mod_name, _mod