"""Compatibility shim – original code lives in top-level *laserenv* folder.
This package simply re-exports those modules so that new imports like
``from rl_laser.laserenv import LaserEnv`` work without physically moving files.
A proper physical move can be performed later without breaking user code."""

import importlib, sys, types

# import the existing top-level package
_origin = importlib.import_module("laserenv")

# expose its attributes at this package level
for _name in dir(_origin):
    if not _name.startswith("__"):
        setattr(sys.modules[__name__], _name, getattr(_origin, _name))

# ensure submodules are available under the new namespace as well
for _mod_name, _mod in sys.modules.items():
    if _mod_name.startswith("laserenv."):
        new_name = _mod_name.replace("laserenv", __name__, 1)
        sys.modules[new_name] = _mod

del importlib, sys, types, _origin, _name, _mod_name, _mod, new_name