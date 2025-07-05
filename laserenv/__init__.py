"""Deprecated shim for backward compatibility. Use `rl_laser.core` instead."""
from importlib import import_module as _imp
import sys as _sys

_sys.modules[__name__] = _imp("rl_laser.core")