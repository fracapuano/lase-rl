"""Utility to lazily import optional heavy dependencies with descriptive error.
Usage:
    np = opt("numpy", "base")
    torch = opt("torch", "torch")
The second argument must match an extras key in setup.cfg.
When running under type-checkers (mypy, pyright) the module is stubbed so that
names are still understood.
"""
from typing import TYPE_CHECKING, Any
import importlib, types

def opt(module: str, extra: str) -> Any:
    """Attempt to import *module*; if missing raise helpful runtime error.
    During static type checking, return a dummy module instead so tools do not
    complain about missing imports.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover
        if TYPE_CHECKING:
            return types.ModuleType(module)
        raise ModuleNotFoundError(
            f"RL-Laser requires the optional dependency '{module}'. "
            f"Install it via  pip install rl-laser[{extra}]"
        ) from exc