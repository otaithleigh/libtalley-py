"""My collection of useful functions and doodads."""

import importlib.resources

from .color import Color  # noqa: F401
from .utils import *  # noqa: F403

__version__ = importlib.resources.read_text(__name__, '__version__')
