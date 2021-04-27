"""My collection of useful functions and doodads."""

import importlib.resources

from .color import Color
from .utils import *

__version__ = importlib.resources.read_text(__name__, '__version__')
