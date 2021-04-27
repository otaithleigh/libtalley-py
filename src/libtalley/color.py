from __future__ import annotations

import importlib.resources
import json
from typing import Dict, List, Tuple

_ColorDatabase = Dict[str, Dict[str, List[int]]]

__all__ = [
    'isrgb',
    'isrgb1',
    'iscmyk',
    'Color',
]


def isrgb(t: tuple):
    """Return True if t is a valid RGB tuple."""
    return len(t) == 3 and all(v >= 0 and v <= 255 for v in t)


def isrgb1(t: tuple):
    """Return True if t is a valid RGB1 tuple."""
    return len(t) == 3 and all(v >= 0 and v <= 1 for v in t)


def iscmyk(t: tuple):
    """Return True if t is a valid CMYK tuple."""
    return len(t) == 4 and all(v >= 0 and v <= 100 for v in t)


class Color:
    """RGB- and CMYK-based color representation.
    
    Attributes
    ----------
    cmyk : Tuple[int, int, int, int], optional
        The color in CMYK (0--100) format.
    rgb : Tuple[int, int, int]
        The color in RGB integer (0--255) format.
    hex : str
        The color in hex code ('#RRGGBB') format.
    rgb1 : Tuple[float, float, float]
        The color in RGB1 floating point (0.0--1.0) format.
    """
    def __init__(self, rgb, cmyk=None, rgb1=False):
        """Create a new Color.

        Parameters
        ----------
        rgb : tuple
            RGB tuple representing the color, with integer values ranging from 0
            to 255. Values are cast to ``int`` during construction: rounding may
            occur.

        cmyk : tuple, optional
            CMYK tuple representing the color, with values ranging from 0 to
            100. (default: None)

        rgb1 : bool, optional
            If ``True``, the argument to ``rgb`` has floating point values
            ranging from 0 to 1. This is still stored as a tuple of ``int``s
            from 0 to 255: rounding will occur. (default: False)
        """
        if cmyk is not None:
            if not iscmyk(cmyk):
                raise ValueError(f"Invalid CMYK tuple: {cmyk!r}")
            self._cmyk = tuple(cmyk)
        else:
            self._cmyk = None

        if rgb1:
            if not isrgb1(rgb):
                raise ValueError(f"Invalid RGB1 tuple: {rgb!r}")
            self._rgb = tuple(int(v*255) for v in rgb)
            self._rgb1 = tuple(float(v) for v in rgb)
        else:
            if not isrgb(rgb):
                raise ValueError(f"Invalid RGB tuple: {rgb!r}")
            self._rgb = tuple(int(v) for v in rgb)
            self._rgb1 = tuple(v/255 for v in rgb)

        self._hex = '#' + ''.join(f'{v:02x}' for v in self._rgb)

    @property
    def hex(self) -> str:
        """Hex code representation of the color's RGB value.

        Example
        -------
        >>> smokey = Color(rgb=(88, 89, 91))
        >>> smokey.hex
        '#58595b'
        """
        return self._hex

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """RGB value of the color, as integers from 0 to 255."""
        return self._rgb

    @property
    def rgb1(self) -> Tuple[float, float, float]:
        """RGB value of the color, as floats normalized to 1."""
        return self._rgb1

    @property
    def cmyk(self):
        """CMYK value of the color, as integers from 0 to 100."""
        return self._cmyk

    def __repr__(self):
        return f"Color(rgb={self.rgb}, cmyk={self.cmyk})"

    @classmethod
    def from_name(cls, name: str):
        color_database = cls._get_color_database()
        try:
            color = color_database[name.casefold()]
        except KeyError as exc:
            raise ValueError(f'Unknown color {name!r}') from exc

        return cls(rgb=color['rgb'], cmyk=color.get('cmyk'))

    @classmethod
    def _load_color_database(cls):
        with importlib.resources.open_text('libtalley', 'colors.json') as f:
            cls._color_database = json.load(f)

    @classmethod
    def _get_color_database(cls) -> _ColorDatabase:
        try:
            return cls._color_database
        except AttributeError:
            cls._load_color_database()
            return cls._color_database
