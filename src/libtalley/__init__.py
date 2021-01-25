"""My collection of useful functions and doodads."""

from __future__ import annotations

import collections
import functools
import os
import sys
import typing

import numpy as np
import pandas as pd
from tabulate import tabulate


def is_even(val):
    """Return ``True`` if even, ``False`` if odd."""
    return not val % 2


def revcumsum(a: np.ndarray, axis=None, dtype=None, out=None) -> np.ndarray:
    """Reverse cumulative summation.

    See np.cumsum for full docs.
    """
    return np.cumsum(a[::-1], axis, dtype, out)[::-1]


def filename_noext(path: str) -> str:
    """Get the name of a file without the extension.

    Parameters
    ----------
    path
        string containing the path to a file.

    Note that this only removes the final extension:
    >>> filename_noext('path/to/file.ext')
    'file'
    >>> filename_noext('path/to/archive.tar.gz')
    'archive.tar'
    """
    return os.path.splitext(os.path.basename(path))[0]


def all_same_sign(iterable: typing.Iterable) -> bool:
    """Check if all the elements in the iterable have the same sign.

    Returns true or false.

    Parameters
    ----------
    iterable
        An iterable item with numeric values.
    """
    v = (all(item >= 0 for item in iterable)
         or all(item < 0 for item in iterable))
    return v


@functools.singledispatch
def round_signif(a, p):
    """Round numeric array a to significant figures p.

    Parameters
    ----------
    a : array_like
        Array to round.
    p : int
        Number of significant figures to round to.

    Source: https://stackoverflow.com/a/59888924, with modifications
    """
    a = np.asanyarray(a)
    a_positive = np.where(np.isfinite(a) & (a != 0), np.abs(a), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(a_positive)))
    return np.round(a * mags) / mags


@round_signif.register
def _(df: pd.DataFrame, p):
    return pd.DataFrame(
        round_signif(df.to_numpy(), p),
        columns=df.columns,
        index=df.index,
    )


@round_signif.register
def _(s: pd.Series, p):
    return pd.Series(
        round_signif(s.to_numpy(), p),
        index=s.index,
    )


def print_table(headers,
                data,
                datafmt='cols',
                tablefmt='pipe',
                stream=sys.stdout,
                **kwargs) -> str:
    """Print a neatly-formatted table.

    Parameters
    ----------
    headers:
        Headers of the table.

    data:
        List of lists to print.

    datafmt = 'cols':
        Order of data in ``data``. If 'cols' (default), each list in ``data`` is
        a column. If 'rows', each list in ``data`` is a row.

    tablefmt = 'pipe':
        Format descriptor. See ``tabulate.tabulate_formats`` for a list.

    stream = sys.stdout:
        Stream to print to. If stream==None, don't print.

    kwargs:
        Additional arguments to ``tabulate.tabulate``.
    """
    # Check input
    if datafmt.lower() == 'cols':
        if len(headers) != len(data):
            raise ValueError("Number of headers must equal number of columns.")

        lengths_of_cols = [len(col) for col in data]
        for length in lengths_of_cols:
            if lengths_of_cols[0] != length:
                raise ValueError("Columns must be of equal length.")

        tabular_data = np.array(data).transpose().tolist()
    elif datafmt.lower() == 'rows':
        tabular_data = data
    else:
        raise ValueError(f"Unrecognized data format: {datafmt}")

    tabulated = tabulate(tabular_data,
                         headers=headers,
                         tablefmt=tablefmt,
                         **kwargs)
    print(tabulated, file=stream)

    return tabulated


def recursive_update(d: dict, u: dict) -> dict:
    """Recursively update a nested dictionary.

    Parameters
    ----------
    d:
        ``dict`` to update.
    u:
        ``dict`` to use for updating.

    Example
    -------
    >>> d = {'a': {1: 2}}
    >>> u = {'a': {2: 1}}
    >>> recursive_update(d, u)
    {'a': {1: 2, 2: 1}}

    Compare:
    >>> d.update(u)
    >>> print(d)
    {'a': {2: 1}}

    Source: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth#3233356
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Color:
    """RGB-based color representation."""
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
        # yapf: disable
        isrgb   = lambda tup: len(tup) == 3 and all([v >= 0 and v <= 255 for v in tup])
        isrgb1  = lambda tup: len(tup) == 3 and all([v >= 0 and v <=   1 for v in tup])
        iscmyk  = lambda tup: len(tup) == 4 and all([v >= 0 and v <= 100 for v in tup])
        iscmyk1 = lambda tup: len(tup) == 4 and all([v >= 0 and v <=   1 for v in tup])
        # yapf: enable

        if cmyk is not None and not iscmyk(cmyk):
            raise ValueError(f"Invalid CMYK tuple: {cmyk}")

        if rgb1:
            if not isrgb1(rgb):
                raise ValueError(f"Invalid RGB1 tuple: {rgb}")
            rgb = [int(v*255) for v in rgb]
        else:
            if not isrgb(rgb):
                raise ValueError(f"Invalid RGB tuple: {rgb}")
            rgb = [int(v) for v in rgb]

        self.RGB = tuple(rgb)
        self.CMYK = tuple(cmyk) if cmyk is not None else None

    @property
    def HEX(self) -> str:
        """Hex code representation of the color's RGB value.

        Example
        -------
        >>> smokey = Color(rgb=(88, 89, 91))
        >>> smokey.HEX
        '#58595b'
        """
        hexcodes = [f'{v:02x}' for v in self.RGB]
        return '#' + ''.join(hexcodes)

    @property
    def RGB1(self) -> typing.Tuple[float, float, float]:
        """RGB value of the color, normalized to 1."""
        return tuple(v/255 for v in self.RGB)

    def __repr__(self):
        return f"Color(rgb={self.RGB}, cmyk={self.CMYK})"

    @classmethod
    def fromTennesseePalette(cls, color_name) -> cls:
        """Select a color from the University of Tennessee Knoxville brand guide.

        Color reference: https://brand.utk.edu/colors/palettes/

        Based on Dr. Mark Denavit's MATLAB function ``TennesseeColorPalette``.
        See https://github.com/denavit/MATLAB-Library for more details.
        """
        try:
            color = cls._TENNESSEE_COLORS[color_name.lower()]
        except KeyError:
            raise ValueError(f"unrecognized color name {color_name!r}")

        return cls(rgb=color['rgb'], cmyk=color['cmyk'])

    # yapf: disable
    _TENNESSEE_COLORS = {
        'orange':      { 'rgb': (255, 130,   0), 'cmyk': (  0,  50, 100,   0) },
        'white':       { 'rgb': (255, 255, 255), 'cmyk': (  0,   0,   0,   0) },
        'smokey':      { 'rgb': ( 88,  89,  91), 'cmyk': (  0,   0,   0,  80) },
        'valley':      { 'rgb': (  0, 116, 111), 'cmyk': (100,  50,  65,   0) },
        'torch':       { 'rgb': (230,  89,  51), 'cmyk': (  0,  85, 100,   0) },
        'globe':       { 'rgb': (  0, 108, 147), 'cmyk': (100,  18,  10,  50) },
        'limestone':   { 'rgb': (240, 237, 227), 'cmyk': (  5,   5,  10,   0) },
        'river':       { 'rgb': ( 81, 124, 150), 'cmyk': ( 70,  40,  25,  10) },
        'leconte':     { 'rgb': (141,  32,  72), 'cmyk': ( 40, 100,  60,  30) },
        'regalia':     { 'rgb': (117,  74, 126), 'cmyk': ( 55, 100,  25,  25) },
        'sunsphere':   { 'rgb': (254, 213,  53), 'cmyk': (  0,  20,  90,   0) },
        'rock':        { 'rgb': (167, 169, 172), 'cmyk': (  0,   0,   0,  40) },
        'legacy':      { 'rgb': ( 87, 149, 132), 'cmyk': ( 65,  20,  50,  10) },
        'summitt':     { 'rgb': (185, 225, 226), 'cmyk': ( 25,   0,  10,   0) },
        'buckskin':    { 'rgb': (112,  85,  80), 'cmyk': ( 60,  70,  70,  15) },
        'energy':      { 'rgb': (238,  62, 128), 'cmyk': (  0,  90,  20,   0) },
        'switchgrass': { 'rgb': (171, 193, 120), 'cmyk': ( 25,   0,  80,  10) },
        'fountain':    { 'rgb': ( 33, 151, 169), 'cmyk': ( 75,  15,  25,  10) },
        'eureka':      { 'rgb': (235, 234, 100), 'cmyk': ( 10,   0,  75,   0) },
        'eureka!':     { 'rgb': (235, 234, 100), 'cmyk': ( 10,   0,  75,   0) },
    }
    # yapf: enable


# In case some of my code still uses the module level variable
_TENNESSEE_COLORS = Color._TENNESSEE_COLORS
