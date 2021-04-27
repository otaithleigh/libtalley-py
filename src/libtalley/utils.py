import collections
import functools
import os
import sys
import typing as t

import numpy as np
import pandas as pd
from tabulate import tabulate

__all__ = [
    'all_same_sign',
    'filename_noext',
    'is_even',
    'print_table',
    'recursive_update',
    'revcumsum',
    'round_signif',
]


#===============================================================================
# Math
#===============================================================================
def is_even(val):
    """Return ``True`` if even, ``False`` if odd."""
    return not val % 2


def revcumsum(a: np.ndarray, axis=None, dtype=None, out=None) -> np.ndarray:
    """Reverse cumulative summation.

    See np.cumsum for full docs.
    """
    return np.cumsum(a[::-1], axis, dtype, out)[::-1]


def all_same_sign(iterable: t.Iterable) -> bool:
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


#---------------------------------------
# Round to significiant figures
#---------------------------------------
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
    a_positive = np.where(np.isfinite(a) & (a != 0), np.abs(a), 10**(p - 1))
    mags = 10**(p - 1 - np.floor(np.log10(a_positive)))
    return np.round(a*mags)/mags


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


#===============================================================================
# Miscellany
#===============================================================================
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

    Source: https://stackoverflow.com/questions/3232943#3233356
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
