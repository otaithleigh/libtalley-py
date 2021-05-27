"""Tools for IPython."""

import numbers
import os
import tempfile

import numpy as np
from IPython.display import Math, SVG, display

from .utils import round_signif

__all__ = [
    'show_fig',
    'show_matrix',
    'show_var',
]


def show_var(name, value, units=None, *, boxed=False, fmt='#.3g'):
    """Show the value of a variable using LaTeX.

    Parameters
    ----------
    name : str
        The name (in LaTeX format) of the variable.
    value
        The value of the variable to show.
    units : str, optional
        The units of the variable. (default: None)

    Keyword-only parameters
    -----------------------
    boxed : bool, optional
        If True, draw a box around the entire expression. (default: False)
    fmt : str, optional
        Format string used to display `value`. (default: '#.3g')
    """
    if units is None:
        units = ''
    else:
        units = r'~\mathrm{%s}' % units

    eqn = f'{name} = {value:{fmt}}{units}'
    if boxed:
        eqn = r'\boxed{%s}' % eqn

    display(Math(eqn))


def show_matrix(name,
                array,
                units=None,
                *,
                boxed=False,
                style='b',
                vector_style='column',
                nsd=3,
                suppress=True):
    """Display a NumPy array as a vector/matrix.

    Parameters
    ----------
    name : str
        The name (in LaTeX format) of the matrix.
    array : array_like
        The vector or matrix to show.
    units : str, optional
        The units of the vector or matrix. (default: None)

    Keyword-only parameters
    -----------------------
    boxed : bool, optional
        If True, draw a box around the entire expression.
    style : {'', 'b', 'B', 'p', 'v', 'V'}, optional
        The matrix border style. (default: 'b')

        ''  - no brackets
        'b' - square brackets [ ]
        'B' - curly brackets { }
        'p' - round brackets ( )
        'v' - single vertical line | |
        'V' - double vertical line || ||
    vector_style : {'row', 'column'}, optional
        How to represent one-dimensional arrays. (default: 'column')
    nsd : int, optional
        Number of significant digits to show. (default: 3)
    suppress : bool, optional
        Whether to suppress very small floating point numbers. (default: True)
    """
    array: np.ndarray = np.asarray(array)
    if array.ndim > 2:
        raise ValueError('Array must have 2 dimensions or fewer')

    matrix_styles = {'', 'b', 'B', 'p', 'v', 'V'}
    if style not in matrix_styles:
        raise ValueError(f'style {style!r} must be one of {matrix_styles}')

    # 0D and 1D -> row or column vector
    if array.ndim < 2:
        vector_styles = {
            'column': (-1, 1),
            'row': (1, -1),
        }
        try:
            new_shape = vector_styles[vector_style]
        except KeyError:
            raise ValueError(f'vector_style {vector_style!r} must be one '
                             f'of {set(vector_styles.keys())}')
        array = array.reshape(new_shape)

    # Rounding
    if issubclass(array.dtype.type, numbers.Number):
        if nsd is not None:
            array = round_signif(array, nsd)
        if suppress:
            array = array.round(10)

    rows = [' & '.join(row) for row in array.astype('U')]
    matrix = ''.join([
        r'\begin{%smatrix}' % style,
        r' \\ '.join(rows),
        r'\end{%smatrix}' % style,
    ])

    show_var(name, matrix, units=units, boxed=boxed, fmt='')


def show_fig(fig, *args, **kwargs):
    """Display a Matplotlib figure as an SVG object, and embed it in the
    notebook.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to show.
    *args, **kwargs
        Additional arguments passed to ``Figure.savefig``.
    """
    with tempfile.NamedTemporaryFile(delete=False) as file:
        fig.savefig(file, *args, format='svg', **kwargs)
        file.close()
        path = os.path.relpath(file.name)
        display(SVG(path))
    try:
        os.unlink(file.name)
    except FileNotFoundError:
        pass
