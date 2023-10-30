"""Matplotlib utilities."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm

__all__ = [
    'calc_grid',
    'GridParameters',
    'multicolor_line',
    'set_limits_to_major_ticks',
]


def set_limits_to_major_ticks(
    ax: Axes = None,
    which: t.Literal['x', 'y', 'both'] = 'both',
    inclusive: bool = False,
):
    """Set axis limits to the closest major ticks that bound the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes to set limits on. (default: plt.gca())
    which : {'x', 'y', 'both'}, optional
        Which axis to set limits on. (default: 'both')
    inclusive : bool, optional
        If True, data interval that must be bounded is a closed range instead of
        an open one. (default: False)
    """
    if ax is None:
        ax = plt.gca()

    def set_axis_limits(axis, set_lim):
        loc = axis.get_major_locator()
        ticks = loc()
        lower, upper = axis.get_data_interval()
        if inclusive:
            lower_lim = ticks[ticks <= lower].max()
            upper_lim = ticks[ticks >= upper].min()
        else:
            lower_lim = ticks[ticks < lower].max()
            upper_lim = ticks[ticks > upper].min()
        set_lim(lower_lim, upper_lim)

    if which in ('x', 'both'):
        set_axis_limits(ax.xaxis, ax.set_xlim)
    if which in ('y', 'both'):
        set_axis_limits(ax.yaxis, ax.set_ylim)


class GridParameters(t.NamedTuple):
    nrows: int
    ncols: int
    figsize: tuple[float, float]


def calc_grid(nplots: int, plot_width: float, plot_height: float, avail_width: float):
    """Calculate grid parameters for a given number of figures and desired
    subplot size.

    Generates parameters to "line-wrap" the plots to a given width.

    Parameters
    ----------
    nplots : int
        Total number of subplots (axes).
    plot_width : float
        Desired width of individual subplot, in inches.
    plot_height : float
        Desired height of individual subplot, in inches.
    avail_width : float
        Available width for plots.

    Returns
    -------
    nrows: int, ncols: int, figsize: tuple[float, float]
    """
    max_ncols = int(avail_width // plot_width)
    if nplots <= max_ncols:
        nrows = 1
        ncols = nplots
    else:
        nrows = nplots // max_ncols
        if nplots % max_ncols:
            nrows += 1
        ncols = max_ncols

    return GridParameters(nrows, ncols, (ncols * plot_width, nrows * plot_height))


def multicolor_line(
    x,
    y,
    c=None,
    vmin=None,
    vmax=None,
    levels=None,
    cmap=None,
    ax: t.Optional[Axes] = None,
    **lc_kwargs,
):
    """Plot a line as multiple segments, so that it varies in color from start
    to finish.

    Parameters
    ----------
    x, y : array-like
        1-d arrays of coordinates.
    c : array-like, optional
        1-d array that will map to colors. If not provided, defaults to a range
        from 0 to len(x).
    vmin : float, optional
        Lower bound for colormap normalization. Defaults to c.min().
    vmax : float, optional
        Upper bound for colormap normalization. Defaults to c.max().
    levels : array-like, int, optional
        If not provided, vary colormap continuously. If an integer, use that
        many discrete linearly-spaced boundaries. If an array-like, use those
        values as the boundaries.
    cmap : str, Colormap, optional
        Colormap to use. Defaults to rc values ('image.cmap').
    ax : Axes, optional
        The axes to plot on. Defaults to plt.gca().
    **kwargs
        Additional arguments to the LineCollection constructor.

    Returns
    -------
    LineCollection
        The generated line collection.

    Notes
    -----
    - Matplotlib does not do joining between segments of a LineCollection (the
      ends are not guaranteed to be at the same point, after all), so lines may
      appear broken or jagged.
    - Each segment can only have one color. This may affect clarity for lines
      that have segments of non-uniform length.
    """
    # --------------------------------------------
    # Get axes
    # --------------------------------------------
    if ax is None:
        ax = plt.gca()

    # --------------------------------------------
    # Validate inputs
    # --------------------------------------------
    x: np.ndarray = np.asarray(x)
    y: np.ndarray = np.asarray(y)
    if c is None:
        c = np.arange(len(x))
    else:
        c = np.asarray(c)

    if x.ndim != 1 or y.ndim != 1 or c.ndim != 1:
        raise ValueError('x, y, and c must all be 1-d')

    # --------------------------------------------
    # Create segments for LineCollection
    # --------------------------------------------
    points = np.empty((c.size, 1, 2))
    points[:, 0, 0] = x
    points[:, 0, 1] = y
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # --------------------------------------------
    # Create normalizer for colormap
    # --------------------------------------------
    vmin = c.min() if vmin is None else vmin
    vmax = c.max() if vmax is None else vmax
    if levels is None:
        # Continuous norm
        norm = plt.Normalize(vmin, vmax)
    else:
        # Discrete norm
        if isinstance(levels, int):
            boundaries = np.linspace(vmin, vmax, levels)
        else:
            boundaries = np.asarray(levels)
        ncolors = plt.get_cmap(cmap).N
        norm = BoundaryNorm(boundaries, ncolors)

    # --------------------------------------------
    # Create and add LineCollection object
    # --------------------------------------------
    lc = LineCollection(segments, norm=norm, cmap=cmap, capstyle='round', **lc_kwargs)
    lc.set_array(c)
    ax.add_collection(lc)

    # LineCollection doesn't trigger autoscale, apparently? Ugh.
    if ax.get_autoscale_on():
        ax.autoscale()

    return lc
