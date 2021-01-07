"""Functions that format strings into the appropriate LaTeX format."""


def diff(f, x, n=1, partial=False):
    """Derivative representation.
    
    Parameters
    ----------
    f:
        function to be differentiated.
    x:
        variable to diff w.r.t.
    n:
        number of times to diff. (default: 1)
    partial
        Whether this is a partial derivative or not.
    """
    if partial:
        differ = R'\partial'
    else:
        differ = Rf"\mathrm{{d}}"
    if n > 1:
        level = f"^{n}"
    else:
        level = ''

    diff = '\\frac{%(differ)s%(level)s{%(f)s}}{%(differ)s{%(x)s}%(level)s}' % {
        'differ': differ,
        'level': level,
        'f': f,
        'x': x,
    }

    return diff


def steel_shape_name(shape, frac='nicefrac'):
    R"""Return LaTeX code for nicely typesetting a steel section name.

    Assumes the "by" part of the section is represented by an 'X', and that
    compound fractions are separated by '-' (hyphen, not endash).

    Only tested on W and HSS names so far.

    Parameters
    ----------
    shape : str
        Name of a steel section.
    frac : {'frac', 'tfrac', 'sfrac', 'nicefrac'}, optional
        The fraction macro to use. (default: 'nicefrac')

    Example
    -------
    >>> name = 'HSS3-1/2X3-1/2X3/16'
    >>> steel_shape_name(name)
    'HSS3\\nicefrac{1}{2}\\(\\times\\)3\\nicefrac{1}{2}\\(\\times\\)\\nicefrac{3}{16}'
    """
    # Whether or not we need to be in math mode to use the specified macro.
    try:
        math_mode = {
            'frac': True,
            'tfrac': True,
            'sfrac': False,
            'nicefrac': False,
        }[frac]
    except KeyError as exc:
        raise ValueError(f'Unrecognized fraction macro {frac!r}') from exc

    def frac_to_nicefrac(f):
        """Return LaTeX code for a nicefrac from a fraction like '3/16'. Does
        not support compound fractions."""
        (numer, denom) = f.split('/')
        frac_code = R'\%s{%s}{%s}' % (frac, numer, denom)
        if math_mode:
            frac_code = R'\(' + frac_code + R'\)'
        return frac_code

    # Split the name into pieces, and deal with any fractions.
    shape_parts = shape.split('X')
    for [index, part] in enumerate(shape_parts):
        if '/' in part and '-' in part:
            # Got a compound fraction
            (integer, fraction) = part.split('-')
            newfraction = frac_to_nicefrac(fraction)
            shape_parts[index] = integer + newfraction
        elif '/' in part:
            # Got a plain fraction
            shape_parts[index] = frac_to_nicefrac(part)

    latex_code = R'\(\times\)'.join(shape_parts)
    return latex_code
