"""Functions that format strings into the appropriate LaTeX format."""


def diff(f, x, n=1):
    """Derivative representation.
    
    Parameters
    ----------
    f:
        function to be differentiated.
    x:
        variable to diff w.r.t.
    n:
        number of times to diff. (default: 1)
    """
    differ = Rf"\mathrm{{d}}"
    if n > 1:
        level = f"^{n}"
    else:
        level = ''

    return Rf"\frac{{{differ}{level}{{{f}}}}}{{{differ}{{{x}}}{level}}}"
