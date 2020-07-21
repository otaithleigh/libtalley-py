import unyt

unyt.define_unit('kip', 1000*unyt.lbf, tex_repr='\\rm{kips}')
unyt.define_unit('kipf', 1000*unyt.lbf, tex_repr='\\rm{kips}')
unyt.define_unit('ksi', (1000, 'lbf/inch**2'))

def process_unit_input(in_, default_units=None, convert=False,
                       registry=None) -> unyt.unyt_array:
    """Process an input value that may or may not have units.

    Accepts the following input styles::

        in_ = 1000           ->  out_ = 1000*default_units
        in_ = (1000, 'psi')  ->  out_ = 1000*psi
        in_ = 1000*psi       ->  out_ = 1000*psi

    If `convert` is True, then values that come in with units are converted to
    `default_units` when returned::

        in_ = 1000           ->  out_ = 1000*default_units
        in_ = (1000, 'psi')  ->  out_ = (1000*psi).to(default_units)
        in_ = 1000*psi       ->  out_ = (1000*psi).to(default_units)

    Parameters
    ----------
    in_
        Input values.
    default_units : str, unyt.Unit, optional
        Default units to use if inputs don't have units associated already.
    convert : bool, optional
        Convert all inputs to `default_units` (default: False)
    registry : unyt.UnitRegistry, optional
        Necessary if the desired units are not in the default unit registry.
        Used to construct the returned unyt.unit_array object.

    Returns
    -------
    q : unyt.unyt_array
    """
    if isinstance(in_, unyt.unyt_array):
        q = in_
    elif isinstance(in_, tuple):
        if len(in_) == 2:
            value, units = in_
            q = unyt.unyt_array(value, units, registry=registry)
        else:
            raise ValueError('Input tuple must be length 2; '
                             f'given had length {len(in_)}')
    else:
        q = unyt.unyt_array(in_, default_units, registry=registry)

    return q.to(default_units) if convert else q
