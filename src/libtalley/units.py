import logging

import unyt

logger = logging.getLogger(__name__)


#===============================================================================
# Units, dimensions, and unit systems
#===============================================================================
def _safe_define(symbol, *args, **kwargs):
    # unyt occasionally adds new built-ins, and throws an error for already
    # defined symbols. Check for existence before defining.
    if not hasattr(unyt, symbol):
        unyt.define_unit(symbol, *args, **kwargs)
    else:
        logger.info(f'Unit {symbol!r} is already defined.')


# Acceleration
_safe_define('g0', unyt.standard_gravity, tex_repr=R'\rm{g_0}')

# Force
_safe_define('kip', (1000.0, 'lbf'))

# Mass
_safe_define('blob', (1.0, 'lbf * s**2 / inch'))
_safe_define('kblob', (1.0, 'kip * s**2 / inch'))
_safe_define('kslug', (1.0, 'kip * s**2 / ft'))

# Stress/pressure
_safe_define('ksi', (1000.0, 'psi'))
_safe_define('psf', (1.0, 'lbf / ft**2'))
_safe_define('ksf', (1000.0, 'psf'))

#---------------------------------------
# Dimensions
#---------------------------------------
unyt.dimensions.stress = unyt.dimensions.pressure
unyt.dimensions.moment = unyt.dimensions.energy

#---------------------------------------
# US customary system
#---------------------------------------
uscs_system = unyt.UnitSystem(
    'uscs',
    length_unit='inch',
    mass_unit='kblob',
    time_unit='s',
    registry=unyt.unit_registry.default_unit_registry,
)
uscs_system['force'] = 'kip'
uscs_system['stress'] = 'ksi'
uscs_system['moment'] = 'kip * inch'


#===============================================================================
# Utility functions
#===============================================================================
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
        Used to construct the returned unyt.unyt_array object.

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

    # Convert scalar unyt_arrays to unyt_quantity. Done through reshaping and
    # indexing to make sure we still have the unit registry. Is that necessary?
    # Not sure!
    if q.ndim == 0:
        q = q.reshape(1)[0]

    return q.to(default_units) if convert else q


def convert(value, units, registry=None):
    """Convert an input value to the given units, and return a bare quantity.

    If the input value doesn't have units, assumes the input is in the requested
    units already.

    Parameters
    ----------
    value : array_like
    units : str, unyt.Unit
    registry : unyt.UnitRegistry, optional

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> convert(30, 's')
    array(30.)
    >>> convert(30*ft, 'm')
    array(9.144)
    >>> convert(([24, 36, 48], 'inch'), 'furlong')
    array([0.0030303 , 0.00454545, 0.00606061])
    """
    return process_unit_input(value, units, convert=True, registry=registry).v


def get_unit_system(system: str) -> unyt.UnitSystem:
    """Retrieve the actual UnitSystem object from the unit systems registry.

    Parameters
    ----------
    system : str
        The name of the unit system to retrieve.
    """
    try:
        return unyt.unit_systems.unit_system_registry[str(system)]
    except KeyError as exc:
        raise ValueError(f'{system!r} is not a valid unit system') from exc
