import logging
import typing as t
from functools import singledispatchmethod

import unyt
from unyt import unyt_array
from unyt.exceptions import UnitConversionError

try:
    import xarray as xr
except ImportError:
    xr = None

logger = logging.getLogger(__name__)

#===============================================================================
# Typing
#===============================================================================
UnitLike = t.Union[str, unyt.Unit]
SystemLike = t.Union[str, unyt.UnitSystem]


#===============================================================================
# Units and dimensions
#===============================================================================
def _safe_define(symbol: str, *args, **kwargs):
    # unyt occasionally adds new built-ins, and throws an error for already
    # defined symbols. Log the error and keep going.
    try:
        unyt.define_unit(symbol, *args, **kwargs)
    except RuntimeError as exc:
        logger.info(exc)


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


#===============================================================================
# Unit systems
#===============================================================================
class UnitSystemError(Exception):
    """Base class for unit-system-related errors."""


class UnitSystemExistsError(UnitSystemError):
    """Raised when trying to override a unit system that already exists."""
    def __init__(self, name) -> None:
        super().__init__(f'Unit system with name {name!r} already exists')


class UnitSystemNotFoundError(UnitSystemError):
    """Raised when a unit system is not found in the registry."""
    def __init__(self, name):
        super().__init__(f'Unit system {name!r} not found in registry')


def get_unit_system(system: SystemLike) -> unyt.UnitSystem:
    """Retrieve the actual UnitSystem object from the unit systems registry.

    If passed a UnitSystem object, the object is returned unchanged.

    Parameters
    ----------
    system : str
        The name of the unit system to retrieve.
    """
    if isinstance(system, unyt.UnitSystem):
        return system

    try:
        return unyt.unit_systems.unit_system_registry[str(system)]
    except KeyError as exc:
        raise UnitSystemNotFoundError(system) from exc


def create_unit_system(length, mass, time, name=None, **kwargs):
    """
    Create a new unit system based on `length`, `mass`, and `time`, with
    additional convenience units set with `**kwargs`. Note that this does not
    check to make sure that any convenience units are consistent with the base
    units.

    Parameters
    ----------
    length : UnitLike
        The base length unit.
    mass : UnitLike
        The base mass unit.
    time : UnitLike
        The base time unit.
    name : str, optional
        Name for the unit system. If not provided, a name is generated from the
        length, mass, and time units. (default: None)

    Raises
    ------
    UnitSystemExistsError
        If a unit system with name `name` already exists

    Example
    -------
    >>> system = create_unit_system('mm', 'Gg', 's', force='kN')
    >>> system
    mm_Gg_s Unit System
     Base Units:
      length: mm
      mass: Gg
      time: s
      temperature: K
      angle: rad
      current_mks: A
      luminous_intensity: cd
      logarithmic: Np
     Other Units:
      force: kN
    """
    if name is None:
        name = f'{length}_{mass}_{time}'

    if name in unyt.unit_systems.unit_system_registry:
        raise UnitSystemExistsError(name)

    system = unyt.UnitSystem(
        name,
        length,
        mass,
        time,
        registry=unyt.unit_registry.default_unit_registry,
    )
    for dim, unit in kwargs.items():
        system[dim] = unit

    return system


#---------------------------------------
# US customary system
#---------------------------------------
uscs_system = create_unit_system(
    name='uscs',
    length='inch',
    mass='kblob',
    time='s',
    force='kip',
    stress='ksi',
    moment='kip * inch',
)


#---------------------------------------
# Short repr for UnitSystems
#---------------------------------------
def _UnitSystem_inline_repr(self):
    clsname = self.__class__.__name__
    length = self['length']
    mass = self['mass']
    time = self['time']
    return f'<{clsname} {self.name!r} [{length}, {mass}, {time}]>'


unyt.UnitSystem._inline_repr = _UnitSystem_inline_repr


#===============================================================================
# Utility functions
#===============================================================================
class UnitInputParser():
    """Parse inputs that may or may not have units."""
    def __init__(self,
                 default_units: UnitLike = None,
                 convert: bool = False,
                 check_dims: bool = False,
                 copy: bool = True,
                 registry: unyt.UnitRegistry = None):
        """
        Parameters
        ----------
        default_units : str, unyt.Unit, optional
            Default units to use if inputs don't have units associated already.
            If None, inputs that don't have units will raise an error. Use '' or
            'dimensionless' for explicitly unitless quantities. (default: None)
        convert : bool, optional
            Convert all inputs to `default_units`. Has no effect if
            `default_units` is None. (default: False)
        check_dims : bool, optional
            If True, ensures that input has units compatible with
            `default_units`, but does not convert the input. Has no effect if
            `default_units` is None or `convert` is True. (default: False)
        copy : bool, optional
            Whether to copy underlying input data. (default: True)
        registry : unyt.UnitRegistry, optional
            Registry used to construct new unyt_array instances. Necessary if
            the desired units are not in the default unit registry. (default:
            None)
        """
        self.registry = registry
        self.default_units = default_units
        self.convert = convert
        self.check_dims = check_dims
        self.copy = copy

    def __repr__(self):
        clsname = self.__class__.__name__
        attrs = [
            f'default_units={self.default_units!r}',
            f'convert={self.convert!r}',
            f'check_dims={self.check_dims!r}',
            f'copy={self.copy!r}',
            f'registry={self.registry!r}',
        ]
        return f'{clsname}(' + ', '.join(attrs) + ')'

    #===========================================================================
    # Units handling
    #===========================================================================
    @property
    def default_units(self) -> t.Union[unyt.Unit, None]:
        """Default units to use if inputs don't have units associated already.

        If None, inputs that don't have units will raise an error.
        """
        return self._default_units

    @default_units.setter
    def default_units(self, units):
        self._default_units = self._parse_unit_expression(units)

    def _parse_unit_expression(self, units) -> t.Optional[unyt.Unit]:
        """Parse the given units expression to a Unit object, using the provided
        unit registry.

        None is passed through to represent missing units, as opposed to
        explicit unitlessness.
        """
        if units is not None:
            units = unyt.Unit(units, registry=self.registry)
        return units

    #===========================================================================
    # Dims checking
    #===========================================================================
    def _get_units(self, q) -> unyt.Unit:
        """Get the units of an object."""
        try:
            units = q.units
        except AttributeError:
            units = unyt.dimensionless
        return unyt.Unit(units, registry=self.registry)

    def _check_dimensions(self, a, b):
        """Check that a and b have the same dimensions, and raise an error if
        they do not.
        """
        units_a = self._get_units(a)
        units_b = self._get_units(b)
        dim_a = units_a.dimensions
        dim_b = units_b.dimensions
        if dim_a != dim_b:
            raise UnitConversionError(units_a, dim_a, units_b, dim_b)

    #===========================================================================
    # Parsing
    #===========================================================================
    def __call__(self, in_, units: t.Optional[UnitLike] = None):
        return self.parse(in_, units)

    def parse(self, in_, units: t.Optional[UnitLike] = None) -> unyt_array:
        """Parse the given input expression.

        Accepts the following input styles::

            in_ = 1000           ->  out = 1000*default_units
            in_ = (1000, 'psi')  ->  out = 1000*psi
            in_ = 1000*psi       ->  out = 1000*psi

        Note that if no default units are set, inputs without units will raise
        a ValueError.

        If `convert` is True, then values that come in with units are converted
        to `default_units` when returned::

            in_ = 1000           ->  out_ = 1000*default_units
            in_ = (1000, 'psi')  ->  out_ = (1000*psi).to(default_units)
            in_ = 1000*psi       ->  out_ = (1000*psi).to(default_units)

        If no default units are set, `convert` has no effect.

        Parameters
        ----------
        in_
            The input expression.
        units : optional
            Override value for `default_units`.

        Returns
        -------
        q : unyt.unyt_array

        Raises
        ------
        ValueError
            - If `in_` is a tuple with length != 2.
            - If `default_units` is None and input is received without units.
        unyt.exceptions.UnitConversionError
            If the units of `in_` are not compatible with `default_units`, and
            either `convert` or `check_dims` are true.
        """
        if units is None:
            units = self.default_units
        else:
            units = self._parse_unit_expression(units)

        q = self._parse_internal(in_, units)

        # Convert scalar unyt_arrays to unyt_quantity. Done through reshaping
        # and indexing to make sure we still have the unit registry.
        if q.ndim == 0:
            q = q.reshape(1)[0]

        if self.copy:
            q = q.copy(order='K')

        if units is not None:
            # Skip dims check if convert is True, since the same check will
            # happen internally inside unyt.
            if self.check_dims and not self.convert:
                self._check_dimensions(q, units)

            if self.convert:
                q.convert_to_units(units)

        return q

    #--------------------------------------------------------
    # Parse internal
    #
    # These methods define how 'parse' processes different
    # types into a unyt_array. They should never copy input
    # data, if possible, and they should always return
    # unyt_array, not unyt_quantity (scalarfication is
    # handled inside `parse`).
    #--------------------------------------------------------
    @singledispatchmethod
    def _parse_internal(self, in_, units=None) -> unyt_array:
        if units is None:
            raise ValueError('No default units set; cannot parse object '
                             f'without units {in_!r}')

        return unyt_array(in_, units, registry=self.registry)

    @_parse_internal.register
    def _(self, in_: unyt.unyt_array, units=None):
        return in_

    @_parse_internal.register
    def _(self, in_: tuple, units=None):
        if len(in_) != 2:
            raise ValueError(f'Input tuple must have 2 items (got {len(in_)})')

        return unyt_array(*in_, registry=self.registry)

    if xr is not None:

        @_parse_internal.register
        def _(self, in_: xr.DataArray, units=None):
            value = in_.values
            units = in_.attrs.get('units', units)
            return self._parse_internal(value, units)


def process_unit_input(in_,
                       default_units: UnitLike = None,
                       convert: bool = False,
                       check_dims: bool = False,
                       copy: bool = True,
                       registry: unyt.UnitRegistry = None) -> unyt_array:
    """Process an input value that may or may not have units.

    If the input value doesn't have units, assumes the input is in the requested
    units already.

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
        Default units to use if inputs don't have units associated already. If
        None, inputs that don't have units will raise an error. Use '' or
        'dimensionless' for explicitly unitless quantities. (default: None)
    convert : bool, optional
        Convert all inputs to `default_units`. Has no effect if `default_units`
        is None. (default: False)
    check_dims : bool, optional
        If True, ensures that input has units compatible with `default_units`,
        but does not convert the input. Has no effect if `default_units` is
        None or `convert` is True. (default: False)
    copy : bool, optional
        Whether to copy underlying input data. (default: True)
    registry : unyt.UnitRegistry, optional
        Necessary if the desired units are not in the default unit registry.
        Used to construct the returned unyt.unyt_array object.

    Returns
    -------
    q : unyt.unyt_array

    Raises
    ------
    ValueError
        - If `in_` is a tuple with length != 2.
        - If `default_units` is None and input is received without units.
    unyt.exceptions.UnitConversionError
        If the units of `in_` are not compatible with `default_units`, and
        either `convert` or `check_dims` are true.
    """
    parser = UnitInputParser(default_units=default_units,
                             convert=convert,
                             check_dims=check_dims,
                             copy=copy,
                             registry=registry)
    return parser.parse(in_)


def convert(value, units: UnitLike, registry: unyt.UnitRegistry = None):
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
