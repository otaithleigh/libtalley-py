import logging
import typing as t
import uuid
import warnings
from functools import singledispatchmethod
from math import isclose

import numpy as np
import unyt
from unyt import unyt_array
from unyt.dimensions import area, force, length
from unyt.exceptions import IllDefinedUnitSystem, UnitConversionError

from unyt.array import _sanitize_units_convert, LARGE_INPUT
from unyt.unit_registry import _sanitize_unit_system
from unyt.exceptions import MissingMKSCurrent, UnitsNotReducible

try:
    import xarray as xr
except ImportError:
    xr = None

__all__ = [
    'assume_no_em',
    'check_consistent_unit_system',
    'ConsistentUnitSystemCheck',
    'convert',
    'create_unit_system',
    'get_unit_system',
    'logger',
    'process_unit_input',
    'SystemLike',
    'UnitConversionError',
    'UnitInputParser',
    'UnitLike',
    'UnitSystemError',
    'UnitSystemExistsError',
    'UnitSystemNotFoundError',
]

logger = logging.getLogger(__name__)

# ===============================================================================
# Typing
# ===============================================================================
UnitLike = t.Union[str, unyt.Unit]
SystemLike = t.Union[str, unyt.UnitSystem]


# ===============================================================================
# Units and dimensions
# ===============================================================================
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
_safe_define('kilogram_force', (unyt.standard_gravity_mks.value, 'N'))
_safe_define('kgf', (unyt.standard_gravity_mks.value, 'N'))
_safe_define('tonne_force', (1000.0, 'kgf'), prefixable=True)
_safe_define('tf', (1000.0, 'kgf'), prefixable=True)

# Mass
_safe_define('blob', (1.0, 'lbf * s**2 / inch'))
_safe_define('kblob', (1.0, 'kip * s**2 / inch'))
_safe_define('kslug', (1.0, 'kip * s**2 / ft'))
_safe_define('hyl', (1.0, 'kgf * s**2 / m'), prefixable=True)
_safe_define('tonne', (1000.0, 'kg'), prefixable=True)
_safe_define('t', (1000.0, 'kg'), prefixable=True)

# Stress/pressure
_safe_define('ksi', (1000.0, 'psi'))
_safe_define('psf', (1.0, 'lbf / ft**2'))
_safe_define('ksf', (1000.0, 'psf'))

# ---------------------------------------
# Dimensions
# ---------------------------------------
unyt.dimensions.stress = force / area
unyt.dimensions.moment = force * length


# ===============================================================================
# Unit systems
# ===============================================================================
class UnitSystemError(Exception):
    """Base class for unit-system-related errors."""


class UnitSystemConsistencyError(UnitSystemError):
    """Raised when a unit system is not consistent but should be."""


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


def create_unit_system(
    length: UnitLike,
    mass: UnitLike,
    time: UnitLike,
    temperature: UnitLike = None,
    angle: UnitLike = None,
    current_mks: UnitLike = None,
    luminous_intensity: UnitLike = None,
    logarithmic: UnitLike = None,
    name: str | None = None,
    registry: t.Optional[unyt.UnitRegistry] = None,
    consistent: bool = False,
    strict_dims: bool = True,
    **convenience_units: UnitLike,
) -> unyt.UnitSystem:
    """
    Create a new unit system.

    Parameters
    ----------
    length : UnitLike
        The base length unit.
    mass : UnitLike
        The base mass unit.
    time : UnitLike
        The base time unit.
    temperature : UnitLike, optional
        The base temperature unit. (default: 'K')
    angle : UnitLike, optional
        The base angle unit. (default: 'rad')
    current_mks : UnitLike, optional
        The base current unit. (default: 'A')
    luminous_intensity : UnitLike, optional
        The base luminous intensity unit. (default: 'Cd')
    logarithmic : UnitLike, optional
        The base logarithmic unit. (default: 'Np')
    name : str, optional
        Name for the unit system. If not provided, a name is generated from the
        provided base units. (default: None)
    registry : UnitRegistry, optional
        The unit registry for the system. If None, the default unit registry is
        used.
    consistent : bool, optional
        If True, enforce consistency between convenience units and base units
        such that, in base units, convenience units have a magnitude of 1.0. For
        example, force='N' is consistent with MKS (N = 1.0 kg*m/s**2), but
        force='kN' is not (kN = 1e3 kg*m/s**2). (default: False)
    strict_dims : bool, optional
        If True, strictly enforce convenience units having the dimensions they
        map to. (default: True)
    **convenience_units : str, optional
        Mapping of dimension names to convenience units, e.g., ``force='kN'``.

    Raises
    ------
    IllDefinedUnitSystem
        If `strict_dims` is True and convenience units do not strictly map
    UnitSystemExistsError
        If a unit system with name `name` already exists
    UnitSystemConsistencyError
        If `consistent` is True and unit system is not consistent

    Example
    -------
    Unit system that uses millimeters and gigagrams, with a convenience
    kilonewton unit:

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

    Unit system with different base temperature unit, but still auto-generated
    name:

    >>> system = create_unit_system('m', 'kg', 's', temperature='degC')
    >>> system
    m_kg_s_degC Unit System
     Base Units:
      length: m
      mass: kg
      time: s
      temperature: degC
      angle: rad
      current_mks: A
      luminous_intensity: cd
      logarithmic: Np
     Other Units:
    """
    base_units = dict(length_unit=length, mass_unit=mass, time_unit=time)

    # Handle other default base units; don't include them in auto-generated name
    # if not provided. They already have defaults set by the UnitSystem
    # constructor.
    if temperature is not None:
        base_units['temperature_unit'] = temperature
    if angle is not None:
        base_units['angle_unit'] = angle
    if current_mks is not None:
        base_units['current_mks_unit'] = current_mks
    if luminous_intensity is not None:
        base_units['luminous_intensity_unit'] = luminous_intensity
    if logarithmic is not None:
        base_units['logarithmic_unit'] = logarithmic

    # Generate name if not provided.
    if name is None:
        name = '_'.join(map(str, base_units.values()))

    # Check against existing unit systems.
    if name in unyt.unit_systems.unit_system_registry:
        raise UnitSystemExistsError(name)

    if registry is None:
        registry = unyt.unit_registry.default_unit_registry

    # Create new system with base units.
    system = unyt.UnitSystem(str(name), **base_units, registry=registry)
    try:
        # Apply convenience units.
        for dim, unit in convenience_units.items():
            if strict_dims:
                dim_obj = getattr(unyt.dimensions, dim)
                unit_obj = unyt.Unit(unit)
                if unit_obj.dimensions != dim_obj:
                    raise IllDefinedUnitSystem(
                        f'{dim} [{dim_obj}] -> {unit_obj} [{unit_obj.dimensions}]'
                    )

            system[dim] = unit

        # Check consistency if asked to.
        if consistent:
            check = check_consistent_unit_system(system)
            if not check.is_consistent:
                raise UnitSystemConsistencyError(
                    f'Inconsistent unit for dimension {check.bad_dim!r}: '
                    f'1.0 {check.bad_unit} = {check.bad_base}'
                )
    except:
        # Remove bad system from registry
        del unyt.unit_systems.unit_system_registry[name]
        raise

    return system


class ConsistentUnitSystemCheck(t.NamedTuple):
    """
    Parameters
    ----------
    is_consistent : bool
        Whether the unit system is consistent or not.
    bad_dim : str | None
        If not consistent, the dimension corresponding to `bad_unit`.
    bad_unit : unyt.Unit | None
        If not consistent, the first convenience unit that was inconsistent.
    bad_base : unyt.unyt_quantity | None
        If not consistent, the quantity `bad_dim * bad_unit` in base units.
    """

    is_consistent: bool
    bad_dim: t.Optional[str]
    bad_unit: t.Optional[unyt.Unit]
    bad_base: t.Optional[unyt.unyt_quantity]


def check_consistent_unit_system(system: unyt.UnitSystem):
    """Check that a unit system's convenience units are consistent with its base
    units.

    Consistency in this case means convenience units must evaluate to 1.0 in the
    base units. For example, force in newtons is consistent with MKS base units
    (N = 1.0 kg*m/s**2), but force in kilonewtons is not (kN = 10**3 kg*m/s**2).

    Parameters
    ----------
    system : UnitSystem
        The unit system to check.
    """
    # Get base dims as str; subscript from 1:-1 to drop parentheses from
    # dimension names
    base_dims = [str(d)[1:-1] for d in system.base_units]
    base_units = dict(zip(base_dims, system.base_units.values()))

    # Create unit system without convenience units
    temp_name = str(uuid.uuid4())
    system_no_conv = create_unit_system(
        **base_units, name=temp_name, registry=system.registry
    )

    convenience_units = {
        dim: system[dim] for dim in system._dims if dim not in base_dims
    }

    try:
        for dim, unit in convenience_units.items():
            q = 1.0 * unit
            q.convert_to_base(system_no_conv)
            if not isclose(q.v, 1.0):
                is_consistent = False
                bad_dim = dim
                bad_unit = unit
                bad_base = q
                break
        else:
            is_consistent = True
            bad_dim = None
            bad_unit = None
            bad_base = None
    finally:
        del unyt.unit_systems.unit_system_registry[temp_name]

    return ConsistentUnitSystemCheck(is_consistent, bad_dim, bad_unit, bad_base)


# -------------------------------------------------------------------
# Consistent unit systems for typical use, e.g., in OpenSees
# -------------------------------------------------------------------
uscs_system = create_unit_system(
    name='uscs',
    length='inch',
    mass='kblob',
    time='s',
    force='kip',
    stress='ksi',
    moment='kip * inch',
    consistent=True,
)
metric_system = create_unit_system(
    name='metric',
    length='mm',
    mass='Gg',
    time='s',
    force='kN',
    stress='GPa',
    moment='kN*mm',
    consistent=True,
)


# ---------------------------------------
# Short repr for UnitSystems
# ---------------------------------------
def _UnitSystem_inline_repr(self):
    clsname = self.__class__.__name__
    length = self['length']
    mass = self['mass']
    time = self['time']
    return f'<{clsname} {self.name!r} [{length}, {mass}, {time}]>'


if not hasattr(unyt.UnitSystem, '_inline_repr'):
    unyt.UnitSystem._inline_repr = _UnitSystem_inline_repr


# ===============================================================================
# Utility functions
# ===============================================================================
class UnitInputParser:
    """Parse inputs that may or may not have units."""

    def __init__(
        self,
        default_units: UnitLike = None,
        convert: bool = False,
        check_dims: bool = False,
        copy: bool = True,
        registry: unyt.UnitRegistry = None,
    ):
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

    # ===========================================================================
    # Units handling
    # ===========================================================================
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

    # ===========================================================================
    # Dims checking
    # ===========================================================================
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

    # ===========================================================================
    # Parsing
    # ===========================================================================
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

    # --------------------------------------------------------
    # Parse internal
    #
    # These methods define how 'parse' processes different
    # types into a unyt_array. They should never copy input
    # data, if possible, and they should always return
    # unyt_array, not unyt_quantity (scalarfication is
    # handled inside `parse`).
    # --------------------------------------------------------
    @singledispatchmethod
    def _parse_internal(self, in_, units=None) -> unyt_array:
        if units is None:
            raise ValueError(
                f'No default units set; cannot parse object without units {in_!r}'
            )

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


def process_unit_input(
    in_,
    default_units: UnitLike = None,
    convert: bool = False,
    check_dims: bool = False,
    copy: bool = True,
    registry: unyt.UnitRegistry = None,
) -> unyt_array:
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
    parser = UnitInputParser(
        default_units=default_units,
        convert=convert,
        check_dims=check_dims,
        copy=copy,
        registry=registry,
    )
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


# ===============================================================================
# Replacement conversion methods
#
# These are faster unit conversion methods that don't support anything to do
# with electromagnetic units; unyt's default methods spend a lot of time
# checking if something needs special E&M handling.
#
# Last updated for unyt version 2.8.0.
# ===============================================================================


class assume_no_em:
    """Assert that no electromagnetic units are being used in the current
    program. Replaces several unyt methods with faster versions that skip
    E&M-related checks.

    May be used as a context manager, restoring the standard methods afterwards.
    """

    def __init__(self):
        unyt.Unit.get_base_equivalent = _Unit_get_base_equivalent_no_em
        unyt.unyt_array.in_base = _unyt_array_in_base_no_em
        unyt.unyt_array.in_units = _unyt_array_in_units_no_em
        unyt.unyt_array.convert_to_units = _unyt_array_convert_to_units_no_em

    def __enter__(self):
        pass

    def __exit__(self, *exc):
        unyt.Unit.get_base_equivalent = _Unit_get_base_equivalent
        unyt.unyt_array.in_base = _unyt_array_in_base
        unyt.unyt_array.in_units = _unyt_array_in_units
        unyt.unyt_array.convert_to_units = _unyt_array_convert_to_units


# Store old methods
_Unit_get_base_equivalent = unyt.Unit.get_base_equivalent
_unyt_array_in_base = unyt.unyt_array.in_base
_unyt_array_in_units = unyt.unyt_array.in_units
_unyt_array_convert_to_units = unyt.unyt_array.convert_to_units


def _Unit_get_base_equivalent_no_em(self, unit_system=None):
    """Create and return dimensionally-equivalent units in a specified base.

    **Assumes that no electromagnetic units are being used.**

    >>> from unyt import g, cm
    >>> (g/cm**3).get_base_equivalent('mks')
    kg/m**3
    >>> (g/cm**3).get_base_equivalent('solar')
    Mearth/AU**3
    """

    unit_system = _sanitize_unit_system(unit_system, self)

    try:
        new_units = unit_system[self.dimensions]
    except MissingMKSCurrent:
        raise UnitsNotReducible(self.units, unit_system)  # noqa: B904 -- monkeypatching, don't change the error
    return unyt.Unit(new_units, registry=self.registry)


def _unyt_array_in_base_no_em(self, unit_system=None):
    """
    Creates a copy of this array with the data in the specified unit
    system, and returns it in that system's base units.

    **Assumes that no electromagnetic units are being used.**

    Parameters
    ----------
    unit_system : string, optional
        The unit system to be used in the conversion. If not specified,
        the configured default base units of are used (defaults to MKS).

    Examples
    --------
    >>> from unyt import erg, s
    >>> E = 2.5*erg/s
    >>> print(E.in_base("mks"))
    2.5e-07 W
    """
    to_units = self.units.get_base_equivalent(unit_system)
    conv, offset = self.units.get_conversion_factor(to_units, self.dtype)

    new_dtype = np.dtype('f' + str(self.dtype.itemsize))
    conv = new_dtype.type(conv)
    ret = self.v * conv
    if offset:
        ret = ret - offset
    return type(self)(ret, to_units)


def _unyt_array_in_units_no_em(self, units, equivalence=None, **kwargs):
    """
    Creates a copy of this array with the data converted to the
    supplied units, and returns it.

    Optionally, an equivalence can be specified to convert to an
    equivalent quantity which is not in the same dimensions.

    **Assumes that no electromagnetic units are being used.**

    Parameters
    ----------
    units : Unit object or string
        The units you want to get a new quantity in.
    equivalence : string, optional
        The equivalence you wish to use. To see which equivalencies
        are supported for this object, try the ``list_equivalencies``
        method. Default: None
    kwargs: optional
        Any additional keyword arguments are supplied to the
        equivalence

    Raises
    ------
    If the provided unit does not have the same dimensions as the array
    this will raise a UnitConversionError

    Examples
    --------
    >>> from unyt import c, gram
    >>> m = 10*gram
    >>> E = m*c**2
    >>> print(E.in_units('erg'))
    8.987551787368176e+21 erg
    >>> print(E.in_units('J'))
    898755178736817.6 J
    """
    units = _sanitize_units_convert(units, self.units.registry)
    if equivalence is None:
        new_units = units
        (conversion_factor, offset) = self.units.get_conversion_factor(
            new_units, self.dtype
        )

        dsize = self.dtype.itemsize
        if self.dtype.kind in ('u', 'i'):
            large = LARGE_INPUT.get(dsize, 0)
            if large and np.any(np.abs(self.d) > large):
                warnings.warn(
                    "Overflow encountered while converting to units '%s'" % new_units,
                    RuntimeWarning,
                    stacklevel=2,
                )
        new_dtype = np.dtype('f' + str(dsize))
        conversion_factor = new_dtype.type(conversion_factor)
        ret = np.asarray(self.ndview * conversion_factor, dtype=new_dtype)
        if offset:
            np.subtract(ret, offset, ret)

        try:
            new_array = type(self)(
                ret, new_units, bypass_validation=True, name=self.name
            )
        except TypeError:
            # subclasses might not take name as a kwarg
            new_array = type(self)(ret, new_units, bypass_validation=True)

        return new_array
    else:
        return self.to_equivalent(units, equivalence, **kwargs)


def _unyt_array_convert_to_units_no_em(self, units, equivalence=None, **kwargs):
    """
    Convert the array to the given units in-place.

    Optionally, an equivalence can be specified to convert to an
    equivalent quantity which is not in the same dimensions.

    **Assumes that no electromagnetic units are being used.**

    Parameters
    ----------
    units : Unit object or string
        The units you want to convert to.
    equivalence : string, optional
        The equivalence you wish to use. To see which equivalencies
        are supported for this object, try the ``list_equivalencies``
        method. Default: None
    kwargs: optional
        Any additional keyword arguments are supplied to the equivalence

    Raises
    ------
    If the provided unit does not have the same dimensions as the array
    this will raise a UnitConversionError

    Examples
    --------

    >>> from unyt import cm, km
    >>> length = [3000, 2000, 1000]*cm
    >>> length.convert_to_units('m')
    >>> print(length)
    [30. 20. 10.] m
    """
    units = _sanitize_units_convert(units, self.units.registry)
    if equivalence is None:
        new_units = units
        (conv_factor, offset) = self.units.get_conversion_factor(new_units, self.dtype)

        self.units = new_units
        values = self.d
        # if our dtype is an integer do the following somewhat awkward
        # dance to change the dtype in-place. We can't use astype
        # directly because that will create a copy and not update self
        if self.dtype.kind in ('u', 'i'):
            # create a copy of the original data in floating point
            # form, it's possible this may lose precision for very
            # large integers
            dsize = values.dtype.itemsize
            new_dtype = 'f' + str(dsize)
            large = LARGE_INPUT.get(dsize, 0)
            if large and np.any(np.abs(values) > large):
                warnings.warn(
                    "Overflow encountered while converting to units '%s'" % new_units,
                    RuntimeWarning,
                    stacklevel=2,
                )
            float_values = values.astype(new_dtype)
            # change the dtypes in-place, this does not change the
            # underlying memory buffer
            values.dtype = new_dtype
            self.dtype = new_dtype
            # actually fill in the new float values now that our
            # dtype is correct
            np.copyto(values, float_values)
        values *= conv_factor

        if offset:
            np.subtract(values, offset, values)
    else:
        self.convert_to_equivalent(units, equivalence, **kwargs)
