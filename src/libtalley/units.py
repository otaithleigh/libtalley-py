import importlib.resources
import typing as t
from functools import singledispatchmethod

import pint

#===============================================================================
# Typing
#===============================================================================
UnitLike = t.Union[str, pint.Unit]

#===============================================================================
# Unit registry
#===============================================================================
ureg = pint.UnitRegistry()
pint.set_application_registry(ureg)

with importlib.resources.path('libtalley', 'units.txt') as p:
    ureg.load_definitions(str(p))

ureg.default_format = '~P'
ureg.default_system = 'uscs'

Quantity = ureg.Quantity


#===============================================================================
# Utility functions
#===============================================================================
class UnitInputParser():
    """Parse inputs that may or may not have units."""
    def __init__(self,
                 default_units: UnitLike = None,
                 convert: bool = False,
                 check_dims: bool = False,
                 registry: pint.UnitRegistry = None):
        """
        Parameters
        ----------
        default_units : str, pint.Unit, optional
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
        registry : pint.UnitRegistry, optional
            Registry used to construct new Quantity instances. Necessary if
            the desired units are not in the default unit registry. (default:
            None)
        """
        if registry is None:
            registry = ureg

        self.registry = registry
        self.default_units = default_units
        self.convert = convert
        self.check_dims = check_dims

    @property
    def default_units(self) -> t.Union[pint.Unit, None]:
        """Default units to use if inputs don't have units associated already.

        If None, inputs that don't have units will raise an error.
        """
        return self._default_units

    @default_units.setter
    def default_units(self, units):
        self._default_units = self._parse_unit_expression(units)

    def _parse_unit_expression(self,
                               units: t.Optional[str]) -> t.Optional[pint.Unit]:
        """Parse the given units expression to a Unit object, using the provided
        unit registry.

        None is passed through to represent missing units, as opposed to
        explicit unitlessness.
        """
        if units is not None:
            units = self.registry.parse_units(units)
        return units

    def __call__(self, in_, units: UnitLike = None) -> pint.Quantity:
        return self.parse(in_, units)

    def parse(self, in_, units: UnitLike = None) -> pint.Quantity:
        """Parse the given input expression.

        Accepts the following input styles::

            in_ = 1000           ->  out = 1000*default_units
            in_ = (1000, 'psi')  ->  out = 1000*psi
            in_ = 1000*psi       ->  out = 1000*psi
            in_ = '1000 psi'     ->  out = 1000*psi

        Note that if no default units are set, inputs without units will raise
        a ValueError.

        If `convert` is True, then values that come in with units are converted
        to `default_units` when returned::

            in_ = 1000           ->  out_ = 1000*default_units
            in_ = (1000, 'psi')  ->  out_ = (1000*psi).to(default_units)
            in_ = 1000*psi       ->  out_ = (1000*psi).to(default_units)
            in_ = '1000 psi'     ->  out_ = (1000*psi).to(default_units)

        If no default units are set, `convert` has no effect.

        Parameters
        ----------
        in_
            The input expression.
        units : optional
            Override value for `default_units`.

        Returns
        -------
        q : pint.Quantity

        Raises
        ------
        ValueError
            - If `in_` is a tuple with length != 2.
            - If `default_units` is None and input is received without units.
        pint.DimensionalityError
            If the units of `in_` are not compatible with `default_units`, and
            either `convert` or `check_dims` are true.
        """
        if units is None:
            units = self.default_units
        else:
            units = self._parse_unit_expression(units)

        q = self._parse_internal(in_, units)

        if units is not None:
            # Skip dims check if convert is True, since the same check will
            # happen internally before converting.
            if self.check_dims and not self.convert:
                self._check_dimensions(q, units)

            if self.convert:
                q = q.to(units)

        return q

    @singledispatchmethod
    def _parse_internal(self, in_, units=None) -> pint.Quantity:
        if units is None:
            raise ValueError('No default units set; cannot parse object '
                             f'without units {in_!r}')

        return self.registry.Quantity(in_, units)

    @_parse_internal.register
    def _(self, in_: str, units=None):
        return self.registry.parse_expression(in_)

    @_parse_internal.register
    def _(self, in_: pint.Quantity, units=None):
        return in_

    @_parse_internal.register
    def _(self, in_: tuple, units=None):
        if len(in_) != 2:
            raise ValueError(f'Input tuple must have 2 items (got {len(in_)})')

        return self.registry.Quantity(*in_)

    def _get_units(self, q) -> pint.Unit:
        """Get the units of an object."""
        try:
            units = q.units
        except AttributeError:
            if isinstance(q, pint.Unit):
                units = q
            else:
                units = self.registry.dimensionless
        return units

    def _check_dimensions(self, a, b):
        units_a = self._get_units(a)
        units_b = self._get_units(b)
        dims_a = units_a.dimensionality
        dims_b = units_b.dimensionality
        if dims_a != dims_b:
            raise pint.DimensionalityError(units_a, units_b, dims_a, dims_b)


def process_unit_input(in_,
                       default_units: UnitLike = None,
                       convert: bool = False,
                       check_dims: bool = False,
                       registry: pint.UnitRegistry = None) -> pint.Quantity:
    """Process an input value that may or may not have units.

    If the input value doesn't have units, assumes the input is in the requested
    units already.

    Accepts the following input styles::

        in_ = 1000           ->  out_ = 1000*default_units
        in_ = (1000, 'psi')  ->  out_ = 1000*psi
        in_ = 1000*psi       ->  out_ = 1000*psi
        in_ = '1000 psi'     ->  out_ = 1000*psi

    Note that if no default units are set, inputs without units will raise
    a ValueError.

    If `convert` is True, then values that come in with units are converted to
    `default_units` when returned::

        in_ = 1000           ->  out_ = 1000*default_units
        in_ = (1000, 'psi')  ->  out_ = (1000*psi).to(default_units)
        in_ = 1000*psi       ->  out_ = (1000*psi).to(default_units)

    Parameters
    ----------
    in_
        Input values.
    default_units : str, pint.Unit, optional
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
    registry : pint.UnitRegistry, optional
        Necessary if the desired units are not in the default unit registry.
        Used to construct the returned Quantity object.

    Returns
    -------
    q : pint.Quantity

    Raises
    ------
    ValueError
        - If `in_` is a tuple with length != 2.
        - If `default_units` is None and input is received without units.
    pint.DimensionalityError
        If the units of `in_` are not compatible with `default_units`, and
        either `convert` or `check_dims` are true.
    """
    parser = UnitInputParser(default_units=default_units,
                             convert=convert,
                             check_dims=check_dims,
                             registry=registry)
    return parser.parse(in_)


def convert(value, units: UnitLike, registry: pint.UnitRegistry = None):
    """Convert an input value to the given units, and return a bare quantity.

    If the input value doesn't have units, assumes the input is in the requested
    units already.

    Parameters
    ----------
    value : array_like
    units : str, pint.Unit
    registry : pint.UnitRegistry, optional

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> convert(30, 's')
    30
    >>> convert(30*ft, 'm')
    9.144
    >>> convert(([24, 36, 48], 'inch'), 'furlong')
    array([0.0030303 , 0.00454545, 0.00606061])
    """
    return process_unit_input(value, units, convert=True, registry=registry).m
