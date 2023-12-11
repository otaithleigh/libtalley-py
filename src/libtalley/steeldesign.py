from __future__ import annotations

import abc
import dataclasses
import enum
import fractions
import importlib.resources
import json
import math
import warnings
from typing import ClassVar, Dict, Literal, NamedTuple, Tuple, Union

import numpy as np
import pandas as pd
import unyt

from . import units

# ===============================================================================
# Constants
# ===============================================================================
TRUE_VALUES = ['T']
FALSE_VALUES = ['F']
NA_VALUES = ['\N{EN DASH}']


class SteelError(Exception):
    """Steel design errors."""


# ===============================================================================
# Materials
# ===============================================================================
@dataclasses.dataclass
class SteelMaterial:
    """A steel material.

    Parameters
    ----------
    name : str
        Name of the material.
    E : float, unyt.unyt_array
        Elastic modulus. If units are not specified, assumed to be ksi.
    Fy : float, unyt.unyt_array
        Design yield strength. If units are not specified, assumed to be ksi.
    Fu : float, unyt.unyt_array
        Design tensile strength. If units are not specified, assumed to be ksi.
    Ry : float
        Expected yield strength factor. Dimensionless.
    Rt : float
        Expected tensile strength factor. Dimensionless.
    """

    name: str
    E: unyt.unyt_quantity
    Fy: unyt.unyt_quantity
    Fu: unyt.unyt_quantity
    Ry: float
    Rt: float

    def __post_init__(self):
        get_stress = units.UnitInputParser(default_units='ksi')
        get_factor = units.UnitInputParser(default_units='', convert=True)

        self.E = get_stress(self.E)
        self.Fy = get_stress(self.Fy)
        self.Fu = get_stress(self.Fu)
        self.Ry = get_factor(self.Ry).item()
        self.Rt = get_factor(self.Rt).item()

        if self.Fy > self.Fu:
            raise SteelError(
                'SteelMaterial: yield strength must be less than tensile strength'
            )

    @property
    def eFy(self):
        return self.Fy * self.Ry

    @property
    def eFu(self):
        return self.Fu * self.Rt

    @classmethod
    def from_name(
        cls, name: str, grade: str | None = None, application: str | None = None
    ):
        """Look up a steel material based on name, grade, and application.

        All look-ups are case-insensitive.

        Parameters
        ----------
        name : str
            The name of the material specification (e.g. 'A572').
        grade : int, str, optional
            The grade of the material (e.g. 50 or 'B'). Only required if
            multiple grades are available.
        application : {'brb', 'hot-rolled', 'hss', 'plate', 'rebar'}, optional
            The application the material is used for. Only required if multiple
            applications are defined for the same material specification.

        Raises
        ------
        ValueError
            - If no materials are found.
            - If multiple materials match the provided information. For example,
              requesting an A500 steel also requires specifying a grade (A or
              B).
        """
        name, grade = _check_deprecated_material(name, grade)

        # Construct database query
        query_name = str(name).casefold()
        query_list = [f'name == {query_name!r}']
        if grade is not None:
            query_grade = str(grade).casefold()
            query_list.append(f'grade == {query_grade!r}')
        if application is not None:
            query_application = str(application).casefold()
            query_list.append(f'application == {query_application!r}')
        query = ' & '.join(query_list)

        # Lookup -- should get a DataFrame of length 1.
        material = cls._get_materials_db().query(query)
        if len(material) == 0:
            raise ValueError('No materials found')
        elif len(material) != 1:
            raise ValueError(f'Multiple materials found:\n{material}')

        data = material.iloc[0]
        name, grade, application = data.name
        display_grade = '' if pd.isna(grade) else f' Gr. {grade}'
        display_name = f'{name}{display_grade} ({application})'
        return cls(display_name.title(), **data)

    @classmethod
    def available_materials(cls):
        """Return a DataFrame of the available materials, whose rows can be used
        as lookups for `from_name`."""
        return cls._get_materials_db().index.to_frame(index=False)

    @classmethod
    def _get_materials_db(cls) -> pd.DataFrame:
        # Delay loading the materials database until required.
        try:
            return cls._materials_db
        except AttributeError:
            cls._materials_db = _load_materials_db('steel-materials-us.csv')
            return cls._materials_db


def _check_deprecated_material(name, grade):
    if ' Gr. ' in name:
        _name, grade = name.split(' Gr. ')
        warnings.warn(
            f'Convert old material name {name!r} to ({_name!r}, grade={grade!r})',
            stacklevel=3,
        )
        return (_name, grade)
    else:
        return (name, grade)


def _load_materials_db(filename):
    with importlib.resources.path(__package__, filename) as p:
        material_df = pd.read_csv(p, index_col=['name', 'grade', 'application'])
    material_df.sort_index(inplace=True)
    return material_df


# ===============================================================================
# Shapes table
# ===============================================================================
class ShapesTable:
    def __init__(self, data: pd.DataFrame, units: pd.Series, name: str | None = None):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Base data for the table.
        units : pd.Series
            Units for each (numeric) column in `data`.
        name : str, optional
            Name for this table.
        """
        self.data = data
        self.units = units
        self.name = name

    def __repr__(self):
        clsname = f'{self.__class__.__module__}.{self.__class__.__name__}'
        selfname = '' if self.name is None else f' {self.name!r}'
        nshapes = len(self.data)

        return f'<{clsname}{selfname} with {nshapes} shapes>'

    def get_prop(self, shape: str, prop: str):
        """Return a property from the table with units.

        If a property is not defined for the given shape, nan is returned. If
        units are not defined for the property, the raw quantity is returned
        from the table.

        Parameters
        ----------
        shape : str
            Name of the shape to look up.
        prop : str
            Name of the property to look up.

        Returns
        -------
        q : unyt.unyt_quantity
            Value of the property with units.

        Raises
        ------
        KeyError
            If `shape` is not found in the table; if `prop` is not found in the
            table.
        """
        value = self.data.at[shape.casefold(), prop]
        units = self.units.get(prop)
        if units is None:
            return value
        else:
            return unyt.unyt_quantity(value, units)

    def get_shape(self, shape: str, include_units: bool = True):
        """
        Parameters
        ----------
        shape : str
            Name of the shape to retrieve.
        include_units : bool
            Whether to include units in the returned shape data. (default: True)

        Returns
        -------
        pd.Series
        """
        shape_data = self.data.loc[shape.casefold()].dropna()
        if include_units:
            for prop, value in shape_data.items():
                units = self.units.get(prop)
                if units is not None:
                    shape_data[prop] = unyt.unyt_quantity(value, units)
        return shape_data

    def lightest_shape(self, shape_list):
        """Return the lightest shape (force/length) from the given list.

        Works across different shape series, e.g. comparing an HSS and W works
        fine. If two or more shapes have the same lightest weight, a shape is
        returned, but which one is returned is undefined.

        Parameters
        ----------
        shape_list : list
            List of shapes to check.

        Examples
        --------
        >>> lightest_shape(['W14X82', 'W44X335'])
        'W14X82'
        >>> lightest_shape(['W14X82', 'HSS4X4X1/2'])
        'HSS4X4X1/2'
        """
        index = self.data.loc[pd.Series(shape_list).str.casefold()].W.idxmin()
        return self.data.at[index, 'AISC_Manual_Label']

    @classmethod
    def from_file(
        cls,
        file,
        units,
        name=None,
        true_values=TRUE_VALUES,
        false_values=FALSE_VALUES,
        na_values=NA_VALUES,
    ):
        """Load a shapes table from a file.

        Parameters
        ----------
        file : str, file-like
            Name of the file to load.
        units : dict
            Dictionary of units, with keys corresponding to the column names.
        name : str
            Name to use for the created ShapesTable.
        true_values : list, optional
            List of values to convert to ``True``. (default: ['T'])
        false_values : list, optional
            List of values to convert to ``False``. (default: ['F'])
        na_values : list, optional
            List of values to convert to ``nan``. (default: ['–']) (note that
            this is an en-dash U+2013, not an ASCII hyphen U+002D)
        """  # noqa: RUF002
        data: pd.DataFrame = pd.read_csv(
            file,
            true_values=true_values,
            false_values=false_values,
            na_values=na_values,
        )
        data.index = pd.Index(data['AISC_Manual_Label'].str.casefold(), name='')

        # Convert fractions to floats
        def str2frac2float(s):
            return float(sum(fractions.Fraction(i) for i in s.split()))

        for column in data.columns[data.dtypes == object]:
            if column not in ['AISC_Manual_Label', 'Type']:
                notnull_data = data[column][data[column].notnull()]
                converted_data = notnull_data.apply(str2frac2float)
                data[column].update(converted_data)
                data[column] = pd.to_numeric(data[column])

        data.sort_index(inplace=True)
        units = pd.Series(units)
        units.sort_index(inplace=True)
        return cls(data, units, name=name)

    @staticmethod
    def from_resource(key):
        return _ShapesResource.from_registry(key).get_resource()


@dataclasses.dataclass
class _ShapesResource:
    name: str
    units: Dict[str, str]
    filename: str
    package: str = __package__

    _registry: ClassVar[Dict[str, _ShapesResource]] = {}

    def load_resource(self):
        with importlib.resources.path(self.package, self.filename) as p:
            return ShapesTable.from_file(p, self.units, name=self.name)

    def get_resource(self):
        try:
            resource = self._resource
        except AttributeError:
            resource = self._resource = self.load_resource()

        return resource

    @classmethod
    def register(cls, key, name, units, filename, package=__package__):
        if key in cls._registry:
            raise ValueError(f'resource with key {key!r} is already registered')

        cls._registry[key] = cls(name, units, filename, package)

    @classmethod
    def from_registry(cls, key):
        return cls._registry[key]


def _register_resources():
    s = importlib.resources.read_text(__package__, 'shapes-resources.json')
    shapes_resources: Dict[str, dict] = json.loads(s)['shapes_resources']
    for key, resource in shapes_resources.items():
        _ShapesResource.register(
            key,
            name=resource['name'],
            filename=resource['filename'],
            units=resource['units'],
            package=resource.get('package', __package__),
        )


_register_resources()
shapes_US = ShapesTable.from_resource('US')
shapes_SI = ShapesTable.from_resource('SI')


def property_lookup(shape, prop):
    """Retrieve a property from the US shapes table.

    Returns values without units for legacy reasons.

    Parameters
    ----------
    shape : str
        Name of the shape to look up.
    prop : str
        Name of the property to look up.
    """
    warnings.warn(
        'Replace `property_lookup(shape, prop)` with '
        '`shapes_US.get_prop(shape, prop)`',
        stacklevel=2,
    )
    return shapes_US.data.at[str(shape).casefold(), prop]


def lightest_shape(shape_list):
    """Return the lightest shape (force/length) from the given list.

    Works across different shape series, e.g. comparing an HSS and W works fine.
    If two or more shapes have the same lightest weight, a shape is returned,
    but which is one is undefined.

    Parameters
    ----------
    shape_list : list
        List of shapes to check.

    Examples
    --------
    >>> lightest_shape(['W14X82', 'W44X335'])
    'W14X82'
    >>> lightest_shape(['W14X82', 'HSS4X4X1/2'])
    'HSS4X4X1/2'
    """
    warnings.warn(
        'Replace `lightest_shape(shape_list)` with '
        '`shapes_US.lightest_shape(shape_list)`',
        stacklevel=2,
    )
    return shapes_US.lightest_shape(shape_list)


# ===============================================================================
# Design
# ===============================================================================
class MemberType(enum.Enum):
    BRACE = 'BRACE'
    BEAM = 'BEAM'
    COLUMN = 'COLUMN'


class Ductility(enum.Enum):
    HIGH = 'HIGH'
    MODERATE = 'MODERATE'


class WtrResults(NamedTuple):
    passed: bool
    ht: float
    ht_max: float
    bt: float
    bt_max: float


class WideFlangeWtrChecker(abc.ABC):
    ht: float
    bt: float
    E_eFy: float
    Ca: float

    def __init__(self, shape: str, material: SteelMaterial, Pr: unyt.unyt_quantity):
        self.ht = shapes_US.get_prop(shape, 'h/tw').item()
        self.bt = shapes_US.get_prop(shape, 'bf/2tf').item()
        self.E_eFy = math.sqrt(material.E / material.eFy)
        self.Ca = self._calculate_Ca(shape, material, Pr)

    @abc.abstractmethod
    def _calculate_Ca(
        self, shape: str, material: SteelMaterial, Pr: unyt.unyt_quantity
    ) -> float:
        """Calculate Ca for the given shape and factored axial load.

        Ca is defined differently for different editions of the Provisions.
        """

    def wtr_max(self, mem_type: MemberType, level: Ductility) -> Tuple[float, float]:
        return {
            ('BRACE', 'MODERATE'): self._wtr_brace_moderate,
            ('BRACE', 'HIGH'): self._wtr_brace_high,
            ('COLUMN', 'MODERATE'): self._wtr_beam_column_moderate,
            ('COLUMN', 'HIGH'): self._wtr_beam_column_high,
            ('BEAM', 'MODERATE'): self._wtr_beam_column_moderate,
            ('BEAM', 'HIGH'): self._wtr_beam_column_high,
        }[mem_type.name, level.name](self.E_eFy, self.Ca)

    def check(self, mem_type: MemberType, level: Ductility) -> WtrResults:
        ht_max, bt_max = self.wtr_max(mem_type, level)
        return WtrResults(
            self.ht <= ht_max and self.bt <= bt_max, self.ht, ht_max, self.bt, bt_max
        )


class WideFlangeWtrChecker2016(WideFlangeWtrChecker):
    def _calculate_Ca(
        self, shape: str, material: SteelMaterial, Pr: unyt.unyt_quantity
    ) -> float:
        # LRFD: Ca = Pu / (φ Ry Fy Ag)
        Ag = shapes_US.get_prop(shape, 'A')
        φ = 0.9
        return (abs(Pr) / (φ * material.eFy * Ag)).to_value('dimensionless')

    @staticmethod
    def _wtr_brace_moderate(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a moderately ductile wide-flange brace."""
        ht_max = 1.57 * E_eFy
        bt_max = 0.40 * E_eFy
        return ht_max, bt_max

    @staticmethod
    def _wtr_brace_high(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a highly ductile wide-flange brace."""
        ht_max = 1.57 * E_eFy
        bt_max = 0.32 * E_eFy
        return ht_max, bt_max

    @staticmethod
    def _wtr_beam_column_moderate(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a moderately ductile beam/column."""
        bt_max = 0.40 * E_eFy
        if Ca <= 0.114:
            ht_max = 3.96 * E_eFy * (1 - 3.04 * Ca)
        else:
            ht_max = max(1.29 * E_eFy * (2.12 - Ca), 1.57 * E_eFy)
        return ht_max, bt_max

    @staticmethod
    def _wtr_beam_column_high(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a highly ductile beam/column."""
        bt_max = 0.32 * E_eFy
        if Ca <= 0.114:
            ht_max = 2.57 * E_eFy * (1 - 1.04 * Ca)
        else:
            ht_max = max(0.88 * E_eFy * (2.68 - Ca), 1.57 * E_eFy)
        return ht_max, bt_max


class WideFlangeWtrChecker2022(WideFlangeWtrChecker):
    def _calculate_Ca(
        self, shape: str, material: SteelMaterial, Pr: unyt.unyt_quantity
    ) -> float:
        # LRFD: Ca = α Pr / (Ry Fy Ag); α = 1.0
        Ag = shapes_US.get_prop(shape, 'A')
        return (abs(Pr) / (material.eFy * Ag)).to_value('dimensionless')

    @staticmethod
    def _wtr_brace_moderate(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a moderately ductile wide-flange brace."""
        ht_max = 1.49 * E_eFy
        bt_max = 0.38 * E_eFy
        return ht_max, bt_max

    @staticmethod
    def _wtr_brace_high(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a highly ductile wide-flange brace."""
        ht_max = 1.49 * E_eFy
        bt_max = 0.30 * E_eFy
        return ht_max, bt_max

    @staticmethod
    def _wtr_beam_column_moderate(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a moderately ductile beam/column."""
        bt_max = 0.38 * E_eFy
        if Ca <= 0.113:
            ht_max = 3.76 * (1 - 3.05 * Ca) * E_eFy
        else:
            ht_max = max(2.61 * (1 - 0.49 * Ca) * E_eFy, 1.56 * E_eFy)
        return ht_max, bt_max

    @staticmethod
    def _wtr_beam_column_high(E_eFy, Ca):
        """Maximum width-to-thickness ratio for a highly ductile beam/column."""
        bt_max = 0.30 * E_eFy
        if Ca <= 0.113:
            ht_max = 2.45 * (1 - 1.04 * Ca) * E_eFy
        else:
            ht_max = max(2.26 * (1 - 0.38 * Ca) * E_eFy, 1.56 * E_eFy)
        return ht_max, bt_max


def check_seismic_wtr_wide_flange(
    shape: str,
    mem_type: Union[str, MemberType],
    level: Union[str, Ductility],
    Pr: unyt.unyt_quantity,
    material: SteelMaterial | None = None,
    edition: Literal[2016, 2022] = 2016,
) -> WtrResults:
    """Check the width-to-thickness ratio of a W shape for the given ductility.

    Parameters
    ----------
    shape : str
        AISC manual name for the shape being checked.
    mem_type : MemberType
        MemberType of the member.
    level : Ductility
        Level of Ductility being checked.
    Pr : unyt_quantity
        Factored axial load demand.
    material : SteelMaterial, optional
        Material to use. (default A992, Fy = 50 ksi)
    edition : {2016, 2022}
        Edition of the AISC Seismic Provisions to use when checking the width-
        to-thickness ratio. (default: 2016)

    Returns
    -------
    passed:
        Bool pass/fail. (ht <= ht_max and bt <= bt_max)
    ht:
        The h/tw value for the section
    ht_max:
        The maximum h/tw value for the section
    bt:
        The bf/2tf value for the section
    bt_max:
        The maximum bf/2tf value for the section

    Reference
    ---------
    - AISC 341-16, Table D1.1 (pp. 9.1-14 -- 9.1-17)
    - AISC 341-22, Tables D1.1a, D1.1b (pp. 9.1-16 -- 9.1-21)
    """
    mem_type = MemberType(mem_type)
    level = Ductility(level)
    if material is None:
        material = SteelMaterial.from_name('A992')

    if edition == 2016:
        checker_cls = WideFlangeWtrChecker2016
    elif edition == 2022:
        checker_cls = WideFlangeWtrChecker2022
    else:
        raise ValueError(f'Unsupported Seismic Provisions edition: {edition!r}')

    return checker_cls(shape, material, Pr).check(mem_type, level)


class Capacity(NamedTuple):
    tension: unyt.unyt_quantity
    compression: unyt.unyt_quantity
    postbuckling: unyt.unyt_quantity


def brace_capacity(
    shape: str, length: unyt.unyt_quantity, material: SteelMaterial
) -> Capacity:
    """
    Parameters
    ----------
    shape : str
        Steel shape of the brace.
    length : unyt_quantity
        Unbraced length of the brace.
    material : SteelMaterial
        Brace material.
    """
    shape = shapes_US.get_shape(shape)
    ry = shape['ry']
    Fe = np.pi**2 * material.E / (length / ry) ** 2
    RyFy = material.eFy
    RyFy_Fe = RyFy / Fe

    if RyFy_Fe <= 2.25:
        Fcre = 0.658**RyFy_Fe * RyFy
    else:
        Fcre = 0.877 * Fe

    Ag = shape['A']
    tension = RyFy * Ag
    compression = min(tension, 1 / 0.877 * Fcre * Ag)
    postbuckling = 0.3 * compression

    return Capacity(tension, compression, postbuckling)
