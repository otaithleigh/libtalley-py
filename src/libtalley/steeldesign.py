from __future__ import annotations

import dataclasses
import enum
import fractions
import importlib.resources
from typing import ClassVar, Dict, NamedTuple, Union
import warnings

import numpy as np
import pandas as pd
import unyt

from . import units

#===============================================================================
# Constants
#===============================================================================
TRUE_VALUES = ['T']
FALSE_VALUES = ['F']
NA_VALUES = ['\N{EN DASH}']


class SteelError(Exception):
    """Steel design errors."""
    pass


#===============================================================================
# Materials
#===============================================================================
@dataclasses.dataclass
class SteelMaterial():
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
            raise SteelError('SteelMaterial: yield strength must'
                             ' be less than tensile strength')

    @property
    def eFy(self):
        return self.Fy*self.Ry

    @property
    def eFu(self):
        return self.Fu*self.Rt

    @classmethod
    def from_name(cls, name: str, grade: str = None, application: str = None):
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
            If multiple materials match the provided information. For example,
            requesting an A500 steel also requires specifying a grade (A or B).
        """
        name, grade = _check_deprecated_material(name, grade)

        def normalize(input):
            if pd.isna(input):
                return slice(None)
            else:
                return str(input).casefold()

        material = cls._get_materials_db().loc[normalize(name),
                                               normalize(grade),
                                               normalize(application)]

        # Lookup succeeds if we get a Series (exact indexes) or if we get a
        # DataFrame of length 1 (one or more indexes were sliced, but still only
        # one result returned).
        if isinstance(material, pd.DataFrame):
            if len(material) != 1:
                raise ValueError('Multiple materials found: specify grade '
                                 'and/or application to narrow search')
            material = material.iloc[0]

        name, grade, application = material.name
        display_grade = '' if pd.isna(grade) else f' Gr. {grade}'
        display_name = f'{name}{display_grade} ({application})'
        return cls(display_name.title(), **material)

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
        warnings.warn(f'Convert old material name {name!r} to '
                      f'({_name!r}, grade={grade!r})')
        return (_name, grade)
    else:
        return (name, grade)


def _load_materials_db(filename):
    with importlib.resources.path('libtalley', filename) as p:
        material_df = pd.read_csv(p, index_col=['name', 'grade', 'application'])
    material_df.sort_index(inplace=True)
    return material_df


#===============================================================================
# Shapes table
#===============================================================================
class ShapesTable():
    def __init__(self, data: pd.DataFrame, units: pd.Series, name: str = None):
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
    def from_file(cls,
                  file,
                  units,
                  name=None,
                  true_values=TRUE_VALUES,
                  false_values=FALSE_VALUES,
                  na_values=NA_VALUES):
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
        """
        data: pd.DataFrame = pd.read_csv(file,
                                         true_values=true_values,
                                         false_values=false_values,
                                         na_values=na_values)
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
class _ShapesResource():
    name: str
    units: Dict[str, str]
    filename: str
    package: str = 'libtalley'

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
    def register(cls, key, name, units, filename, package='libtalley'):
        if key in cls._registry:
            raise ValueError(f'resource with key {key!r} is already registered')

        cls._registry[key] = cls(name, units, filename, package)

    @classmethod
    def from_registry(cls, key):
        return cls._registry[key]


_ShapesResource.register('US',
                         name='AISC Shapes Database v15.0 (US)',
                         filename='aisc-shapes-database-v15-0-US.csv.bz2',
                         units={
                             'W': 'lbf/ft',
                             'A': 'inch**2',
                             'd': 'inch',
                             'ddet': 'inch',
                             'Ht': 'inch',
                             'h': 'inch',
                             'OD': 'inch',
                             'bf': 'inch',
                             'bfdet': 'inch',
                             'B': 'inch',
                             'b': 'inch',
                             'ID': 'inch',
                             'tw': 'inch',
                             'twdet': 'inch',
                             'twdet/2': 'inch',
                             'tf': 'inch',
                             'tfdet': 'inch',
                             't': 'inch',
                             'tnom': 'inch',
                             'tdes': 'inch',
                             'kdes': 'inch',
                             'kdet': 'inch',
                             'k1': 'inch',
                             'x': 'inch',
                             'y': 'inch',
                             'eo': 'inch',
                             'xp': 'inch',
                             'yp': 'inch',
                             'bf/2tf': 'dimensionless',
                             'b/t': 'dimensionless',
                             'b/tdes': 'dimensionless',
                             'h/tw': 'dimensionless',
                             'h/tdes': 'dimensionless',
                             'D/t': 'dimensionless',
                             'Ix': 'inch**4',
                             'Zx': 'inch**3',
                             'Sx': 'inch**3',
                             'rx': 'inch',
                             'Iy': 'inch**4',
                             'Zy': 'inch**3',
                             'Sy': 'inch**3',
                             'ry': 'inch',
                             'Iz': 'inch**4',
                             'rz': 'inch',
                             'Sz': 'inch**3',
                             'J': 'inch**4',
                             'Cw': 'inch**6',
                             'C': 'inch**3',
                             'Wno': 'inch**2',
                             'Sw1': 'inch**4',
                             'Sw2': 'inch**4',
                             'Sw3': 'inch**4',
                             'Qf': 'inch**3',
                             'Qw': 'inch**3',
                             'ro': 'inch',
                             'H': 'dimensionless',
                             'tan(α)': 'dimensionless',
                             'Iw': 'inch**4',
                             'zA': 'inch',
                             'zB': 'inch',
                             'zC': 'inch',
                             'wA': 'inch',
                             'wB': 'inch',
                             'wC': 'inch',
                             'SzA': 'inch**3',
                             'SzB': 'inch**3',
                             'SzC': 'inch**3',
                             'SwA': 'inch**3',
                             'SwB': 'inch**3',
                             'SwC': 'inch**3',
                             'rts': 'inch',
                             'ho': 'inch',
                             'PA': 'inch',
                             'PA2': 'inch',
                             'PB': 'inch',
                             'PC': 'inch',
                             'PD': 'inch',
                             'T': 'inch',
                             'WGi': 'inch',
                             'WGo': 'inch',
                         })

_ShapesResource.register('SI',
                         name='AISC Shapes Database v15.0 (SI)',
                         filename='aisc-shapes-database-v15-0-SI.csv.bz2',
                         units={
                             'W': 'kg/m',
                             'A': 'mm**2',
                             'd': 'mm',
                             'ddet': 'mm',
                             'Ht': 'mm',
                             'h': 'mm',
                             'OD': 'mm',
                             'bf': 'mm',
                             'bfdet': 'mm',
                             'B': 'mm',
                             'b': 'mm',
                             'ID': 'mm',
                             'tw': 'mm',
                             'twdet': 'mm',
                             'twdet/2': 'mm',
                             'tf': 'mm',
                             'tfdet': 'mm',
                             't': 'mm',
                             'tnom': 'mm',
                             'tdes': 'mm',
                             'kdes': 'mm',
                             'kdet': 'mm',
                             'k1': 'mm',
                             'x': 'mm',
                             'y': 'mm',
                             'eo': 'mm',
                             'xp': 'mm',
                             'yp': 'mm',
                             'bf/2tf': 'dimensionless',
                             'b/t': 'dimensionless',
                             'b/tdes': 'dimensionless',
                             'h/tw': 'dimensionless',
                             'h/tdes': 'dimensionless',
                             'D/t': 'dimensionless',
                             'Ix': '1e6*mm**4',
                             'Zx': '1e3*mm**3',
                             'Sx': '1e3*mm**3',
                             'rx': 'mm',
                             'Iy': '1e6*mm**4',
                             'Zy': '1e3*mm**3',
                             'Sy': '1e3*mm**3',
                             'ry': 'mm',
                             'Iz': '1e6*mm**4',
                             'rz': 'mm',
                             'Sz': '1e3*mm**3',
                             'J': '1e3*mm**4',
                             'Cw': '1e9*mm**6',
                             'C': '1e3*mm**3',
                             'Wno': 'mm**2',
                             'Sw1': '1e6*mm**4',
                             'Sw2': '1e6*mm**4',
                             'Sw3': '1e6*mm**4',
                             'Qf': '1e3*mm**3',
                             'Qw': '1e3*mm**3',
                             'ro': 'mm',
                             'H': 'dimensionless',
                             'tan(α)': 'dimensionless',
                             'Iw': '1e6*mm**4',
                             'zA': 'mm',
                             'zB': 'mm',
                             'zC': 'mm',
                             'wA': 'mm',
                             'wB': 'mm',
                             'wC': 'mm',
                             'SzA': '1e3*mm**3',
                             'SzB': '1e3*mm**3',
                             'SzC': '1e3*mm**3',
                             'SwA': '1e3*mm**3',
                             'SwB': '1e3*mm**3',
                             'SwC': '1e3*mm**3',
                             'rts': 'mm',
                             'ho': 'mm',
                             'PA': 'mm',
                             'PA2': 'mm',
                             'PB': 'mm',
                             'PC': 'mm',
                             'PD': 'mm',
                             'T': 'mm',
                             'WGi': 'mm',
                             'WGo': 'mm',
                         })

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
    warnings.warn('Replace `property_lookup(shape, prop)` with '
                  '`shapes_US.get_prop(shape, prop)`')
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
    warnings.warn('Replace `lightest_shape(shape_list)` with '
                  '`shapes_US.lightest_shape(shape_list)`')
    return shapes_US.lightest_shape(shape_list)


#===============================================================================
# Design
#===============================================================================
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


def check_seismic_wtr_wide_flange(shape: str,
                                  mem_type: Union[str, MemberType],
                                  level: Union[str, Ductility],
                                  Ca: float,
                                  material: SteelMaterial = None) -> WtrResults:
    """Check the width-to-thickness ratio of a W shape for the given ductility.

    Parameters
    ----------
    shape : str
        AISC manual name for the shape being checked.
    mem_type : MemberType
        MemberType of the member.
    level : Ductility
        Level of Ductility being checked.
    Ca : float
        = P_u / (φ_c * P_y); adjusts maximum web width-to-thickness ratio
        for beams and columns. Does not affect braces. Should be < 1.0.
    material : SteelMaterial, optional
        Material to use (default A992, Fy = 50 ksi)

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
    AISC 341-16, Table D1.1 (pp. 9.1-14 -- 9.1-17)
    """
    mem_type = MemberType(mem_type)
    level = Ductility(level)
    wtr_max = {
        ('BRACE', 'MODERATE'): _wtr_brace,
        ('BRACE', 'HIGH'): _wtr_brace,
        ('COLUMN', 'MODERATE'): _wtr_beam_column_moderate,
        ('COLUMN', 'HIGH'): _wtr_beam_column_high,
        ('BEAM', 'MODERATE'): _wtr_beam_column_moderate,
        ('BEAM', 'HIGH'): _wtr_beam_column_high,
    }[mem_type.value, level.value]

    if material is None:
        material = SteelMaterial.from_name('A992')
    E_eFy = np.sqrt(material.E/material.eFy).to_value('dimensionless')

    ht = shapes_US.get_prop(shape, 'h/tw').value
    bt = shapes_US.get_prop(shape, 'bf/2tf').value
    ht_max, bt_max = wtr_max(E_eFy, Ca)

    return WtrResults(ht <= ht_max and bt <= bt_max, ht, ht_max, bt, bt_max)


def _wtr_brace(E_eFy, Ca):
    """Maximum width-to-thickness ratio for a brace."""
    ht_max = 1.57*E_eFy
    bt_max = ht_max
    return ht_max, bt_max


def _wtr_beam_column_moderate(E_eFy, Ca):
    """Maximum width-to-thickness ratio for a moderately ductile beam/column."""
    bt_max = 0.40*E_eFy
    if Ca <= 0.114:
        ht_max = 3.96*E_eFy*(1 - 3.04*Ca)
    else:
        ht_max = max(1.29*E_eFy*(2.12 - Ca), 1.57*E_eFy)
    return ht_max, bt_max


def _wtr_beam_column_high(E_eFy, Ca):
    """Maximum width-to-thickness ratio for a highly ductile beam/column."""
    bt_max = 0.32*E_eFy
    if Ca <= 0.114:
        ht_max = 2.57*E_eFy*(1 - 1.04*Ca)
    else:
        ht_max = max(0.88*E_eFy*(2.68 - Ca), 1.57*E_eFy)
    return ht_max, bt_max


class Capacity(NamedTuple):
    tension: unyt.unyt_quantity
    compression: unyt.unyt_quantity
    postbuckling: unyt.unyt_quantity


def brace_capacity(shape: str, length: unyt.unyt_quantity,
                   material: SteelMaterial) -> Capacity:
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
    Fe = np.pi**2*material.E/(length/ry)**2
    RyFy = material.eFy
    RyFy_Fe = RyFy/Fe

    if RyFy_Fe <= 2.25:
        Fcre = 0.658**RyFy_Fe*RyFy
    else:
        Fcre = 0.877*Fe

    Ag = shape['A']
    tension = RyFy*Ag
    compression = min(tension, 1/0.877*Fcre*Ag)
    postbuckling = 0.3*compression

    return Capacity(tension, compression, postbuckling)
