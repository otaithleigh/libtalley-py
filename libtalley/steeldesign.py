import dataclasses
import enum
import fractions
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import unyt

from . import units

#==============================================================================#
#==[ Constants ]===============================================================#
#==============================================================================#
_MODULE_PATH = os.path.dirname(__file__)
TRUE_VALUES = ['T']
FALSE_VALUES = ['F']
NA_VALUES = ['–']


class SteelError(Exception):
    """Steel design errors."""
    pass


#==============================================================================#
#==[ Materials ]===============================================================#
#==============================================================================#
@dataclasses.dataclass
class SteelMaterial():
    """A steel material.

    Parameters
    ----------
    name : str
        Name of the material.
    E : float, unyt.unyt_array
        Elastic modulus. If units are not specified, assumed to be psi.
    Fy : float, unyt.unyt_array
        Yield strength. If units are not specified, assumed to be psi.
    Fu : float, unyt.unyt_array
        Tensile strength. If units are not specified, assumed to be psi.
    Ry : float
        Expected yield strength factor. Dimensionless.
    Rt : float
        Expected tensile strength factor. Dimensionless.
    """
    name: str
    E: float
    Fy: float
    Fu: float
    Ry: float
    Rt: float

    def __post_init__(self):
        self.E = units.process_unit_input(self.E, default_units='psi')
        self.Fy = units.process_unit_input(self.Fy, default_units='psi')
        self.Fu = units.process_unit_input(self.Fu, default_units='psi')
        self.Ry = units.process_unit_input(self.Ry, default_units='dimensionless', convert=True).v
        self.Rt = units.process_unit_input(self.Rt, default_units='dimensionless', convert=True).v
        if self.Fy > self.Fu:
            raise SteelError("SteelMaterial: yield strength must be less than tensile strength")

    @property
    def eFy(self):
        return self.Fy*self.Ry

    @property
    def eFu(self):
        return self.Fu*self.Rt

    @classmethod
    def from_name(cls, name, application=None):
        if application is None:
            material = MATERIALS.loc[name]
        else:
            material = MATERIALS[MATERIALS.application == application].loc[name]
        # Multiple matching materials will be returned as a DataFrame; a single
        # material will be a Series
        if isinstance(material, pd.DataFrame):
            raise SteelError('multiple materials found: specify application to narrow search')
        return cls(name, material.E, material.Fy, material.Fu, material.Ry, material.Rt)


_MATERIALS_FILE = os.path.join(_MODULE_PATH, 'steel-materials.csv')
MATERIALS = pd.read_csv(_MATERIALS_FILE).set_index('name')


#==============================================================================#
#==[ Shapes table ]============================================================#
#==============================================================================#
class ShapesTable():
    def __init__(self, data, units):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Base data for the table.
        units : dict
            Units for each (numeric) column in `data`.
        """
        self.data = data
        self.units = units

    def get_prop(self, shape, prop):
        """Return a property from the table with units.

        If a property is not defined for the given shape, nan is returned.

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
            table; if `prop` is not found in the units dict.
        """
        return unyt.unyt_quantity(self.data.loc[shape][prop], self.units[prop])

    def lightest_shape(self, shape_list):
        """Return the lightest shape (force/length) from the given list.

        Works across different shape series, e.g. comparing an HSS and W works
        fine. If two or more shapes have the same lightest weight, a shape is
        returned, but which is one is undefined.

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
        return self.data.loc[shape_list].W.idxmin()

    @classmethod
    def from_file(
        cls, file, units, true_values=TRUE_VALUES, false_values=FALSE_VALUES, na_values=NA_VALUES
    ):
        """Load a shapes table from a file.

        Parameters
        ----------
        file : str
            Name of the file to load.
        units : dict
            Dictionary of units, with keys corresponding to the column names.
        true_values : list, optional
            List of values to convert to ``True``. (default: ['T'])
        false_values : list, optional
            List of values to convert to ``False``. (default: ['F'])
        na_values : list, optional
            List of values to convert to ``nan``. (default: ['–'])
        """
        data = pd.read_csv(
            file, true_values=true_values, false_values=false_values, na_values=na_values
        ).set_index('AISC_Manual_Label')

        # Convert fractions to floats
        def str2frac2float(s):
            return float(sum(fractions.Fraction(i) for i in s.split()))

        for column in data.columns[data.dtypes == object]:
            if column != 'Type':
                data[column].update(data[column][data[column].notnull()].apply(str2frac2float))
                data[column] = pd.to_numeric(data[column])

        return cls(data, units)


_SHAPES_US_FILE = os.path.join(_MODULE_PATH, 'aisc-shapes-database-v15-0-US.csv.bz2')
_SHAPES_US_UNITS = {
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
}

_SHAPES_SI_FILE = os.path.join(_MODULE_PATH, 'aisc-shapes-database-v15-0-SI.csv.bz2')
_SHAPES_SI_UNITS = {
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
}

shapes_US = ShapesTable.from_file(_SHAPES_US_FILE, _SHAPES_US_UNITS)
shapes_SI = ShapesTable.from_file(_SHAPES_SI_FILE, _SHAPES_SI_UNITS)


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
    return shapes_US.data.loc[shape][prop]


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
    return shapes_US.lightest_shape(shape_list)


#==============================================================================#
#==[ Design ]==================================================================#
#==============================================================================#
class MemberType(enum.Enum):
    BRACE = 'BRACE'
    BEAM = 'BEAM'
    COLUMN = 'COLUMN'


class Ductility(enum.Enum):
    HIGH = 'HIGH'
    MODERATE = 'MODERATE'


def check_seismic_wtr_wide_flange(
    shape, mem_type: MemberType, level: Ductility, Ca, material=MATERIALS['A992Fy50']
) -> (bool, float, float, float, float):
    """Check the width-to-thickness ratio of a W shape for the given ductility.

    Parameters
    ----------
    shape:
        AISC manual name for the shape being checked.
    mem_type:
        MemberType of the member.
    level:
        Level of Ductility being checked.
    Ca:
        = P_u / (phi_c * P_y); adjusts maximum web width-to-thickness ratio
        for beams and columns. Does not affect braces. Should be < 1.0.
    material:
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
    ht = property_lookup(shape, 'h/tw')
    bt = property_lookup(shape, 'bf/2tf')

    common_root = np.sqrt(material.E/material.eFy).to_value('dimensionless')

    if mem_type == MemberType.BRACE:
        ht_max = 1.57*common_root
        bt_max = ht_max
    elif mem_type == MemberType.BEAM or mem_type == MemberType.COLUMN:
        if level == Ductility.MODERATE:
            bt_max = 0.40*common_root
            if Ca <= 0.114:
                ht_max = 3.96*common_root*(1 - 3.04*Ca)
            else:
                ht_max = max(1.29*common_root*(2.12 - Ca), 1.57*common_root)
        elif level == Ductility.HIGH:
            bt_max = 0.32*common_root
            if Ca <= 0.114:
                ht_max = 2.57*common_root*(1 - 1.04*Ca)
            else:
                ht_max = max(0.88*common_root*(2.68 - Ca), 1.57*common_root)
        else:
            raise SteelError("Unsupported ductility level: {}".format(level))
    else:
        raise SteelError("Unsupported member type: {}".format(mem_type))

    WtrResults = namedtuple('WtrResults', ['passed', 'ht', 'ht_max', 'bt', 'bt_max'])
    return WtrResults(ht <= ht_max and bt <= bt_max, ht, ht_max, bt, bt_max)


def brace_capacity(shape, length, material):
    ry = shapes_US.get_prop(shape, 'ry')
    Fe = np.pi**2*material.E/length/ry
    RyFy_Fe = material.Ry*material.Fy/Fe

    if RyFy_Fe <= 2.25:
        Fcre = 0.658**RyFy_Fe*material.Ry*material.Fy
    else:
        Fcre = 0.877*Fe

    Ag = shapes_US.get_prop(shape, 'A')
    tension = material.Ry*material.Fy*Ag
    compression = min(tension, 1/0.877*Fcre*Ag)
    postbuckling = 0.3*compression

    Capacity = namedtuple('Capacity', ['tension', 'compression', 'postbuckling'])
    return Capacity(tension, compression, postbuckling)
