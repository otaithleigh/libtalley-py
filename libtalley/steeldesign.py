from __future__ import annotations

import enum
import logging
import os
import pickle
import sys
import typing as t
from copy import copy

import numpy as np
import pandas as pd
import xlrd

#### Constants ####
_module_path = os.path.dirname(__file__)

default_shapes_version = "v15.0"
shapes_databases = {
    "v15.0": {
        "name": "aisc-shapes-database-v15.0",
        "file": os.path.join(_module_path, "aisc-shapes-database-v15.0.xlsx"),
        "pickle": os.path.join(_module_path, "aisc-shapes-database-v15.0.p"),
        "read_excel": {
            "sheet_name": "Database v15.0",
            "true_values": ['T'],
            "false_values": ['F'],
            "na_values": ['–'],

        }
    }
}


#### Classes ####
class Error(Exception):
    """Steel design errors."""
    pass


# class ShapesDatabase():
#     """An AISC shapes database."""

#     def __init__(self, name, file, sheet):
#         """Initialize a database.
        
#         Parameters
#         ----------
#         name:
#             String identifier for the database.
#         file:
#             Database file.
#         sheet:
#             Sheet name where the actual database is kept in the file.
#         """
#         if not os.path.isfile(file):
#             raise Error("File not found: {}".format(file))

#         self.name = name
#         self._shapes_table = _read_shapes_file(file, sheet)


class Units(enum.Enum):
    """System of units.
    
    Values
    ------
    US:
        Customary US units (kip, in., sec)
    SI:
        International System of units (kg, m, sec)
    """
    US = "US"
    SI = "SI"


class SteelMaterial():
    """A steel material.
    
    Properties
    ----------
    name:
        Name used to refer to the material
    E:
        Elastic modulus, ksi (MPa)
    Fy:
        Yield stress, ksi (MPa)
    Fu:
        Tensile strength, ksi (MPa)
    Ry:
        Expected yield stress factor, dimensionless
    Rt:
        Expected tensile strength factor, dimensionless
    eFy:
        Expected yield stress, ksi (MPa) -- calculated from Fy and Ry
    eFu:
        Expected tensile strength, ksi (MPa) -- calculated from Fu and Rt
    units:
        steeldesign.Units to use for the material properties
    """

    def __init__(self, name, Fy, Fu, Ry, Rt, E=None, units=Units.US):
        if E is None:
            if units is Units.US:
                E = 29000.0
            elif units is Units.SI:
                E = 200000.0
            else:
                raise Error("Unsupported units: {}".format(units))

        if Fy > Fu:
            raise Error("Yield stress must be less than tensile strength")

        self.name = name
        self.E = E
        self.Fy = Fy
        self.Fu = Fu
        self.Ry = Ry
        self.Rt = Rt
        self.eFy = Fy*Ry
        self.eFu = Fu*Rt
        self.units = units

    def __call__(self, units=None):
        if units is None:
            units = self.units

        if units is not self.units:
            Fy = convert_stress(self.Fy, self.units, units)
            Fu = convert_stress(self.Fu, self.units, units)

            new_obj = SteelMaterial(self.name, Fy, Fu, self.Ry, self.Rt, units=units)
        else:
            new_obj = copy(self)

        return new_obj


class materials():
    A992Fy50 = SteelMaterial("A992", 50, 65, 1.1, 1.1)
    A500GrC = SteelMaterial("A500 Gr. C", 50, 65, 1.3, 1.2)


####################################### Functions #######################################
def set_shapes_version(version) -> bool:
    """Select the version of the AISC shapes table to use."""
    global shapes
    success = False

    try:
        shapes_db = shapes_databases[version]
        shapes_file = shapes_db["file"]
        shapes_pickle = shapes_db["pickle"]
        if not os.path.exists(shapes_pickle):
            logger.info("Creating pickle file for version {}".format(version))
            generate_shapes_pickle(shapes_db)

        if os.path.getmtime(shapes_file) > os.path.getmtime(shapes_pickle):
            logger.info("Regenerating pickle file for version {}".format(version))
            generate_shapes_pickle(shapes_db)

        shapes = load_shapes_pickle(shapes_pickle)
        success = True
    except KeyError:
        logger.error("Unsupported shapes database version: {}".format(version))
    finally:
        return success


def generate_shapes_pickle(shapes_db):
    """Create or update the pickle file for a database."""
    table = _read_shapes_file(shapes_db)
    with open(shapes_db['pickle'], 'wb') as the_file:
        pickle.dump(table, the_file)


def load_shapes_pickle(shapes_pickle) -> pd.DataFrame:
    """Load a pickled shapes file."""
    with open(shapes_pickle, 'rb') as the_file:
        return pickle.load(the_file)


def _read_shapes_file(shapes_db):
    try:
        shapes = pd.read_excel(shapes_db['file'], **shapes_db['read_excel'])
    except (FileNotFoundError, xlrd.XLRDError) as err:
        logger.error(err)
        raise err

    return shapes


def property_lookup(shape, prop):
    prop_list = shapes[prop]
    prop_frame = prop_list.loc[shapes.AISC_Manual_Label == shape]
    prop = prop_frame.iloc[0]
    return prop


def convert_stress(value, from_units, to_units):
    """Convert a stress quantity (ksi <--> MPa)."""
    if from_units not in Units:
        raise Error("Unsupported units: {}".format(from_units))
    if to_units not in Units:
        raise Error("Unsupported units: {}".format(to_units))

    if from_units == to_units:
        return value

    if from_units == Units.US:
        return value*6.89475908677537
    elif from_units == Units.SI:
        return value/6.89475908677537


class MemberType(enum.Enum):
    BRACE = "BRACE"
    BEAM = "BEAM"
    COLUMN = "COLUMN"


class Ductility(enum.Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"


def check_seismic_wtr_wide_flange(
    shape, mem_type: MemberType, level: Ductility, Ca, material=materials.A992Fy50
) -> (bool, float, float, float, float):
    """Check the width-to-thickness ratio of a seismic element for the given ductility.

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

    common_root = np.sqrt(material.E/material.eFy)

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
            raise Error("Unsupported ductility level: {}".format(level))
    else:
        raise Error("Unsupported member type: {}".format(mem_type))

    return (ht <= ht_max and bt <= bt_max, ht, ht_max, bt, bt_max)


def lightest_shape(shape_list: t.List[str]):
    """Return the lightest shape (force/length) from the given list.
    
    Works across different shape series, e.g. comparing an HSS and W works fine. If two or
    more shapes have the same lightest weight, a shape is returned, but which is one is
    undefined.

    >>> lightest_shape(['W14X84', 'W44X335'])
    'W14X84'
    >>> lightest_shape(['W14X84', 'HSS4X4X1/2'])
    'HSS4X4X1/2'
    """
    
    return shapes.AISC_Manual_Label[shapes.W[shapes.AISC_Manual_Label.isin(shape_list)].idxmin()]


def latex_name(shape):
    """Return LaTeX code for nicely typesetting a steel section name.
    
    Assumes the "by" part of the section is represented by an 'X', and that
    compound fractions are separated by '-' (hyphen, not endash). Output requires
    the LaTeX package ``nicefrac`` or its superpackage, ``units``.

    Only tested on W and HSS names so far.

    Parameters
    ----------
    shape:
        Name of a steel section.

    Example
    -------
    >>> name = 'HSS3-1/2X3-1/2X3/16'
    >>> latex_name(name)
    'HSS3-\\nicefrac{1}{2}$\\times$3-\\nicefrac{1}{2}$\\times$\\nicefrac{3}{16}'
    """

    def frac_to_nicefrac(frac):
        """Return LaTeX code for a nicefrac from a fraction like '3/16'. No compound fractions!"""
        (numer, denom) = frac.split('/')
        return f"\\nicefrac{{{numer}}}{{{denom}}}"

    shape_parts = shape.split('X')
    for [index, part] in enumerate(shape_parts):
        if '/' in part and '-' in part: # need to activate compound fraction logic
            (front, frac) = part.split('-')
            newfrac = frac_to_nicefrac(frac)
            shape_parts[index] = front + '-' + newfrac
        elif '/' in part: # need to activate fraction logic
            shape_parts[index] = frac_to_nicefrac(part)
            
    return '$\\times$'.join(shape_parts)


#### Initialization ####
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
successful = set_shapes_version(default_shapes_version)
if not successful:
    raise Error("Could not set default steel shapes table. See log for details.")
