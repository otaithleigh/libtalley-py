import pytest
import unyt
from unyt.exceptions import UnitConversionError
from unyt.testing import assert_allclose_units

from libtalley import units
from libtalley.units import get_unit_system, process_unit_input


def test_parse_units_no_default():
    with pytest.raises(ValueError):
        process_unit_input([1, 2, 3, 4])


def test_parse_units_with_default():
    actual = process_unit_input([1, 2, 3, 4], 'ft')
    desired = unyt.unyt_array([1, 2, 3, 4], 'ft')
    assert_allclose_units(actual, desired)


def test_parse_units_check_dims_success():
    desired = unyt.unyt_array([1, 2, 3, 4], 'ft')
    actual = process_unit_input(([1, 2, 3, 4], 'ft'),
                                default_units='inch',
                                check_dims=True)
    assert_allclose_units(actual, desired)


def test_parse_units_check_dims_fail():
    with pytest.raises(UnitConversionError):
        process_unit_input(([1, 2, 3, 4], 'ft'),
                           default_units='kip',
                           check_dims=True)


def test_parse_units_convert_success():
    desired = unyt.unyt_array([12, 24, 36, 48], 'inch')
    actual = process_unit_input(([1, 2, 3, 4], 'ft'),
                                default_units='inch',
                                convert=True)
    assert_allclose_units(actual, desired)


def test_parse_units_bad_tuple():
    with pytest.raises(ValueError):
        process_unit_input(([1, 2, 3, 4], ))


def test_parse_units_already_unyt():
    in_ = unyt.unyt_array([1, 2, 3, 4], 'ft')
    out_ = process_unit_input(in_)
    assert in_ is out_


def test_system_get():
    assert get_unit_system('mks') == unyt.unit_systems.mks_unit_system
