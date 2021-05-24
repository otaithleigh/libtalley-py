import numpy as np
import pytest
import unyt
from unyt.exceptions import UnitConversionError
from unyt.testing import assert_allclose_units

try:
    import xarray as xr
except ImportError:
    xr = None

from libtalley import units
from libtalley.units import UnitInputParser, get_unit_system, process_unit_input


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
    assert_allclose_units(in_, out_)


def test_parse_units_already_unyt_no_copy():
    in_ = unyt.unyt_array([1, 2, 3, 4], 'ft')
    out_ = process_unit_input(in_, copy=False)
    assert in_ is out_


def test_parse_units_array():
    in_ = np.array([1, 2, 3, 4])
    actual = process_unit_input(in_, 'ft')
    desired = unyt.unyt_array([1, 2, 3, 4], 'ft')
    assert actual.base is not in_
    assert_allclose_units(actual, desired)


def test_parse_units_no_copy_array():
    in_ = np.array([1, 2, 3, 4])
    actual = process_unit_input(in_, 'ft', copy=False)
    desired = unyt.unyt_array([1, 2, 3, 4], 'ft')
    assert actual.base is in_
    assert_allclose_units(actual, desired)


if xr is not None:
    def test_parse_units_xarray():
        in_ = xr.DataArray([1, 2, 3, 4], attrs={'units': 'mm'})
        actual = process_unit_input(in_)
        desired = unyt.unyt_array([1, 2, 3, 4], 'mm')
        assert_allclose_units(actual, desired)

    def test_parse_units_xarray_no_copy():
        in_ = xr.DataArray([4, 3, 2, 1], attrs={'units': 'm'})
        actual = process_unit_input(in_, copy=False)
        desired = unyt.unyt_array([4, 3, 2, 1], 'm')
        assert actual.base is in_.values
        assert_allclose_units(actual, desired)

    def test_parse_units_xarray_no_units():
        in_ = xr.DataArray([4, 3, 2, 1])
        with pytest.raises(ValueError):
            process_unit_input(in_)

    def test_parse_units_xarray_no_accept_bad_casing():
        in_ = xr.DataArray([4, 3, 2, 1], attrs={'Units': 'm'})
        with pytest.raises(ValueError):
            process_unit_input(in_)


def test_parse_units_override_default():
    parser = UnitInputParser()
    desired = unyt.unyt_array([12, 24, 36, 48], 'inch')
    actual = parser.parse([12, 24, 36, 48], 'inch')
    assert_allclose_units(actual, desired)


def test_parse_units_override_default_check_dims_fail():
    parser = UnitInputParser(check_dims=True)
    with pytest.raises(UnitConversionError):
        parser.parse((30, 'kip'), 'ksf')


def test_system_get():
    assert get_unit_system('mks') == unyt.unit_systems.mks_unit_system
