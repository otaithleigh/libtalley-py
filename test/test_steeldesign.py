import unyt
import pytest

import libtalley.steeldesign as steel


# ===============================================================================
# Material lookups
# ===============================================================================
def test_material_lookup_exact_match():
    """Material is specified exactly."""
    material = steel.SteelMaterial.from_name('A500', grade='C', application='HSS')
    assert unyt.allclose_units(material.E, 29000 * unyt.ksi)
    assert unyt.allclose_units(material.Fy, 50 * unyt.ksi)
    assert unyt.allclose_units(material.Fu, 62 * unyt.ksi)
    assert unyt.allclose_units(material.Ry, 1.3)
    assert unyt.allclose_units(material.Rt, 1.2)


def test_material_lookup_slice_match_1():
    """Material is only partially specified, but sufficiently to match."""
    material = steel.SteelMaterial.from_name('A500', 'C')
    assert unyt.allclose_units(material.E, 29000 * unyt.ksi)
    assert unyt.allclose_units(material.Fy, 50 * unyt.ksi)
    assert unyt.allclose_units(material.Fu, 62 * unyt.ksi)
    assert unyt.allclose_units(material.Ry, 1.3)
    assert unyt.allclose_units(material.Rt, 1.2)


def test_material_lookup_slice_match_2():
    """Material is only partially specified, but sufficiently to match."""
    material = steel.SteelMaterial.from_name('A992')
    assert unyt.allclose_units(material.E, 29000 * unyt.ksi)
    assert unyt.allclose_units(material.Fy, 50 * unyt.ksi)
    assert unyt.allclose_units(material.Fu, 65 * unyt.ksi)
    assert unyt.allclose_units(material.Ry, 1.1)
    assert unyt.allclose_units(material.Rt, 1.1)


def test_material_lookup_too_many_results():
    with pytest.raises(ValueError, match=r'^Multiple materials found:\n'):
        steel.SteelMaterial.from_name('A500')


def test_material_lookup_no_results():
    with pytest.raises(ValueError, match=r'^No materials found$'):
        steel.SteelMaterial.from_name('ThisIsNotAMaterial')


# ===============================================================================
# Width-to-thickness ratio check
# ===============================================================================
@pytest.fixture()
def check_params():
    # shape, mem_type, level, Pr
    Ag = 8.79
    Ry = 1.1
    Fy = 50
    Py = unyt.unyt_quantity(Ry * Fy * Ag, 'kip')
    return ('W12x30', 'BEAM', 'HIGH', 0.52 * Py)


def test_wtr_check_Pr_sign_should_not_matter(check_params):
    shape, mem_type, level, Pr = check_params
    check_pos = steel.check_seismic_wtr_wide_flange(shape, mem_type, level, Pr)
    check_neg = steel.check_seismic_wtr_wide_flange(shape, mem_type, level, -Pr)
    assert check_pos == check_neg


def test_wtr_check_2016(check_params):
    check = steel.check_seismic_wtr_wide_flange(*check_params, edition=2016)
    assert check.passed is False
    assert check.ht < check.ht_max
    assert check.bt > check.bt_max


def test_wtr_check_2022(check_params):
    check = steel.check_seismic_wtr_wide_flange(*check_params, edition=2022)
    assert check.passed is False
    assert check.ht > check.ht_max
    assert check.bt > check.bt_max
