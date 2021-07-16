import math

from libtalley import fema_p695


def test_seismic_response_coeff():
    Cs = fema_p695.seismic_response_coeff(8, 3.1, 'Dmax')
    assert math.isclose(Cs, 0.044)


def test_seismic_response_coeff_mce():
    Cs = fema_p695.seismic_response_coeff(8, 3.1, 'Dmax', level='mce')
    assert math.isclose(Cs, 0.066)
