from __future__ import annotations

import enum
import warnings

import numpy as np
from scipy.stats import lognorm
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, interp2d

from . import asce7_16
from .units import convert

__all__ = [
    'acmrxx',
    'beta_total',
    'design_response_spectrum',
    'fundamental_period',
    'mapped_value',
    'seismic_response_coeff',
    'sf1',
    'smt',
    'ssf',
]


def acmrxx(beta_total, collapse_prob, xin=0.622):
    """Compute the acceptable value of the adjusted collapse margin ratio (ACMR).

    Parameters
    ----------
    beta_total:
        Total uncertainty present in the system
    collapse_prob:
        Collapse probability being checked (e.g. 0.20 for ACMR20)
    xin = 0.622:
        Starting value for the nonlinear solution. Tweak this if there are
        convergence issues.

    Ref: FEMA P695 Section 7.4
    """

    # Solve lognorm.cdf as a substitute for MATLAB's logninv function
    def f(x):
        return lognorm.cdf(x, beta_total) - collapse_prob

    X = fsolve(f, xin)

    return 1/X[0]


# Uncertainty values for each rating
_rating_values = {
    'A': 0.10,
    'B': 0.20,
    'C': 0.35,
    'D': 0.50,
}


def beta_total(rating_DR: str,
               rating_TD: str,
               rating_MDL: str,
               mu_T: float = 3.0) -> float:
    """Compute the total uncertainty present in th system.

    Parameters
    ----------
    rating_DR:
        Rating of the design requirements for the system
    rating_TD:
        Rating of the test data for the system
    rating_MDL:
        Rating of the model's representation of the system
    mu_T:
        Period-based ductility

    Ref: FEMA P695 Section 7.3.1
    """
    beta_DR = _rating_values[rating_DR.upper()]
    beta_TD = _rating_values[rating_TD.upper()]
    beta_MDL = _rating_values[rating_MDL.upper()]
    beta_RTR = min((0.1 + 0.1*mu_T, 0.4))
    beta = np.sqrt(beta_RTR**2 + beta_DR**2 + beta_TD**2 + beta_MDL**2)

    return round(beta*40)/40


_mapped_value_dict = {
    "dmax": {
        "ss": 1.5,
        # Actually 0.60 but "should be taken as less than 0.60" *eyeroll*
        "s1": 0.59999999999,
        "fa": 1.0,
        "fv": 1.50,
        "sms": 1.50,
        "sm1": 0.90,
        "sds": 1.0,
        "sd1": 0.60,
        "ts": 0.60,
    },
    "dmin": {
        "ss": 0.55,
        "s1": 0.132,
        "fa": 1.36,
        "fv": 2.28,
        "sms": 0.75,
        "sm1": 0.30,
        "sds": 0.50,
        "sd1": 0.20,
        "ts": 0.40,
    },
    "cmin": {
        "ss": 0.33,
        "s1": 0.083,
        "fa": 1.53,
        "fv": 2.4,
        "sms": 0.50,
        "sm1": 0.20,
        "sds": 0.33,
        "sd1": 0.133,
        "ts": 0.40,
    },
    "bmin": {
        "ss": 0.156,
        "s1": 0.042,
        "fa": 1.6,
        "fv": 2.4,
        "sms": 0.25,
        "sm1": 0.10,
        "sds": 0.167,
        "sd1": 0.067,
        "ts": 0.40,
    }
}
_mapped_value_dict["cmax"] = _mapped_value_dict["dmin"]
_mapped_value_dict["bmax"] = _mapped_value_dict["cmin"]


def mapped_value(value: str, sdc: str):
    """Retrieve the mapped seismic parameter for a given design category.

    Parameters
    ----------
    value:
        Mapped parameter to retrieve (SS, S1, Fa, Fv, SMS, SM1, SDS, SD1, Ts)
    """
    return _mapped_value_dict[sdc.lower()][value.lower()]


_T_INTERP = [
    0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6,
    1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0
]
_SNRT_INTERP = [
    0.785, 0.781, 0.767, 0.754, 0.755, 0.742, 0.607, 0.541, 0.453, 0.402, 0.350,
    0.303, 0.258, 0.210, 0.169, 0.149, 0.134, 0.119, 0.106, 0.092, 0.081, 0.063,
    0.053, 0.046, 0.041
]


def sf1(T, sdc):
    """Calculate scale factor 1, which scales ground motions to the MCE.

    Parameters
    ----------
    T : float
        Period of the structure

    sdc : str
        Seismic design category


    Ref: FEMA P695 Section
    """
    T = convert(T, 's')
    if T <= _T_INTERP[0] or T >= _T_INTERP[-1]:
        raise ValueError(f"Period is out of range: T = {T}")

    f = interp1d(_T_INTERP, _SNRT_INTERP)
    SNRT = f(T)
    SMT = smt(T, sdc)

    return SMT/SNRT


def smt(T, sdc):
    """Calculate the SMT.

    Parameters
    ----------
    T : float
        Fundamental period of the structure (seconds)

    sdc : str
        Seismic design category
    """
    T = convert(T, 's')
    SM1 = mapped_value("SM1", sdc)
    SMS = mapped_value("SMS", sdc)
    if T <= SM1/SMS:
        return SMS
    else:
        return SM1/T


#-------------------------------------------
# Spectral shape factor stuff
#-------------------------------------------
_Z_SSF_DICT = {
    "Dmax":
        np.array([
            [1.00, 1.05, 1.10, 1.13, 1.18, 1.22, 1.28, 1.33],
            [1.00, 1.05, 1.11, 1.14, 1.20, 1.24, 1.30, 1.36],
            [1.00, 1.06, 1.11, 1.15, 1.21, 1.25, 1.32, 1.38],
            [1.00, 1.06, 1.12, 1.16, 1.22, 1.27, 1.35, 1.41],
            [1.00, 1.06, 1.13, 1.17, 1.24, 1.29, 1.37, 1.44],
            [1.00, 1.07, 1.13, 1.18, 1.25, 1.31, 1.39, 1.46],
            [1.00, 1.07, 1.14, 1.19, 1.27, 1.32, 1.41, 1.49],
            [1.00, 1.07, 1.15, 1.20, 1.28, 1.34, 1.44, 1.52],
            [1.00, 1.08, 1.16, 1.21, 1.29, 1.36, 1.46, 1.55],
            [1.00, 1.08, 1.16, 1.22, 1.31, 1.38, 1.49, 1.58],
            [1.00, 1.08, 1.17, 1.23, 1.32, 1.40, 1.51, 1.61],
        ]),
    "Dmin":
        np.array([
            [1.00, 1.02, 1.04, 1.06, 1.08, 1.09, 1.12, 1.14],
            [1.00, 1.02, 1.05, 1.07, 1.09, 1.11, 1.13, 1.16],
            [1.00, 1.03, 1.06, 1.08, 1.10, 1.12, 1.15, 1.18],
            [1.00, 1.03, 1.06, 1.08, 1.11, 1.14, 1.17, 1.20],
            [1.00, 1.03, 1.07, 1.09, 1.13, 1.15, 1.19, 1.22],
            [1.00, 1.04, 1.08, 1.10, 1.14, 1.17, 1.21, 1.25],
            [1.00, 1.04, 1.08, 1.11, 1.15, 1.18, 1.23, 1.27],
            [1.00, 1.04, 1.09, 1.12, 1.17, 1.20, 1.25, 1.30],
            [1.00, 1.05, 1.10, 1.13, 1.18, 1.22, 1.27, 1.32],
            [1.00, 1.05, 1.10, 1.14, 1.19, 1.23, 1.30, 1.35],
            [1.00, 1.05, 1.11, 1.15, 1.21, 1.25, 1.32, 1.37],
        ])
}
_Z_SSF_DICT["Cmax"] = _Z_SSF_DICT["Dmin"]
_Z_SSF_DICT["Cmin"] = _Z_SSF_DICT["Dmin"]
_Z_SSF_DICT["Bmax"] = _Z_SSF_DICT["Dmin"]
_Z_SSF_DICT["Bmin"] = _Z_SSF_DICT["Dmin"]

_X_MU_T = [1.0, 1.1, 1.5, 2, 3, 4, 6, 8]
_Y_T = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


def ssf(T, mu_t, sdc):
    """Compute the spectral shape factor (SSF).

    Parameters
    ----------
    T:
        Period of the structure
    mu_t:
        Period-based ductility
    sdc:
        Seismic design category

    Ref: FEMA P695 Section
    """
    T = convert(T, 's')
    try:
        Z_SSF = _Z_SSF_DICT[sdc]
    except KeyError:
        raise ValueError(f"Unknown seismic design category: {sdc}")

    if mu_t < 1:
        raise ValueError("mu_t must be greater than or equal to 1")

    if T <= 0.5:
        if mu_t >= 8:
            ssf = Z_SSF[0, -1]
        else:
            f = interp1d(_X_MU_T, Z_SSF[0, :])
            ssf = f(mu_t)[0]
    elif T >= 1.5:
        if mu_t >= 8:
            ssf = Z_SSF[-1, -1]
        else:
            f = interp1d(_X_MU_T, Z_SSF[-1, :])
            ssf = f(mu_t)[0]
    else:
        if mu_t >= 8:
            f = interp1d(_Y_T, Z_SSF[:, -1])
            ssf = f(mu_t)[0]
        else:
            f = interp2d(_X_MU_T, _Y_T, Z_SSF)
            ssf = f(mu_t, T)[0]

    return ssf


def fundamental_period(hn, Ct, x, sdc):
    """Calculate the fundamental period of the structure.

    Uses the following equation:

    .. math:: T = \\max(C_u C_t h_n^x, 0.25)

    Where :math:`C_u` is the coefficient on the upper limit of the calculated
    period from ASCE 7.

    Parameters
    ----------
    hn : float
        Height of the structure (ft)

    Ct : float
        Building period coefficient.

    x : float
        Exponent on building height.

    sdc : str
        Seismic design category.

    Returns
    -------
    T : float
        Fundamental period in seconds.
    """
    SD1 = mapped_value('SD1', sdc)
    Ta = asce7_16.seismic.approximate_period(hn, Ct, x)
    Cu = asce7_16.seismic.period_upper_limit_coeff(SD1)

    return np.maximum(Cu*Ta, 0.25)


def seismic_response_coeff(R, T, sdc):
    """Calculate the seismic response coefficient, C_s.

    Parameters
    ----------
    R : float
        Response modification factor.

    T : float
        Fundamental period (s).

    sdc : str
        Seismic design category (Dmax, Dmin, etc.).

    Note that this function follows the assumptions and restrictions enforced by
    FEMA P695; namely, it is used only with mapped values from the
    ``mapped_values`` function, and for structures with periods of 4.0 s or
    lower. For a more general function, see ``asce7_16.seismic_response_coeff``.
    """
    T = convert(T, 's')
    if T > 4.0:
        warnings.warn('seismic_response_coeff: Given period '
                      f'(T = {T} s) is out of FEMA P695 range')

    Ts = mapped_value('Ts', sdc)
    SD1 = mapped_value('SD1', sdc)
    SDS = mapped_value('SDS', sdc)

    if T <= Ts:
        Cs = SDS/R
    else:
        Cs = max(SD1/(T*R), 0.044*SDS)

    Cs = max(Cs, 0.01)

    return Cs


def design_response_spectrum(sdc, tl=4.0):
    """Generate the design response spectrum for the given seismic design
    category.

    Parameters
    ----------
    sdc : {'Dmax', 'Dmin', 'Cmax', 'Cmin', 'Bmax', 'Bmin'}
        Seismic design category.

    tl : float, optional
        Transition period between constant velocity and constant displacement
        response domains. Note that FEMA P695 does not include values for T_L
        since it restricts archetypes to having fundamental periods of 4 sec or
        less, and thus T_L should not apply to most FEMA P695 studies.
        (default: 4.0)
    """
    sds = mapped_value('SDS', sdc)
    sd1 = mapped_value('SD1', sdc)
    return asce7_16.seismic.design_response_spectrum(sds, sd1, tl)
