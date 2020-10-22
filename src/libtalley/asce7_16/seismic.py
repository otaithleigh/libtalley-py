"""Seismic load requirements according to ASCE 7-16."""

import numpy as np


def approximate_period(hn: float, Ct: float, x: float) -> float:
    """Calculate the approximate fundamental period T_a (sec).
    
    Parameters
    ----------
    hn:
        Structural height (ft, m).
    Ct:
        Building period coefficient. See Table 12.8-2.
    x:
        Exponent on structural height. See Table 12.8-2.

    Reference: Equation 12.8-7, Table 12.8-2
    """
    return Ct*hn**x


def period_upper_limit_coeff(SD1: float) -> float:
    """Calculate the coefficient for upper limit on calculated period, C_u.
    
    Parameters
    ----------
    SD1:
        Design spectral response acceleration parameter at 1 second (g).

    Reference: Table 12.8-1
    """
    if SD1 >= 0.4:
        cu = 1.4
    elif SD1 <= 0.1:
        cu = 1.7
    else:
        cu = np.interp(SD1, [0.1, 0.15, 0.2, 0.3, 0.4],
                       [1.7, 1.6, 1.5, 1.4, 1.4])
    return cu


def seismic_response_coeff(R, Ie, SDS, SD1, S1, T, T_L) -> float:
    """Calculate the seismic response coefficient, C_s.
    
    Parameters
    ----------
    R:
        Response modification factor
    Ie:
        Seismic importance factor
    SDS:
        Design spectral response acceleration parameter at short periods (g).
    SD1:
        Design spectral response acceleration parameter at 1 second (g).
    S1:
        Mapped MCE spectral response acceleration parameter at 1 second (g).
    T:
        Building fundamental period (sec).
    T_L:
        Long-period transition period (sec).

    Reference: Section 12.8.1.1
    """
    Cs_basic = SDS/(R/Ie)  # Eq. 12.8-2

    if T <= T_L:
        Cs_max = SD1/(T*(R/Ie))  # Eq. 12.8-3
    else:
        Cs_max = SD1*T_L/(T**2*(R/Ie))  # Eq. 12.8-4

    Cs_min = max(0.044*SDS*Ie, 0.01)  # Eq. 12.8-5
    if S1 >= 0.6:
        Cs_min = max(Cs_min, 0.5*S1/(R/Ie))  # Eq. 12.8-6

    Cs = max(min(Cs_basic, Cs_max), Cs_min)
    return Cs
