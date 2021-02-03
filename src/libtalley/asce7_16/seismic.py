"""Seismic load requirements according to ASCE 7-16."""

import json

import numpy as np
import requests

from ..units import convert


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
    return np.interp(SD1, [0.1, 0.15, 0.2, 0.3, 0.4], [1.7, 1.6, 1.5, 1.4, 1.4])


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
    R_Ie = R/Ie
    Cs_basic = SDS/R_Ie  # Eq. 12.8-2

    if T <= T_L:
        Cs_max = SD1/(T*R_Ie)  # Eq. 12.8-3
    else:
        Cs_max = SD1*T_L/(T**2*R_Ie)  # Eq. 12.8-4

    Cs_min = max(0.044*SDS*Ie, 0.01)  # Eq. 12.8-5
    if S1 >= 0.6:
        Cs_min = max(Cs_min, 0.5*S1/R_Ie)  # Eq. 12.8-6

    Cs = max(min(Cs_basic, Cs_max), Cs_min)
    return Cs


def vertical_force_dist(w, h, T):
    """Calculate the vertical force distribution factors C_v.

    Parameters
    ----------
    w : array_like
        Seismic weight at each story
    h : array_like
        Ground-to-level height for each story
    T : float
        Building fundamental period (sec)
    """
    T = convert(T, 's')
    k = np.interp(T, [0.5, 2.5], [1, 2])
    whk = np.asanyarray(w)*np.asanyarray(h)**k
    return whk/whk.sum()


def design_response_spectrum(sds, sd1, tl):
        """Generate a design response spectrum function.

        Parameters
        ----------
        sds : float
            Design spectral acceleration at short periods (g)
        sd1 : float
            Design spectral acceleration at T = 1 sec (g)
        tl : float
            Long-period transition period (sec)

        Returns
        -------
        Sa(T: array_like) -> array_like

        Example
        -------
        >>> Sa = design_response_spectrum(1.0, 0.60, )
        """
        Ts = sd1/sds
        T0 = 0.2*Ts

        def Sa(T):
            T = convert(T, 's')
            return np.piecewise(T, [
                T < T0,
                (T >= T0) & (T < Ts),
                (T >= Ts) & (T < tl),
                T >= tl,
            ], [
                lambda T: sds*(0.4 + 0.6*T/T0),
                lambda T: sds,
                lambda T: sd1/T,
                lambda T: sd1*tl/T**2,
            ])

        return Sa


class SiteSpecificParameters():
    def __init__(self, pgauh, pgad, pga, fpga, pgam, ssrt, crs, ssuh, ssd, ss,
                 fa, sms, sds, sdcs, s1rt, cr1, s1uh, s1d, s1, fv, sm1, sd1,
                 sdc1, sdc, tl, cv, twoPeriodDesignSpectrum,
                 twoPeriodMCErSpectrum, verticalDesignSpectrum,
                 verticalMCErSpectrum):
        self._pgauh: float = pgauh
        self._pgad: float = pgad
        self._pga: float = pga
        self._fpga: float = fpga
        self._pgam: float = pgam
        self._ssrt: float = ssrt
        self._crs: float = crs
        self._ssuh: float = ssuh
        self._ssd: float = ssd
        self._ss: float = ss
        self._fa: float = fa
        self._sms: float = sms
        self._sds: float = sds
        self._sdcs: str = sdcs
        self._s1rt: float = s1rt
        self._cr1: float = cr1
        self._s1uh: float = s1uh
        self._s1d: float = s1d
        self._s1: float = s1
        self._fv: float = fv
        self._sm1: float = sm1
        self._sd1: float = sd1
        self._sdc1: str = sdc1
        self._sdc: str = sdc
        self._tl: float = tl
        self._cv: float = cv
        self._twoPeriodDesignSpectrum = np.asarray(twoPeriodDesignSpectrum)
        self._twoPeriodMCErSpectrum = np.asarray(twoPeriodMCErSpectrum)
        self._verticalDesignSpectrum = np.asarray(verticalDesignSpectrum)
        self._verticalMCErSpectrum = np.asarray(verticalMCErSpectrum)
        self._Sa = design_response_spectrum(sds, sd1, tl)

    @property
    def pgauh(self):
        """Uniform-hazard (2% probability of exceedance in 50 years) peak ground
        acceleration"""
        return self._pgauh

    @property
    def pgad(self):
        """Factored deterministic acceleration value (peak ground acceleration)"""
        return self._pgad

    @property
    def pga(self):
        """Maximum considered earthquake geometric mean (MCE_G) peak ground
        acceleration"""
        return self._pga

    @property
    def fpga(self):
        """Site amplification factor at peak ground acceleration"""
        return self._fpga

    @property
    def pgam(self):
        """Site-modified peak ground acceleration"""
        return self._pgam

    @property
    def ssrt(self):
        """Probabilistic risk-targeted ground motion (0.2 s)"""
        return self._ssrt

    @property
    def crs(self):
        """Mapped risk coefficient value (0.2 s)"""
        return self._crs

    @property
    def ssuh(self):
        """Factored uniform-hazard (2% probability of exceedance in 50 years)
        spectral acceleration (0.2 s)"""
        return self._ssuh

    @property
    def ssd(self):
        """Factored deterministic acceleration value (0.2 s)"""
        return self._ssd

    @property
    def ss(self):
        """Risk-targeted maximum considered earthquake (MCE_R) ground motion
        response acceleration (0.2 s)"""
        return self._ss

    @property
    def fa(self):
        """Site amplification factor (0.2 s)"""
        return self._fa

    @property
    def sms(self):
        """Site-modified spectral acceleration value (0.2 s)"""
        return self._sms

    @property
    def sds(self):
        """Numeric seismic design value (0.2 s)"""
        return self._sds

    @property
    def sdcs(self):
        """Seismic design category from SDS"""
        return self._sdcs

    @property
    def s1rt(self):
        """Probabilistic risk-targeted ground motion (1.0 s)"""
        return self._s1rt

    @property
    def cr1(self):
        """Mapped risk coefficient value (1.0 s)"""
        return self._cr1

    @property
    def s1uh(self):
        """Factored uniform-hazard (2% probability of exceedance in 50 years
        spectral acceleration (1.0 s)"""
        return self._s1uh

    @property
    def s1d(self):
        """Factored deterministic acceleration value (1.0 s)"""
        return self._s1d

    @property
    def s1(self):
        """Risk-targeted maximum considered earthquake (MCE_R) ground motion
        response acceleration (1.0 s)"""
        return self._s1

    @property
    def fv(self):
        """Site amplification factor (1.0 s)"""
        return self._fv

    @property
    def sm1(self):
        """Site-modified spectral acceleration value (1.0 s)"""
        return self._sm1

    @property
    def sd1(self):
        """Numeric seismic design value (1.0 s)"""
        return self._sd1

    @property
    def sdc1(self):
        """Seismic design category from SD1"""
        return self._sdc1

    @property
    def sdc(self):
        """Seismic design category from sdcs and sdc1"""
        return self._sdc

    @property
    def tl(self):
        """Long-period transition period in seconds"""
        return self._tl

    @property
    def cv(self):
        """Vertical coefficient"""
        return self._cv

    @property
    def twoPeriodDesignSpectrum(self):
        """Design horizontal response spectrum"""
        return self._twoPeriodDesignSpectrum

    @property
    def twoPeriodMCErSpectrum(self):
        """Risk-targeted maximum considered earthquake (MCE_R) horizontal
        response spectrum"""
        return self._twoPeriodMCErSpectrum

    @property
    def verticalDesignSpectrum(self):
        """Design vertical response spectrum"""
        return self._verticalDesignSpectrum

    @property
    def verticalMCErSpectrum(self):
        """Risk-targeted maximum considered earthquake (MCE_R) vertical response
        spectrum"""
        return self._verticalMCErSpectrum

    def seismic_response_coeff(self, R, T):
        """Calculate the seismic response coefficient C_s for a building at this
        site.

        Parameters
        ----------
        R : float
            Response modification factor
        T : float
            Building fundamental period (seconds)
        """
        return seismic_response_coeff(R, 1.0, self.sds, self.sd1, self.s1, T,
                                      self.tl)

    def design_spectral_acceleration(self, T):
        """Calculate the design spectral acceleration Sa for a building at this
        site.
        
        Parameters
        ----------
        T : float
            Building fundamental period (seconds)
        """
        return self._Sa(T)

    @classmethod
    def from_usgs(cls, latitude, longitude, risk_category, site_class):
        """Get site-specific parameters from the USGS web service.

        Parameters
        ----------
        latitude : int
            Latitude of the site
        longitude : int
            Longitude of the site
        risk_category : str
            Risk category for the structure (e.g. 'II')
        site_class : str
            NEHRP site classification (e.g. 'D')
        """
        response = requests.get(
            'https://earthquake.usgs.gov/ws/designmaps/asce7-16.json',
            params={
                'latitude': latitude,
                'longitude': longitude,
                'riskCategory': risk_category,
                'siteClass': site_class,
                'title': '',  # I'm not sure what this is other than metadata
            })
        if response.status_code != 200:
            if response.status_code == 500:
                error_desc = json.loads(response.content)['response']
                raise RuntimeError(f'Server returned error: {error_desc!r}')
            else:
                raise RuntimeError(
                    f'Server returned error code {response.status_code}')

        data = json.loads(response.content)['response']['data']
        for key in ['fpga_note', 'fv_note', 't-sub-l']:
            try:
                del data[key]
            except KeyError:
                pass
        return cls(**data)
