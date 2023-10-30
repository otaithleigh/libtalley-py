from enum import Enum
from functools import cached_property

import unyt

from .units import create_unit_system, get_unit_system


# ---------------------------------------
# Unit systems
# ---------------------------------------
class Units(int, Enum):
    """Enumeration for available units in SAP."""

    lb_in_F = 1
    lb_ft_F = 2
    kip_in_F = 3
    kip_ft_F = 4
    kN_mm_C = 5
    kN_m_C = 6
    kgf_mm_C = 7
    kgf_m_C = 8
    N_mm_C = 9
    N_m_C = 10
    Ton_mm_C = 11
    Ton_m_C = 12
    kN_cm_C = 13
    kgf_cm_C = 14
    N_cm_C = 15
    Ton_cm_C = 16

    @classmethod
    def get(cls, v):
        if isinstance(v, str):
            return cls[v]
        else:
            return cls(v)

    def __getitem__(self, key):
        return self.system[key]

    # ---------------------------------
    # Physical constants
    # ---------------------------------
    @cached_property
    def g0(self):
        """Standard acceleration due to gravity."""
        g0 = unyt.physical_constants.standard_gravity
        return g0.to(self.length / self.system['time'] ** 2)

    # ---------------------------------
    # Units, base and derived
    # ---------------------------------
    @cached_property
    def system(self):
        """Unit system associated with this enum."""
        return get_unit_system(self.name)

    @cached_property
    def acceleration(self) -> unyt.Unit:
        return self.length / self.time**2

    @cached_property
    def area(self) -> unyt.Unit:
        return self.length**2

    @cached_property
    def energy(self) -> unyt.Unit:
        return self.system['energy']

    @cached_property
    def force(self) -> unyt.Unit:
        return self.system['force']

    @cached_property
    def length(self) -> unyt.Unit:
        return self.system['length']

    @cached_property
    def mass(self) -> unyt.Unit:
        return self.system['mass']

    @cached_property
    def moment(self) -> unyt.Unit:
        return self.force * self.length

    @cached_property
    def rotational_stiffness(self) -> unyt.Unit:
        return self.force * self.length / unyt.radian

    @cached_property
    def shear_stiffness(self) -> unyt.Unit:
        return self.force / self.length

    @cached_property
    def stress(self) -> unyt.Unit:
        return self.system['pressure']

    @cached_property
    def time(self) -> unyt.Unit:
        return self.system['time']

    @cached_property
    def velocity(self) -> unyt.Unit:
        return self.length / self.time


lb_in_F_system = create_unit_system(
    name='lb_in_F',
    length='inch',
    mass='blob',
    time='s',
    force='lbf',
    stress='psi',
    energy='lbf*inch',
    temperature='degF',
    consistent=True,
)
lb_ft_F_system = create_unit_system(
    name='lb_ft_F',
    length='ft',
    mass='slug',
    time='s',
    force='lbf',
    stress='psf',
    energy='lbf*ft',
    temperature='degF',
    consistent=True,
)
kip_in_F_system = create_unit_system(
    name='kip_in_F',
    length='inch',
    mass='kblob',
    time='s',
    force='kip',
    stress='ksi',
    energy='kip*inch',
    temperature='degF',
    consistent=True,
)
kip_ft_F_system = create_unit_system(
    name='kip_ft_F',
    length='ft',
    mass='kslug',
    time='s',
    force='kip',
    stress='ksf',
    energy='kip*ft',
    temperature='degF',
    consistent=True,
)
kN_mm_C_system = create_unit_system(
    name='kN_mm_C',
    length='mm',
    mass='Gg',
    time='s',
    force='kN',
    stress='GPa',
    energy='J',
    temperature='degC',
    consistent=True,
)
kN_m_C_system = create_unit_system(
    name='kN_m_C',
    length='m',
    mass='Mg',
    time='s',
    force='kN',
    stress='kPa',
    energy='kJ',
    temperature='degC',
    consistent=True,
)
kgf_mm_C_system = create_unit_system(
    name='kgf_mm_C',
    length='mm',
    mass='khyl',
    time='s',
    force='kgf',
    consistent=True,
)
kgf_m_C_system = create_unit_system(
    name='kgf_m_C',
    length='m',
    mass='hyl',
    time='s',
    force='kgf',
    consistent=True,
)
N_mm_C_system = create_unit_system(
    name='N_mm_C',
    length='mm',
    mass='Mg',
    time='s',
    force='N',
    energy='mJ',
    consistent=True,
)
N_m_C_system = create_unit_system(
    name='N_m_C',
    length='m',
    mass='kg',
    time='s',
    force='N',
    energy='J',
    consistent=True,
)
Ton_mm_C_system = create_unit_system(
    name='Ton_mm_C',
    length='mm',
    mass='Mhyl',
    time='s',
    force='tonne_force',
    consistent=True,
)
Ton_m_C_system = create_unit_system(
    name='Ton_m_C',
    length='m',
    mass='khyl',
    time='s',
    force='tonne_force',
    consistent=True,
)
kN_cm_C_system = create_unit_system(
    name='kN_cm_C',
    length='cm',
    mass='10**5 * kg',
    time='s',
    force='kN',
    consistent=True,
)
kgf_cm_C_system = create_unit_system(
    name='kgf_cm_C',
    length='cm',
    mass='10**2 * hyl',
    time='s',
    force='kgf',
    consistent=True,
)
N_cm_C_system = create_unit_system(
    name='N_cm_C',
    length='cm',
    mass='10**2 * kg',
    time='s',
    force='N',
    consistent=True,
)
Ton_cm_C_system = create_unit_system(
    name='Ton_cm_C',
    length='cm',
    mass='10**5 * hyl',
    time='s',
    force='tonne_force',
    consistent=True,
)
