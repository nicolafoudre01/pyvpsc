"""
pyvpsc -- viscoplastic self-consistent polycrystal model (strain-rate driven), built on pyhom.

    texture        read the texture file (Bunge Euler angles, weights, grain shape)
    slip_systems   FCC / BCC / HCP libraries, active set with critical shear stresses
    grain_law      power law, grain Newton solver
    linearization  secant, affine (explicit formulas)
    taylor, vpsc   the models (vpsc.py holds the self-consistent algorithm, equation by equation)
"""
from .texture import Texture, read_texture, random_texture, bunge_matrix
from .slip_systems import (SlipSystems, build_slip_systems, library, systems_records, systems_table,
                           system_label, sym3_to_mandel, mandel_to_sym3)
from .grain_law import CrystalPlasticity, slip_rates, strain_rate, solve_grain_stress
from .linearization import linearize
from .aggregate import build_aggregate, strain_rate_to_mandel, sigma_eq, deviator3
from .taylor import taylor
from .vpsc import vpsc, check_equations

__version__ = "0.2.0"
