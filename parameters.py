"""
parameters.py -- the ONE file to edit.   Run with:   python run.py
(stress unit = unit of CRSS, e.g. MPa ; time unit = unit of GAMMA0 and D, e.g. 1/s)
"""
import numpy as np

# =============================== TEXTURE ===============================
TEXTURE_FILE = "textures/SOR_sobol_2048.csv"
# line 1: grain semi-axes a1 a2 a3 ;  next lines: phi1 Phi phi2 weight  (Bunge, degrees)

# =============================== CRYSTAL ===============================
CRYSTAL = "hcp"            # "fcc" | "bcc" | "hcp"
C_OVER_A = 1.593           # hcp only (Ti 1.587, Mg 1.624, Zr 1.593, Zn 1.856)

# ---- slip systems: one set of parameters PER FAMILY, then optional overrides PER SYSTEM ----
#   fcc : "111<110>"
#   bcc : "110<111>", "112<111>", "123<111>"
#   hcp : "basal", "prismatic", "pyr_a", "pyr_ca_I", "pyr_ca_II"
#         ({0001}<11-20>, {10-10}<11-20>, {10-11}<11-20>, {10-11}<11-23>, {11-22}<11-23>)
# Per family:  tau_c  critical resolved shear stress (required)
#              gamma0 reference slip rate            (optional, default DEFAULT_GAMMA0)
#              active False to switch the whole family off (optional, default True)
# A family that is not listed is not used.  (PLACEHOLDER values, not material data)
DEFAULT_GAMMA0 = 1.0
SLIP = {
    "basal":     dict(tau_c=1.0 , gamma0=1.0),
    "prismatic": dict(tau_c=1.0 , gamma0=1.0),
    "pyr_a":     dict(tau_c=1.0 , gamma0=0  ),
    "pyr_ca_I":  dict(tau_c=50  , gamma0=1.0),
    "pyr_ca_II": dict(tau_c=1.0 , gamma0=0  ),
}
# Per-SYSTEM overrides, key = (family, index within the family, starting at 1).
# The index and the Miller-Bravais label of every system are listed by:   python run.py --systems
# Allowed keys: tau_c, gamma0, active.   Examples:
SYSTEM_OVERRIDES = {
    #("prismatic", 2): dict(active=False),                  # switch one system off
    # ("basal", 1):     dict(tau_c=1.2, gamma0=0.5),        # one system with its own parameters
}
# CRYSTAL = "fcc";  SLIP = {"111<110>": dict(tau_c=1.0)}
# CRYSTAL = "bcc";  SLIP = {"110<111>": dict(tau_c=1.0), "112<111>": dict(tau_c=1.0)}

# ============================= LOCAL LAW ===============================
N_EXP = 5                  # stress exponent n = 1/m  (n = 1: linear viscous)
K_PEN = None               # Hutchinson penalty bulk modulus; None -> 1e4 * mean(tau_c) / mean(gamma0)

# ====================== LOADING (strain-rate driven) ====================
# Macroscopic strain-rate tensor in the SAMPLE axes (1 = x, 2 = y, 3 = z).
# Must be symmetric and traceless.  All components of D are imposed.
EPS_RATE = 1.0
D = EPS_RATE * np.array([[-0.5, 0.0, 0.0],
                         [0.0 , 1.0, 0.0],
                         [0.0 , 0.0,-0.5]])               # uniaxial tension along yy
# D = EPS_RATE * np.array([[0, .5, 0], [.5, 0, 0], [0, 0, 0]])   # simple shear (symmetric part)

# ============================ SELF-CONSISTENT ===========================
LINEARIZATION = "affine"   # "secant" | "affine"
NQ = 32                    # quadrature order of the Hill tensor integral (n x n points)
TOL = 1e-6                 # convergence on the grain stresses
MAX_ITER = 300
RELAX = None               # None: automatic (0.5, 0.25, 0.1) ; or a number in ]0, 1]
VERBOSE = True             # print every iteration

# ================================ OUTPUT ================================
OUTPUT_GRAINS_CSV = "output_grains.csv"   # per-grain results (None to skip)
PRINT_ACTIVITY_PER_SYSTEM = False         # also print the activity of every individual system
