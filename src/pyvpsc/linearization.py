"""
Linearization of the grain law around the current grain stress sigma:   d = M sigma + d0.

    (4)  M_sec = sum_s w_s  m_s x m_s ,    w_s = gamma0_s / tau_c_s * |tau_s / tau_c_s|^(n-1)   (secant)
    (5)  SECANT :  M = M_sec ,         d0 = 0
         AFFINE :  M = M_tg = n M_sec , d0 = (1 - n) d(sigma)       (tangent compliance + extrapolated rate)
         (d = M_sec sigma, and M_tg sigma = n d  by Euler; so  M_tg sigma + d0 = d : exact at the point)
    (6)  Hutchinson (nearly incompressible):  M <- M + J / (3 k_pen)

TANGENT (Lebensohn-Tome) is left to be defined: add it here once its equations are fixed.
Returns (M, d0, d) with d the nonlinear strain rate d(sigma).
"""
import numpy as np

from .grain_law import J_SPH, strain_rate


def linearize(sigma, S, law, kind):
    tau = S @ sigma                                              # eq (1)
    x = np.abs(tau / law.tau_c)
    w = law.gamma0 / law.tau_c * x ** (law.n - 1.0)              # secant weights, eq (4)
    M_sec = S.T @ (w[:, None] * S)                               # eq (4):  sum_s w_s m_s x m_s
    d = strain_rate(sigma, S, law)                               # eq (3)
    if kind == "secant":
        M, d0 = M_sec, np.zeros(6)                               # eq (5), secant
    elif kind == "affine":
        M, d0 = law.n * M_sec, (1.0 - law.n) * d                 # eq (5), affine
    else:
        raise NotImplementedError(f"linearization '{kind}' not available (secant, affine)")
    M = M + J_SPH / (3.0 * law.kappa)                            # eq (6)
    return M, d0, d
