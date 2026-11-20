"""
Grain constitutive law (rate-dependent power law) and the Newton solver for ONE grain.

    (1)  tau_s   = m_s : sigma                          resolved shear stress  (Mandel: S[s] . sigma)
    (2)  gdot_s  = gamma0_s * sign(tau_s) * |tau_s / tau_c_s|^n
    (3)  d       = sum_s gdot_s m_s                     (Mandel: S^T gdot)

S : (Ns, 6) Schmid tensors of the grain in the SAMPLE frame.   tau_c, gamma0 : (Ns,) one value per system.
Incompressibility: Hutchinson treatment, see linearization.py (eq 7).
"""
from dataclasses import dataclass

import numpy as np

from .slip_systems import SlipSystems

J_SPH = np.zeros((6, 6)); J_SPH[:3, :3] = 1.0 / 3.0       # spherical projector J (Mandel)


@dataclass
class CrystalPlasticity:
    """The local law: active slip systems (each with its tau_c and gamma0), exponent n, Hutchinson penalty."""
    slip: SlipSystems
    n: float = 5.0
    k_pen: float = None               # penalty bulk modulus; None -> 1e4 * mean(tau_c) / mean(gamma0)

    @property
    def tau_c(self):
        return self.slip.tau_c

    @property
    def gamma0(self):
        return self.slip.gamma0

    @property
    def kappa(self):
        return 1e4 * np.mean(self.tau_c) / np.mean(self.gamma0) if self.k_pen is None else self.k_pen


def resolved_shear(sigma, S):
    return S @ sigma                                                      # eq (1)


def slip_rates(sigma, S, law):
    x = resolved_shear(sigma, S) / law.tau_c
    return law.gamma0 * np.sign(x) * np.abs(x) ** law.n                   # eq (2)


def strain_rate(sigma, S, law):
    return S.T @ slip_rates(sigma, S, law)                                # eq (3)


def solve_grain_stress(d, S, law, tol=1e-12, max_iter=100):
    """
    Stress of ONE grain submitted to the traceless strain rate d:  solve  d(sigma) = d  (eq 2-3).
    Damped Newton with continuation on the exponent n. Returns sigma (6,).
    (internally dimensionless: stresses / mean(tau_c), rates / mean(gamma0))
    """
    d = np.asarray(d, float)
    tref, gref = float(np.mean(law.tau_c)), float(np.mean(law.gamma0))
    tc = law.tau_c / tref                                  # dimensionless critical stresses
    g0 = law.gamma0 / gref                                 # dimensionless reference rates
    Jk = J_SPH / (3.0 * law.kappa * gref / tref)           # compressibility term
    r = d / gref                                           # dimensionless strain rate

    def resid(s, nn):                                      # s = sigma / tref
        x = (S @ s) / tc
        return S.T @ (g0 * np.sign(x) * np.abs(x) ** nn) + Jk @ s - r

    try:
        s = np.linalg.solve((S.T * (g0 / tc)) @ S + Jk, r)     # linear start (n = 1)
    except np.linalg.LinAlgError:
        raise RuntimeError("grain problem singular: the active slip systems cannot accommodate "
                           "5 independent strain-rate modes (activate more systems)")
    ns = [v for v in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0) if v < law.n] + [float(law.n)]
    for nn in ns:
        for _ in range(max_iter):
            R = resid(s, nn)
            if np.linalg.norm(R) <= tol * np.linalg.norm(r):
                break
            x = np.abs((S @ s) / tc)
            Jac = (S.T * (g0 * nn * x ** (nn - 1.0) / tc)) @ S + Jk
            ds = -np.linalg.solve(Jac, R)
            lam, r0 = 1.0, np.linalg.norm(R)
            while lam > 1e-6 and np.linalg.norm(resid(s + lam * ds, nn)) > (1 - 1e-4 * lam) * r0:
                lam *= 0.5
            s = s + lam * ds
        else:
            raise RuntimeError(f"solve_grain_stress: no convergence (n = {nn})")
    return s * tref
