"""
Self-consistent VPSC, strain-rate driven, written equation by equation.

Unknowns: grain stresses sigma_g (g = 1..G), effective stiffness Lbar, macroscopic stress Sigma.
Given     : macroscopic strain rate D (6,), weights w_g, orientations, local law, grain shape a.

LOOP (fixed point on sigma_g), see docs/EQUATIONS.md :
  start     sigma_g = Taylor stresses (d_g = D), Lbar = <L_g>
  step 1    linearise each grain at sigma_g:      d = M_g sigma + d0_g ,   L_g = M_g^-1        (eq 4-6, 8)
  step 2    Hill tensor of the effective medium:  P = P(Lbar, a) ,   Q* = P^-1 - Lbar          (eq 10)
  step 3    localisation:  d_g = (L_g + Q*)^-1 [ Sigma + Q* D + L_g d0_g ]                     (eq 12)
  step 4    averages:   Lam  = sum_g w_g (L_g + Q*)^-1 ,    Lbar = Lam^-1 - Q*                 (eq 13)
                        X    = sum_g w_g (L_g + Q*)^-1 L_g d0_g ,   Sigma = Lbar D - Lam^-1 X   (eq 14)
  step 5    grain stresses:  sigma_g,new = L_g (d_g - d0_g)                                   (eq 8)
  step 6    sigma_g <- r sigma_g,new + (1-r) sigma_g ;  stop when max|new-old| / max|new| < tol (eq 15)

All tensors: Mandel 6-vectors / 6x6 matrices (MFront order, factor sqrt(2) on shear).
Per-grain quantities are computed in explicit loops so that every line can be inspected.
"""
import numpy as np
from pyhom import tenseurP

from .grain_law import J_SPH, slip_rates, strain_rate
from .linearization import linearize
from .taylor import taylor


def _iterate(D, agg, law, kind, a, nq, relax, tol, max_iter, sigma0, verbose, history):
    phases, w = agg.phases, agg.fractions
    G = len(phases)
    sigma = sigma0.copy()
    Lbar = None
    for it in range(1, max_iter + 1):
        # ---- step 1: linearise every grain at its current stress (eq 4-6) and invert (eq 8)
        L, d0 = [], []
        for g, ph in enumerate(phases):
            M_g, d0_g, _ = linearize(sigma[g], ph.internal["S"], law, kind)
            L.append(np.linalg.inv(M_g))                         # L_g = M_g^-1
            d0.append(d0_g)
        if Lbar is None:
            Lbar = sum(w[g] * L[g] for g in range(G))            # start: Voigt average of L_g

        # ---- step 2: Hill tensor of the effective medium (eq 10)
        P = tenseurP(a[0], a[1], a[2], 0.5 * (Lbar + Lbar.T), nq)
        Q = np.linalg.inv(P) - Lbar                              # Q* = P^-1 - Lbar

        # ---- step 4 (needs (L_g+Q*)^-1 of step 3): averages
        Ainv = [np.linalg.inv(L[g] + Q) for g in range(G)]       # (L_g + Q*)^-1
        Lam = sum(w[g] * Ainv[g] for g in range(G))              # eq 13
        Lam_inv = np.linalg.inv(Lam)
        Lbar_new = Lam_inv - Q                                   # eq 13
        X = sum(w[g] * Ainv[g] @ L[g] @ d0[g] for g in range(G))     # eq 14
        Sigma = Lbar_new @ D - Lam_inv @ X                       # eq 14

        # ---- step 3 + 5: grain strain rates (eq 12) and stresses (eq 8)
        d = np.array([Ainv[g] @ (Sigma + Q @ D + L[g] @ d0[g]) for g in range(G)])
        sigma_new = np.array([L[g] @ (d[g] - d0[g]) for g in range(G)])

        # ---- step 6: convergence (eq 15)
        err = np.abs(sigma_new - sigma).max() / np.abs(sigma_new).max()
        dL = np.abs(Lbar_new - Lbar).max() / np.abs(Lbar_new).max()
        Lbar = Lbar_new
        sigma = relax * sigma_new + (1.0 - relax) * sigma
        if history is not None:
            history.append({"it": it, "err": err, "dLbar": dL, "Sigma": Sigma.copy()})
        if verbose:
            print(f"  it {it:3d}  err(sigma_g) = {err:.2e}   err(Lbar) = {dL:.2e}   "
                  f"Sigma_11 = {Sigma[0]: .5f}   <d>-D = {np.abs(w @ d - D).max():.1e}")
        if err < tol:
            return dict(Sigma=Sigma, Lbar=Lbar, sigma=sigma_new, d=d, d0=np.array(d0), Q=Q,
                        n_iter=it, converged=True, relax=relax)
    return dict(Sigma=Sigma, Lbar=Lbar, sigma=sigma_new, d=d, d0=np.array(d0), Q=Q,
                n_iter=it, converged=False, relax=relax)


def vpsc(D, agg, law, kind="affine", a=(1.0, 1.0, 1.0), nq=16, tol=1e-9,
         max_iter=300, relax=None, verbose=False):
    """
    Self-consistent estimate for the imposed macroscopic strain rate D (traceless, Mandel 6-vector).
    agg  : pyhom Aggregate of grains (fraction = weight, orientation, internal["S"] = Schmid tensors)
    kind : 'secant' or 'affine'.   relax=None tries 0.5, 0.25, 0.1 until convergence.
    Returns dict: Sigma, Lbar, sigma (G,6), d (G,6), d0, Q, n_iter, converged, history.
    """
    Sigma_T, sigma0 = taylor(D, agg, law)                        # start: Taylor stresses (eq 0)
    history = []
    for r in ((0.5, 0.25, 0.1) if relax is None else (relax,)):
        history.clear()
        if verbose:
            print(f"[vpsc] {kind}, n = {law.n}, relaxation = {r}")
        try:
            out = _iterate(D, agg, law, kind, a, nq, r, tol, max_iter, sigma0, verbose, history)
        except np.linalg.LinAlgError:
            continue
        if out["converged"]:
            out["history"] = list(history)
            out["Sigma_taylor"] = Sigma_T
            out["kind"] = kind
            return out
    raise RuntimeError(f"vpsc: no convergence (n = {law.n}, {kind}); try a smaller n or a given relax")


def check_equations(res, D, agg, law):
    """
    A posteriori check of the equations on a converged solution (all should be ~0).
      R1  <d_g> - D                         (eq 13, mean strain rate)
      R2  d_g - [d(sigma_g) + J sigma_g/(3 k_pen)]   (grain law eq 2-3 + Hutchinson term eq 6:
                                            the linearisation is exact at the solution)
      R3  (sigma_g - Sigma) + Q* (d_g - D)  (interaction equation, eq 11)
      R4  <sigma_g> - Sigma                 (mean stress)
      R5  Lbar - [Lam^-1 - Q*]              (self-consistency of the effective stiffness, eq 13)
    Residuals are relative to |D|, |Sigma| or |Lbar| (max abs value).
    """
    w, d, sg, Q = agg.fractions, res["d"], res["sigma"], res["Q"]
    Sigma, Lbar = res["Sigma"], res["Lbar"]
    sD, sS = np.abs(D).max(), np.abs(Sigma).max()
    r1 = np.abs(w @ d - D).max() / sD
    r2 = max(np.abs(d[g] - strain_rate(sg[g], ph.internal["S"], law) - J_SPH @ sg[g] / (3.0 * law.kappa)).max()
             for g, ph in enumerate(agg.phases)) / sD
    r3 = np.abs((sg - Sigma) + (d - D) @ Q.T).max() / sS
    r4 = np.abs(w @ sg - Sigma).max() / sS
    Lg = [np.linalg.inv(linearize(sg[g], ph.internal["S"], law, res.get("kind", "affine"))[0])
          for g, ph in enumerate(agg.phases)]
    Lam = sum(w[g] * np.linalg.inv(Lg[g] + Q) for g in range(len(Lg)))
    r5 = np.abs(Lbar - (np.linalg.inv(Lam) - Q)).max() / np.abs(Lbar).max()
    return {"R1 <d>-D": r1, "R2 grain law": r2, "R3 interaction": r3, "R4 <sigma>-Sigma": r4, "R5 Lbar": r5}
