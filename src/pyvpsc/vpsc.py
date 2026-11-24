"""
Visco-Plastic Self-Consistent, strain-rate driven

Unknowns: grain stresses sigma_g (g = 1..G), effective stiffness Lbar, macroscopic stress Sigma.
Given     : macroscopic strain rate D (6,), weights w_g, orientations, local law, grain shape a.

All tensors: Mandel 6-vectors / 6x6 matrices (MFront order, factor sqrt(2) on shear).
Per-grain quantities are computed in explicit loops so that every line can be inspected.

This version of the Visco-Plastic Self-Consistent model decouples the loop on the local phase-averaged stress fields
and the loop on the self-consistent effective stiffness.  The latter is solved by a nested loop. This could be improved in future
versions by computing the effective stifness at the same time as the local stress fields.
"""
import numpy as np
from pyhom import ptensor

from .grain_law import J_SPH, slip_rates, strain_rate
from .linearization import linearize
from .taylor import taylor


def _iterate(D, agg, law, kind, a, nq, relax, tol, max_iter, sigma0, verbose, history):
    phases, w = agg.phases, agg.fractions
    G = len(phases)
    sigma = sigma0.copy()
    Lbar = None
    Mbar = None
    for it in range(1, max_iter + 1):
        # ---- step 1: linearise every grain at its current stress (eq 4-6) and invert (eq 8)
        M, L, d0 = [], [], []
        for g, ph in enumerate(phases):
            M_g, d0_g, _ = linearize(sigma[g], ph.internal["S"], law, kind)
            M.append(M_g)
            L.append(np.linalg.inv(M_g))                         # L_g = M_g^-1
            d0.append(d0_g)
        
        # ---- step 2: self-consistent estimate of the effective stiffness
        if Lbar is None:
            Lbar = sum(w[g] * L[g] for g in range(G))            # start: Voigt average of L_g    
        for it in range(1, max_iter + 1):                        # nested loop for self-consistent Lbar
            P = ptensor(a[0], a[1], a[2], Lbar, nq)
            Q = np.linalg.inv(P) - Lbar
            L_moy = sum(f * np.linalg.inv(L + Q) for f, L in zip(w, L))
            L_new = np.linalg.inv(L_moy) - Q

            err = np.max(np.abs(L_new - Lbar)) / np.max(np.abs(L_new))
            Lbar = L_new
            if verbose:
                print(f"iter {it:3d}  err = {err:.3e}")
            if err < tol:
                break
            else:
                print("warning: self-consistent loop did not converge")
        Mbar = np.linalg.inv(Lbar)
        P = ptensor(a[0], a[1], a[2], Lbar, nq)
        Q = np.linalg.inv(P) - Lbar
        T = np.linalg.inv(Q)  # T = Q*^-1
        B = np.array([np.linalg.inv(M[g] + T)@(Mbar + T) for g in range(G)])

        # ---- step 3: calculation of the macroscopic stress-free strain rate deps0
        D0 = sum(w[g] * B[g].T @ d0[g] for g in range(G))
        
        # ---- step 4: calculation of the macroscopic stress
        Sigma = Lbar @ (D - D0)
        
        # ---- step 5: localization of the phase-averaged strain-rate and stress
        sigma_new = np.array([B[g] @ Sigma + np.linalg.inv(M[g] + T)@(D0 - d0[g]) for g in range(G)])
        d = np.array([M[g] @ (sigma_new[g]) + d0[g] for g in range(G)])

        # ---- step 6: convergence test on phase-averaged stress
        err = np.abs(sigma_new - sigma).max() / np.abs(sigma_new).max()
        sigma = relax * sigma_new + (1.0 - relax) * sigma
        if history is not None:
            history.append({"it": it, "err": err, "Sigma": Sigma.copy()})
        if verbose:
            print(f"  it {it:3d}  err(sigma_g) = {err:.2e}   Sigma_11 = {Sigma[0]: .5f}   <d>-D = {np.abs(w @ d - D).max():.1e}")
        if err < tol:
            return dict(Sigma=Sigma, Lbar=Lbar, sigma=sigma_new, d=d, d0=np.array(d0), Q=Q, n_iter=it, converged=True, relax=relax)
    return dict(Sigma=Sigma, Lbar=Lbar, sigma=sigma_new, d=d, d0=np.array(d0), Q=Q, n_iter=it, converged=False, relax=relax)


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
    for r in ((1.0, 0.5, 0.25, 0.1) if relax is None else (relax,)):
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
