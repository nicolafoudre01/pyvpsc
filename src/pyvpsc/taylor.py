import numpy as np

from .grain_law import solve_grain_stress


def taylor(D, aggregate, law):
    """
    Full-constraint (Taylor) model:  d_g = D for every grain (eq 0), grain stress from eq (2)-(3).
    Returns (Sigma, sigma_g (N,6)) with Sigma = sum_g w_g sigma_g.
    """
    sig = np.array([solve_grain_stress(D, ph.internal["S"], law) for ph in aggregate.phases])
    w = aggregate.fractions
    return w @ sig, sig
