"""Build the polycrystal as a pyhom Aggregate: one Phase per grain."""
import numpy as np
from pyhom import Aggregate, Ellipsoid, Phase

from .slip_systems import mandel_to_sym3, sym3_to_mandel


def build_aggregate(texture, law):
    """
    Phase fields:  id = grain index, fraction = weight, behaviour = the local law,
                   shape = Ellipsoid(a), orientation = R (crystal -> sample),
                   internal["S"] = Schmid tensors (Ns, 6) in the sample frame.
    """
    phases = []
    for i, (wi, R) in enumerate(zip(texture.weights, texture.R)):
        ph = Phase(id=i, fraction=float(wi), behaviour=law, shape=Ellipsoid(a=texture.shape), orientation=R)
        ph.internal["S"] = law.slip.schmid(R)
        phases.append(ph)
    return Aggregate(phases)


def strain_rate_to_mandel(D3, tol=1e-12):
    """3x3 strain-rate tensor (sample axes) -> Mandel vector; must be symmetric and traceless."""
    D3 = np.asarray(D3, float)
    if not np.allclose(D3, D3.T, atol=tol):
        raise ValueError("D must be symmetric (rigid rotation is not considered)")
    if abs(np.trace(D3)) > tol * max(1.0, np.abs(D3).max()):
        raise ValueError("D must be traceless (plastic incompressibility)")
    return sym3_to_mandel(D3)


def sigma_eq(Sigma):
    """von Mises equivalent stress of a Mandel vector."""
    dev = Sigma - np.concatenate([np.full(3, Sigma[:3].sum() / 3), np.zeros(3)])
    return float(np.sqrt(1.5 * dev @ dev))


def deviator3(Sigma):
    """Deviatoric stress as a 3x3 matrix."""
    S3 = mandel_to_sym3(Sigma)
    return S3 - np.trace(S3) / 3 * np.eye(3)
