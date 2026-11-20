"""
Texture: list of grains = (Bunge Euler angles, weight), plus the grain shape.

File format (VPSC style, any whitespace / comma / semicolon separator):
    line 1 :  a1 a2 a3             semi-axes of the ellipsoidal grains (or 6 numbers: axes + 3 angles)
    line 2+:  phi1 Phi phi2 weight   Bunge Euler angles in DEGREES, weight of the grain

Convention (Bunge, ZXZ):  g = matrix SAMPLE -> CRYSTAL,   v_crystal = g @ v_sample
                          R = g^T = matrix CRYSTAL -> SAMPLE (used everywhere else: v_sample = R @ v_crystal)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Texture:
    euler_deg: np.ndarray        # (N, 3)  phi1, Phi, phi2 in degrees
    weights: np.ndarray          # (N,)    sum = 1
    shape: tuple = (1.0, 1.0, 1.0)

    def __len__(self):
        return len(self.weights)

    @property
    def g(self):
        """(N,3,3) Bunge matrices, sample -> crystal."""
        return bunge_matrix(self.euler_deg)

    @property
    def R(self):
        """(N,3,3) rotation matrices crystal -> sample (v_sample = R v_crystal)."""
        return np.transpose(self.g, (0, 2, 1))


def bunge_matrix(euler_deg):
    """Bunge ZXZ rotation matrices g (sample -> crystal) for angles in degrees, shape (N,3)."""
    e = np.radians(np.atleast_2d(np.asarray(euler_deg, float)))
    c1, c, c2 = np.cos(e[:, 0]), np.cos(e[:, 1]), np.cos(e[:, 2])
    s1, s, s2 = np.sin(e[:, 0]), np.sin(e[:, 1]), np.sin(e[:, 2])
    g = np.empty((len(e), 3, 3))
    g[:, 0, 0] = c1 * c2 - s1 * s2 * c
    g[:, 0, 1] = s1 * c2 + c1 * s2 * c
    g[:, 0, 2] = s2 * s
    g[:, 1, 0] = -c1 * s2 - s1 * c2 * c
    g[:, 1, 1] = -s1 * s2 + c1 * c2 * c
    g[:, 1, 2] = c2 * s
    g[:, 2, 0] = s1 * s
    g[:, 2, 1] = -c1 * s
    g[:, 2, 2] = c
    return g


def _numbers(line):
    return [float(x) for x in line.replace(",", " ").replace(";", " ").split()]


def read_texture(path):
    """Read a texture file (see module docstring)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = _numbers(lines[0])
    if len(header) not in (3, 6):
        raise ValueError(f"{path}: first line must hold 3 (axes) or 6 (axes + angles) numbers, got {header}")
    if len(header) == 6 and not np.allclose(header[3:], 90.0):
        raise NotImplementedError("rotated grain-shape axes are not implemented (angles must be 90 90 90)")
    data = np.array([_numbers(ln) for ln in lines[1:]])
    if data.shape[1] == 3:                              # no weight column: equal weights
        data = np.hstack([data, np.ones((len(data), 1))])
    if data.shape[1] != 4:
        raise ValueError(f"{path}: rows must be 'phi1 Phi phi2 [weight]'")
    w = data[:, 3] / data[:, 3].sum()
    return Texture(euler_deg=data[:, :3], weights=w, shape=tuple(header[:3]))


def random_texture(n_grains=500, seed=0):
    """Uniformly distributed random orientations (Bunge angles), equal weights, spherical grains."""
    rng = np.random.default_rng(seed)
    phi1 = rng.uniform(0, 360, n_grains)
    Phi = np.degrees(np.arccos(rng.uniform(-1, 1, n_grains)))
    phi2 = rng.uniform(0, 360, n_grains)
    return Texture(np.column_stack([phi1, Phi, phi2]), np.full(n_grains, 1.0 / n_grains))
