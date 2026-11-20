"""
Slip-system libraries (FCC, BCC, HCP) and the active set chosen in parameters.py.

Crystal frame (Cartesian):
    cubic : axes of the cube.
    HCP   : z // [0001] (c),  x // [2-1-10] (a1),  y = z x x // [01-10].   c/a given.
A slip system is a unit slip direction b and a unit plane normal n (b . n = 0).
Systems are bidirectional (sign of the slip follows the resolved shear), so (b, n) ~ (+-b, +-n).
Schmid tensor m = sym(b x n), stored as a Mandel vector (6,).

Every individual system has its own parameters: active (True/False), critical shear stress tau_c and
reference slip rate gamma0.  They are set per family and can be overridden system by system
(system = (family, index within the family, starting at 1), see systems_table / `python run.py --systems`).
"""
from dataclasses import dataclass, field
from functools import reduce
from itertools import permutations, product
from math import gcd

import numpy as np

S2 = np.sqrt(2.0)


def sym3_to_mandel(A):
    return np.array([A[0, 0], A[1, 1], A[2, 2], S2 * A[0, 1], S2 * A[0, 2], S2 * A[1, 2]])


def mandel_to_sym3(v):
    return np.array([[v[0], v[3] / S2, v[4] / S2],
                     [v[3] / S2, v[1], v[5] / S2],
                     [v[4] / S2, v[5] / S2, v[2]]])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _unique_up_to_sign(vectors, tol=1e-9):
    out = []
    for v in vectors:
        if not any(np.allclose(v, w, atol=tol) or np.allclose(v, -w, atol=tol) for w in out):
            out.append(v)
    return out


def _systems_from(planes, dirs):
    """All (b, n) with b . n = 0 for the given lists of planes and directions."""
    B, N = [], []
    for n in planes:
        for b in dirs:
            if abs(b @ n) < 1e-9:
                B.append(b); N.append(n)
    return np.array(B), np.array(N)


# ----------------------------------------------------------------------
# Cubic
# ----------------------------------------------------------------------
def _cubic_family(base):
    """Unit vectors of the cubic family <base> (all permutations and signs, up to sign)."""
    vs = [np.array([s * x for s, x in zip(sg, p)], float) for p in set(permutations(base))
          for sg in product((1, -1), repeat=3)]
    return [_unit(v) for v in _unique_up_to_sign(vs)]


def fcc_library():
    return {"111<110>": _systems_from(_cubic_family((1, 1, 1)), _cubic_family((1, 1, 0)))}


def bcc_library():
    d111 = _cubic_family((1, 1, 1))
    return {"110<111>": _systems_from(_cubic_family((1, 1, 0)), d111),
            "112<111>": _systems_from(_cubic_family((1, 1, 2)), d111),
            "123<111>": _systems_from(_cubic_family((1, 2, 3)), d111)}


# ----------------------------------------------------------------------
# Hexagonal
# ----------------------------------------------------------------------
def _hcp_operations():
    """The 12 proper rotations of the hexagonal point group 6/mmm (D6)."""
    ops = []
    for k in range(6):                                   # rotations about c
        t = k * np.pi / 3
        ops.append(np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]]))
    for j in range(6):                                   # pi rotations about axes in the basal plane
        t = j * np.pi / 6
        u = np.array([np.cos(t), np.sin(t), 0.0])
        ops.append(2.0 * np.outer(u, u) - np.eye(3))
    return ops


def hcp_plane_normal(h, k, i, l, c_over_a):
    """Unit normal of the plane (h k i l):  n ~ (h, (h + 2k)/sqrt(3), l / (c/a))."""
    return _unit([h, (h + 2 * k) / np.sqrt(3.0), l / c_over_a])


def hcp_direction(u, v, t, w, c_over_a):
    """Unit vector of the direction [u v t w] = u a1 + v a2 + t a3 + w c."""
    a1 = np.array([1.0, 0, 0])
    a2 = np.array([-0.5, np.sqrt(3) / 2, 0])
    a3 = np.array([-0.5, -np.sqrt(3) / 2, 0])
    c = np.array([0, 0, c_over_a])
    return _unit(u * a1 + v * a2 + t * a3 + w * c)


def hcp_library(c_over_a):
    ops = _hcp_operations()
    orbit = lambda v: _unique_up_to_sign([R @ v for R in ops])
    r = c_over_a
    a_dirs = orbit(hcp_direction(1, 1, -2, 0, r))                 # <11-20>
    ca_dirs = orbit(hcp_direction(1, 1, -2, 3, r))                # <11-23>
    basal = orbit(hcp_plane_normal(0, 0, 0, 1, r))                # {0001}
    prism = orbit(hcp_plane_normal(1, 0, -1, 0, r))               # {10-10}
    pyr1 = orbit(hcp_plane_normal(1, 0, -1, 1, r))                # {10-11}
    pyr2 = orbit(hcp_plane_normal(1, 1, -2, 2, r))                # {11-22}
    return {"basal": _systems_from(basal, a_dirs),                # {0001}<11-20>
            "prismatic": _systems_from(prism, a_dirs),            # {10-10}<11-20>
            "pyr_a": _systems_from(pyr1, a_dirs),                 # {10-11}<11-20>
            "pyr_ca_I": _systems_from(pyr1, ca_dirs),             # {10-11}<11-23>
            "pyr_ca_II": _systems_from(pyr2, ca_dirs)}            # {11-22}<11-23>


def library(crystal, c_over_a=None):
    """Dict family -> (B (k,3), N (k,3)) for 'fcc', 'bcc' or 'hcp'."""
    if crystal == "fcc":
        return fcc_library()
    if crystal == "bcc":
        return bcc_library()
    if crystal == "hcp":
        if c_over_a is None:
            raise ValueError("hcp needs c_over_a")
        return hcp_library(c_over_a)
    raise ValueError(f"unknown crystal '{crystal}' (fcc, bcc, hcp)")


# ----------------------------------------------------------------------
# Miller / Miller-Bravais labels
# ----------------------------------------------------------------------
def _integerize(v, tol=1e-6):
    """Smallest integer vector proportional to v (first non-zero component positive), or None."""
    v = np.asarray(v, float)
    v = v / np.abs(v[np.abs(v) > 1e-9]).min()
    for k in (1, 2, 3, 4, 5, 6, 8, 9, 10, 12):
        w = v * k
        if np.allclose(w, np.round(w), atol=tol * k):
            w = np.round(w).astype(int)
            w = w // reduce(gcd, [abs(int(x)) for x in w if x != 0])
            return w if w[np.nonzero(w)[0][0]] > 0 else -w
    return None


def _fmt(idx):
    return " ".join(str(int(x)) for x in idx) if idx is not None else "?"


def system_label(crystal, b, n, c_over_a=None):
    """'(h k l)[u v w]' (cubic) or '(h k i l)[u v t w]' (hexagonal, Miller-Bravais)."""
    if crystal != "hcp":
        return f"({_fmt(_integerize(n))})[{_fmt(_integerize(b))}]"
    r = c_over_a
    h, k, l = _integerize([n[0], (np.sqrt(3) * n[1] - n[0]) / 2, n[2] * r])
    V = 2 * b[1] / np.sqrt(3)
    U = b[0] + b[1] / np.sqrt(3)
    u, v = (2 * U - V) / 3, (2 * V - U) / 3
    d = _integerize([u, v, -(u + v), b[2] / r])
    return f"({h} {k} {-(h + k)} {l})[{_fmt(d)}]"


# ----------------------------------------------------------------------
# The systems: parameters per family, overrides per system
# ----------------------------------------------------------------------
_ALLOWED = {"tau_c", "gamma0", "active"}


def _family_spec(spec, default_gamma0):
    """Family parameters: a number (= tau_c) or a dict(tau_c=..., gamma0=..., active=...)."""
    if not isinstance(spec, dict):
        spec = {"tau_c": float(spec)}
    bad = set(spec) - _ALLOWED
    if bad:
        raise KeyError(f"unknown parameter(s) {sorted(bad)}; allowed: {sorted(_ALLOWED)}")
    if "tau_c" not in spec:
        raise KeyError("each family needs tau_c")
    return {"tau_c": float(spec["tau_c"]), "gamma0": float(spec.get("gamma0", default_gamma0)),
            "active": bool(spec.get("active", True))}


def systems_records(crystal, families, c_over_a=None, overrides=None, default_gamma0=1.0,
                    include_unlisted=False):
    """
    One record per individual system: family, index (from 1), label, b, n, tau_c, gamma0, active.
    families  : {family: tau_c or dict(tau_c, gamma0, active)}
    overrides : {(family, index): dict(tau_c=..., gamma0=..., active=...)}
    include_unlisted : also list the systems of the families not in `families` (shown inactive).
    """
    lib = library(crystal, c_over_a)
    overrides = dict(overrides or {})
    for fam in families:
        if fam not in lib:
            raise KeyError(f"family '{fam}' unknown for {crystal}; available: {list(lib)}")
    for (fam, idx), ov in overrides.items():
        if fam not in families:
            raise KeyError(f"override for ({fam!r}, {idx}): family '{fam}' is not in the families dict")
        if not 1 <= idx <= len(lib[fam][0]):
            raise KeyError(f"override for ({fam!r}, {idx}): '{fam}' has {len(lib[fam][0])} systems (index from 1)")
        if set(ov) - _ALLOWED:
            raise KeyError(f"override for ({fam!r}, {idx}): unknown parameter(s) {sorted(set(ov) - _ALLOWED)}")
    records = []
    for fam, (B, N) in lib.items():
        listed = fam in families
        if not listed and not include_unlisted:
            continue
        base = _family_spec(families[fam], default_gamma0) if listed else \
            {"tau_c": float("nan"), "gamma0": default_gamma0, "active": False}
        for j in range(len(B)):
            p = dict(base)
            p.update(overrides.get((fam, j + 1), {}))
            p = {"tau_c": float(p["tau_c"]), "gamma0": float(p["gamma0"]), "active": bool(p["active"])}
            records.append(dict(family=fam, index=j + 1, label=system_label(crystal, B[j], N[j], c_over_a),
                                b=B[j], n=N[j], **p))
    return records


def systems_table(records):
    """Printable table of the records."""
    lines = [f"{'family':10s} {'#':>3s}  {'system':28s} {'active':>6s} {'tau_c':>8s} {'gamma0':>8s}"]
    for r in records:
        tc = "-" if np.isnan(r["tau_c"]) else f"{r['tau_c']:g}"
        lines.append(f"{r['family']:10s} {r['index']:3d}  {r['label']:28s} {str(r['active']):>6s} "
                     f"{tc:>8s} {r['gamma0']:8g}")
    return "\n".join(lines)


@dataclass
class SlipSystems:
    """The ACTIVE systems (arrays of length Ns) + the full table of records (incl. deactivated ones)."""
    B: np.ndarray                  # (Ns, 3) slip directions (crystal frame)
    N: np.ndarray                  # (Ns, 3) plane normals   (crystal frame)
    tau_c: np.ndarray              # (Ns,)   critical resolved shear stress of each system
    gamma0: np.ndarray             # (Ns,)   reference slip rate of each system
    family: list = field(default_factory=list)     # (Ns,) family name
    label: list = field(default_factory=list)      # (Ns,) Miller(-Bravais) label
    index: list = field(default_factory=list)      # (Ns,) index of the system within its family (from 1)
    records: list = field(default_factory=list)    # all systems of the listed families, active or not

    def __len__(self):
        return len(self.B)

    def schmid(self, R=None):
        """(Ns, 6) Schmid tensors in the SAMPLE frame. R: crystal -> sample (identity if None)."""
        B, N = (self.B, self.N) if R is None else (self.B @ R.T, self.N @ R.T)
        return np.array([sym3_to_mandel(0.5 * (np.outer(b, n) + np.outer(n, b))) for b, n in zip(B, N)])

    def n_independent(self):
        """Number of independent strain-rate modes the active set can produce (5 are needed)."""
        sv = np.linalg.svd(self.schmid(), compute_uv=False)
        return int(np.sum(sv > 1e-9 * sv[0]))


def build_slip_systems(crystal, families, c_over_a=None, overrides=None, default_gamma0=1.0):
    """
    Active systems from the families (and per-system overrides), see systems_records.
    Deactivated systems (active=False) are removed from the arrays but kept in `.records`.
    """
    records = systems_records(crystal, families, c_over_a, overrides, default_gamma0)
    act = [r for r in records if r["active"]]
    if not act:
        raise ValueError("no active slip system")
    return SlipSystems(B=np.array([r["b"] for r in act]), N=np.array([r["n"] for r in act]),
                       tau_c=np.array([r["tau_c"] for r in act]), gamma0=np.array([r["gamma0"] for r in act]),
                       family=[r["family"] for r in act], label=[r["label"] for r in act],
                       index=[r["index"] for r in act], records=records)
