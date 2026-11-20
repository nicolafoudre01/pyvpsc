import numpy as np
import pytest
import pyvpsc as pv
from pyvpsc.grain_law import J_SPH

# a slip set with a DIFFERENT tau_c and gamma0 in every family, plus one system overridden
FAM = {"basal": dict(tau_c=1.0, gamma0=1.0), "prismatic": dict(tau_c=1.5, gamma0=0.5),
       "pyr_a": dict(tau_c=3.0, gamma0=2.0), "pyr_ca_I": dict(tau_c=3.0, gamma0=1.0),
       "pyr_ca_II": dict(tau_c=3.5, gamma0=0.7)}
OVR = {("basal", 2): dict(gamma0=3.0, tau_c=1.2)}
SL = pv.build_slip_systems("hcp", FAM, 1.587, OVR)
R = pv.random_texture(1, seed=3).R[0]
S = SL.schmid(R)
LAW = pv.CrystalPlasticity(SL, n=8.0)
A = np.random.default_rng(1).normal(size=(3, 3)); A = 0.5 * (A + A.T); A -= np.trace(A) / 3 * np.eye(3)
D = pv.sym3_to_mandel(A)


def test_per_system_parameters_are_arrays():
    assert LAW.gamma0.shape == LAW.tau_c.shape == (30,)
    assert len(set(LAW.gamma0)) == 5 and len(set(LAW.tau_c)) == 5               # 5 distinct values (incl. override)


def test_stress_reproduces_rate_and_is_deviatoric():
    s = pv.solve_grain_stress(D, S, LAW)
    assert np.linalg.norm(pv.strain_rate(s, S, LAW) - D) / np.linalg.norm(D) < 1e-9
    assert abs(s[:3].sum()) / np.linalg.norm(s) < 1e-9


def test_homogeneity_and_oddness():
    s = pv.solve_grain_stress(D, S, LAW)
    assert np.linalg.norm(pv.solve_grain_stress(3 * D, S, LAW) - 3 ** (1 / LAW.n) * s) / np.linalg.norm(s) < 1e-8
    assert np.linalg.norm(pv.solve_grain_stress(-D, S, LAW) + s) / np.linalg.norm(s) < 1e-8


def test_scaling_all_tau_c_and_all_gamma0():
    s = pv.solve_grain_stress(D, S, LAW)
    big = {k: dict(tau_c=2.5 * v["tau_c"], gamma0=2.0 * v["gamma0"]) for k, v in FAM.items()}
    o = {("basal", 2): dict(gamma0=2.0 * 3.0, tau_c=2.5 * 1.2)}
    law2 = pv.CrystalPlasticity(pv.build_slip_systems("hcp", big, 1.587, o), n=8.0)
    assert np.linalg.norm(pv.solve_grain_stress(2 * D, S, law2) - 2.5 * s) / np.linalg.norm(s) < 1e-8


def test_tau_c_gamma0_equivalence_per_system():
    """gdot_s = gamma0_s (tau/tau_c_s)^n is unchanged by (tau_c_s, gamma0_s) -> (lam_s tau_c_s, lam_s^n gamma0_s)."""
    n = 3.0
    lam = np.linspace(0.7, 1.8, 30)
    base = pv.build_slip_systems("hcp", FAM, 1.587, OVR)
    other = pv.build_slip_systems("hcp", FAM, 1.587, OVR)
    other.tau_c = base.tau_c * lam
    other.gamma0 = base.gamma0 * lam ** n
    la = pv.CrystalPlasticity(base, n=n, k_pen=1e5)
    lb = pv.CrystalPlasticity(other, n=n, k_pen=1e5)
    sa, sb = pv.solve_grain_stress(D, S, la), pv.solve_grain_stress(D, S, lb)
    assert np.linalg.norm(sa - sb) / np.linalg.norm(sa) < 1e-8
    assert np.abs(pv.slip_rates(sa, S, la) - pv.slip_rates(sb, S, lb)).max() < 1e-8


def test_rotation_covariance():
    s = pv.solve_grain_stress(D, S, LAW)
    D_crys = pv.sym3_to_mandel(R.T @ pv.mandel_to_sym3(D) @ R)
    sI = pv.solve_grain_stress(D_crys, SL.schmid(), LAW)
    ref = pv.sym3_to_mandel(R @ pv.mandel_to_sym3(sI) @ R.T)
    assert np.linalg.norm(s - ref) / np.linalg.norm(s) < 1e-8


def test_linearizations_exact_at_the_point_and_tangent_fd():
    s = pv.solve_grain_stress(D, S, LAW)
    pen = J_SPH / (3 * LAW.kappa)
    for kind in ("secant", "affine"):
        M, d0, d = pv.linearize(s, S, LAW, kind)
        assert np.abs(M @ s + d0 - d - pen @ s).max() < 1e-9                    # M sigma + d0 = d (+ Hutchinson)
    Ms, d0s, _ = pv.linearize(s, S, LAW, "secant")
    Ma, d0a, d = pv.linearize(s, S, LAW, "affine")
    assert np.abs(d0s).max() == 0 and np.abs(d0a - (1 - LAW.n) * d).max() < 1e-12
    assert np.abs((Ma - pen) - LAW.n * (Ms - pen)).max() < 1e-10               # M_tg = n M_sec
    h, Mfd = 1e-6, np.zeros((6, 6))
    for j in range(6):
        e = np.zeros(6); e[j] = h
        Mfd[:, j] = (pv.strain_rate(s + e, S, LAW) - pv.strain_rate(s - e, S, LAW)) / (2 * h)
    assert np.abs((Ma - pen) - Mfd).max() / np.abs(Mfd).max() < 1e-6          # tangent = d d / d sigma


def test_slower_system_carries_less_slip():
    """Lowering gamma0 of one system (strain rate imposed) lowers its share of the slip."""
    def share(g0):
        sl = pv.build_slip_systems("hcp", FAM, 1.587, {("basal", 2): dict(gamma0=g0)})
        law = pv.CrystalPlasticity(sl, n=3.0)
        s = pv.solve_grain_stress(D, S, law)
        gam = np.abs(pv.slip_rates(s, S, law))
        return gam[1] / gam.sum()
    assert share(0.01) < share(1.0) < share(10.0)


def test_tangent_not_implemented():
    with pytest.raises(NotImplementedError):
        pv.linearize(np.ones(6), S, LAW, "tangent")


def test_singular_when_not_enough_systems():
    sl = pv.build_slip_systems("hcp", {"basal": 1.0, "prismatic": 1.0}, 1.587)
    with pytest.raises(RuntimeError):
        pv.solve_grain_stress(D, sl.schmid(R), pv.CrystalPlasticity(sl, n=3.0))
