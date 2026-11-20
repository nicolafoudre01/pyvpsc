from pathlib import Path

import numpy as np
import pytest
import pyvpsc as pv
from pyhom import selfconsistent

TEX = Path(__file__).resolve().parent.parent / "textures" / "SOR_random_500.csv"
CRSS = {"basal": 1.0, "prismatic": 1.0, "pyr_a": 3.0, "pyr_ca_I": 3.0, "pyr_ca_II": 3.0}
D = pv.strain_rate_to_mandel(np.diag([1.0, -0.5, -0.5]))


def setup(n, tex=None, crss=CRSS, crystal="hcp", ca=1.587):
    tex = pv.read_texture(TEX) if tex is None else tex
    law = pv.CrystalPlasticity(pv.build_slip_systems(crystal, crss, ca), n=n)
    return tex, law, pv.build_aggregate(tex, law)


def small_texture(n_grains=60, seed=0):
    return pv.random_texture(n_grains, seed)


# ---------------- the linear limit against pyhom ----------------
def test_n1_equals_pyhom_selfconsistent_on_the_real_texture():
    tex, law, agg = setup(1.0)
    Lg = [np.linalg.inv(pv.linearize(np.ones(6), ph.internal["S"], law, "affine")[0]) for ph in agg.phases]
    Lsc = selfconsistent(agg.fractions, Lg, a=tex.shape, n=16)[0]
    for kind in ("affine", "secant"):
        r = pv.vpsc(D, agg, law, kind, nq=16)
        assert np.abs(r["Sigma"] - Lsc @ D).max() / np.abs(r["Sigma"]).max() < 1e-7


# ---------------- equations ----------------
@pytest.mark.parametrize("kind", ["affine", "secant"])
def test_equation_residuals(kind):
    tex, law, agg = setup(3.0, small_texture())
    r = pv.vpsc(D, agg, law, kind, tol=1e-10, max_iter=500)
    res = pv.check_equations(r, D, agg, law)
    assert max(res["R1 <d>-D"], res["R3 interaction"], res["R4 <sigma>-Sigma"], res["R5 Lbar"]) < 1e-9
    assert res["R2 grain law"] < 1e-8


def test_weights_are_used():
    """One grain split into two identical copies (weights 0.3 w0 and 0.7 w0) = the same polycrystal."""
    t = small_texture(20, 1)
    w = t.weights.copy()
    A = pv.Texture(t.euler_deg, w, t.shape)
    B = pv.Texture(np.vstack([t.euler_deg, t.euler_deg[:1]]),
                   np.concatenate([[0.3 * w[0]], w[1:], [0.7 * w[0]]]), t.shape)
    law = pv.CrystalPlasticity(pv.build_slip_systems("hcp", CRSS, 1.587), n=3.0)
    rA = pv.vpsc(D, pv.build_aggregate(A, law), law, "affine", tol=1e-10, max_iter=500)
    rB = pv.vpsc(D, pv.build_aggregate(B, law), law, "affine", tol=1e-10, max_iter=500)
    assert np.abs(rA["Sigma"] - rB["Sigma"]).max() / np.abs(rA["Sigma"]).max() < 1e-7


def test_non_uniform_weights_change_the_answer():
    """Sanity: the weights matter (a texture dominated by one orientation is not the uniform one)."""
    t = small_texture(20, 1)
    heavy = np.full(20, 0.01); heavy[0] = 1 - 0.01 * 19
    law = pv.CrystalPlasticity(pv.build_slip_systems("hcp", CRSS, 1.587), n=3.0)
    r1 = pv.vpsc(D, pv.build_aggregate(t, law), law, "affine")
    r2 = pv.vpsc(D, pv.build_aggregate(pv.Texture(t.euler_deg, heavy, t.shape), law), law, "affine")
    assert np.abs(r1["Sigma"] - r2["Sigma"]).max() / np.abs(r1["Sigma"]).max() > 1e-2


@pytest.mark.parametrize("kind", ["affine", "secant"])
def test_symmetry_scaling_homogeneity(kind):
    tex, law, agg = setup(3.0, small_texture())
    r = pv.vpsc(D, agg, law, kind)
    assert np.abs(pv.vpsc(-D, agg, law, kind)["Sigma"] + r["Sigma"]).max() / np.abs(r["Sigma"]).max() < 1e-6   # odd
    r2 = pv.vpsc(2 * D, agg, law, kind)                                                                      # degree 1/n
    assert np.abs(r2["Sigma"] - 2 ** (1 / 3.0) * r["Sigma"]).max() / np.abs(r["Sigma"]).max() < 1e-6
    law2 = pv.CrystalPlasticity(pv.build_slip_systems("hcp", {k: 2.0 * v for k, v in CRSS.items()}, 1.587), n=3.0)
    r3 = pv.vpsc(D, pv.build_aggregate(tex, law2), law2, kind)                                               # CRSS x2
    assert np.abs(r3["Sigma"] - 2 * r["Sigma"]).max() / np.abs(r["Sigma"]).max() < 1e-6


def test_taylor_is_an_upper_bound():
    tex, law, agg = setup(3.0, small_texture())
    sT = pv.sigma_eq(pv.taylor(D, agg, law)[0])
    for kind in ("affine", "secant"):
        assert pv.sigma_eq(pv.vpsc(D, agg, law, kind)["Sigma"]) <= sT + 1e-9


def test_strain_rate_validation():
    with pytest.raises(ValueError):
        pv.strain_rate_to_mandel([[1, 0, 0], [0, 1, 0], [0, 0, 1]])           # not traceless
    with pytest.raises(ValueError):
        pv.strain_rate_to_mandel([[0, 1, 0], [0, 0, 0], [0, 0, 0]])           # not symmetric


# ---------------- published FCC values (slow) ----------------
def test_published_fcc_n2():
    """Random FCC polycrystal vs the published figure (digitised, +-0.01): Taylor 2.335, secant 2.245, affine 2.19."""
    tex = pv.random_texture(500, seed=0)
    law = pv.CrystalPlasticity(pv.build_slip_systems("fcc", {"111<110>": 1.0}), n=2.0)
    agg = pv.build_aggregate(tex, law)
    ref = {"taylor": 2.335, "secant": 2.245, "affine": 2.19}
    got = {"taylor": pv.sigma_eq(pv.taylor(D, agg, law)[0]),
           "secant": pv.sigma_eq(pv.vpsc(D, agg, law, "secant")["Sigma"]),
           "affine": pv.sigma_eq(pv.vpsc(D, agg, law, "affine")["Sigma"])}
    for k in ref:
        assert abs(got[k] - ref[k]) / ref[k] < 2.5e-2, (k, got[k])


# ---------------- per-system parameters and deactivation in the polycrystal ----------------
FAM2 = {"basal": dict(tau_c=1.0, gamma0=1.0), "prismatic": dict(tau_c=1.0, gamma0=0.5),
        "pyr_a": dict(tau_c=3.0, gamma0=2.0), "pyr_ca_I": dict(tau_c=3.0, gamma0=1.0),
        "pyr_ca_II": dict(tau_c=3.0, gamma0=0.7)}


def test_n1_pyhom_with_per_system_gamma0_and_a_deactivated_system():
    tex = pv.read_texture(TEX)
    sl = pv.build_slip_systems("hcp", FAM2, 1.587, {("basal", 2): dict(gamma0=3.0), ("pyr_ca_I", 4): dict(active=False)})
    law = pv.CrystalPlasticity(sl, n=1.0)
    agg = pv.build_aggregate(tex, law)
    Lg = [np.linalg.inv(pv.linearize(np.ones(6), ph.internal["S"], law, "affine")[0]) for ph in agg.phases]
    Lsc = selfconsistent(agg.fractions, Lg, a=tex.shape, n=16)[0]
    r = pv.vpsc(D, agg, law, "affine", nq=16)
    assert np.abs(r["Sigma"] - Lsc @ D).max() / np.abs(r["Sigma"]).max() < 1e-7


def test_tau_c_gamma0_equivalence_in_the_polycrystal():
    """(tau_c_s, gamma0_s) -> (lam_s tau_c_s, lam_s^n gamma0_s): same slip rates, same macroscopic stress."""
    n = 3.0
    tex = small_texture(40, 2)
    a = pv.build_slip_systems("hcp", FAM2, 1.587)
    b = pv.build_slip_systems("hcp", FAM2, 1.587)
    lam = np.linspace(0.8, 1.5, len(a))
    b.tau_c, b.gamma0 = a.tau_c * lam, a.gamma0 * lam ** n
    la, lb = pv.CrystalPlasticity(a, n=n, k_pen=1e5), pv.CrystalPlasticity(b, n=n, k_pen=1e5)
    ra = pv.vpsc(D, pv.build_aggregate(tex, la), la, "affine", tol=1e-10, max_iter=500)
    rb = pv.vpsc(D, pv.build_aggregate(tex, lb), lb, "affine", tol=1e-10, max_iter=500)
    assert np.abs(ra["Sigma"] - rb["Sigma"]).max() / np.abs(ra["Sigma"]).max() < 1e-6


def test_deactivating_a_system_stiffens_the_polycrystal():
    """Removing slip systems can only make the polycrystal harder at the same strain rate."""
    tex = small_texture(60, 3)
    full = pv.build_slip_systems("hcp", FAM2, 1.587)
    cut = pv.build_slip_systems("hcp", FAM2, 1.587, {("basal", 1): dict(active=False),
                                                     ("pyr_ca_II", 3): dict(active=False)})
    out = []
    for sl in (full, cut):
        law = pv.CrystalPlasticity(sl, n=3.0)
        out.append(pv.sigma_eq(pv.vpsc(D, pv.build_aggregate(tex, law), law, "affine")["Sigma"]))
    assert out[1] > out[0]
