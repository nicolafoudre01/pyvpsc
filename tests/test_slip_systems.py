import numpy as np
import pytest
import pyvpsc as pv

ALL_HCP = {"basal": 1, "prismatic": 1, "pyr_a": 1, "pyr_ca_I": 1, "pyr_ca_II": 1}


@pytest.mark.parametrize("crystal,ca,counts", [
    ("fcc", None, {"111<110>": 12}),
    ("bcc", None, {"110<111>": 12, "112<111>": 12, "123<111>": 24}),
    ("hcp", 1.587, {"basal": 3, "prismatic": 3, "pyr_a": 6, "pyr_ca_I": 12, "pyr_ca_II": 6}),
    ("hcp", 1.856, {"basal": 3, "prismatic": 3, "pyr_a": 6, "pyr_ca_I": 12, "pyr_ca_II": 6}),
])
def test_counts_and_geometry(crystal, ca, counts):
    lib = pv.library(crystal, ca)
    for fam, n in counts.items():
        B, N = lib[fam]
        assert len(B) == n, fam
        assert np.abs(np.linalg.norm(B, axis=1) - 1).max() < 1e-12
        assert np.abs(np.linalg.norm(N, axis=1) - 1).max() < 1e-12
        assert np.abs(np.sum(B * N, axis=1)).max() < 1e-9                       # b . n = 0
        S = np.array([pv.sym3_to_mandel(0.5 * (np.outer(b, n_) + np.outer(n_, b))) for b, n_ in zip(B, N)])
        G = np.abs(S @ S.T) - 0.5 * np.eye(len(S))                              # |m_i : m_j| = 1/2 iff same system
        assert G.max() < 0.5 - 1e-9                                             # all systems distinct


def test_hcp_specific_geometry():
    lib = pv.library("hcp", 1.587)
    Bb, Nb = lib["basal"]
    assert np.allclose(np.abs(Nb), [[0, 0, 1]] * 3) and np.abs(Bb[:, 2]).max() < 1e-12     # basal: n // c, b in plane
    Bp, Np = lib["prismatic"]
    assert np.abs(Np[:, 2]).max() < 1e-12 and np.abs(Bp[:, 2]).max() < 1e-12              # prism: all in the basal plane
    Bc, _ = lib["pyr_ca_I"]
    assert np.all(np.abs(Bc[:, 2]) > 0.5)                                                  # <c+a> has a c component
    # <c+a> direction [11-23]/|.| : z-component = (c/a)/sqrt(1+(c/a)^2)
    assert np.allclose(np.abs(Bc[:, 2]), 1.587 / np.sqrt(1 + 1.587 ** 2))


def test_independent_modes():
    assert pv.build_slip_systems("hcp", {"basal": 1, "prismatic": 1}, 1.587).n_independent() == 4
    assert pv.build_slip_systems("hcp", ALL_HCP, 1.587).n_independent() == 5
    assert pv.build_slip_systems("fcc", {"111<110>": 1}).n_independent() == 5
    assert pv.build_slip_systems("bcc", {"110<111>": 1}).n_independent() == 5


def test_schmid_tensors_and_rotation():
    sl = pv.build_slip_systems("hcp", ALL_HCP, 1.587)
    S = sl.schmid()
    assert np.abs(S[:, :3].sum(axis=1)).max() < 1e-12                            # traceless
    assert np.abs(np.sum(S * S, axis=1) - 0.5).max() < 1e-12                     # m:m = 1/2
    R = pv.read_texture("textures/SOR_random_500.csv").R[7]
    ref = np.array([pv.sym3_to_mandel(R @ pv.mandel_to_sym3(m) @ R.T) for m in S])
    assert np.abs(sl.schmid(R) - ref).max() < 1e-12


def test_crss_assignment_and_errors():
    sl = pv.build_slip_systems("hcp", {"basal": 1.0, "pyr_ca_I": 4.0}, 1.587)
    assert len(sl) == 15 and set(sl.tau_c[:3]) == {1.0} and set(sl.tau_c[3:]) == {4.0}
    with pytest.raises(KeyError):
        pv.build_slip_systems("hcp", {"nope": 1.0}, 1.587)
    with pytest.raises(ValueError):
        pv.build_slip_systems("hcp", {"basal": 1.0})                              # c/a missing


# ---------------- labels, deactivation, per-system parameters ----------------
def _parse(label):
    """'(h k i l)[u v t w]' -> two integer tuples."""
    p, d = label.split(")[")
    return tuple(int(x) for x in p[1:].split()), tuple(int(x) for x in d[:-1].split())


def test_hcp_labels_are_miller_bravais_and_consistent():
    r = 1.587
    rec = pv.systems_records("hcp", ALL_HCP, r)
    labels = [x["label"] for x in rec]
    assert len(set(labels)) == 30 and all("?" not in lb for lb in labels)
    for x in rec:
        (h, k, i, l), (u, v, t, w) = _parse(x["label"])
        assert i == -(h + k) and t == -(u + v)                                   # 4-index rule
        n = np.array([h, (h + 2 * k) / np.sqrt(3), l / r]); b = np.array(
            [u - 0.5 * v - 0.5 * t, (np.sqrt(3) / 2) * (v - t), w * r])
        assert abs(n @ b) < 1e-9 * np.linalg.norm(n) * np.linalg.norm(b)        # label rebuilds a b . n = 0 system
        assert np.allclose(np.abs(n / np.linalg.norm(n)), np.abs(x["n"])) or \
            np.allclose(np.abs(n / np.linalg.norm(n)), np.abs(-x["n"]))
    assert {x["label"].split("[")[0] for x in rec if x["family"] == "basal"} == {"(0 0 0 1)"}
    assert {x["label"].split("[")[0] for x in rec if x["family"] == "pyr_ca_II"} <= \
        {f"({a})" for a in ("1 1 -2 2", "1 -2 1 -2", "2 -1 -1 -2", "1 1 -2 -2", "1 -2 1 2", "2 -1 -1 2")}


def test_cubic_labels():
    rec = pv.systems_records("fcc", {"111<110>": 1.0})
    assert all(r["label"].count("1") >= 2 for r in rec) and len({r["label"] for r in rec}) == 12


def test_family_parameters_and_overrides():
    fam = {"basal": dict(tau_c=1.0, gamma0=2.0), "prismatic": dict(tau_c=1.5), "pyr_a": 3.0}
    ov = {("basal", 2): dict(active=False), ("prismatic", 3): dict(tau_c=9.0, gamma0=0.1)}
    sl = pv.build_slip_systems("hcp", fam, 1.587, ov, default_gamma0=0.5)
    assert len(sl) == 3 - 1 + 3 + 6 and len(sl.records) == 12                    # 1 system removed, kept in records
    assert [(f, i) for f, i in zip(sl.family, sl.index) if f == "basal"] == [("basal", 1), ("basal", 3)]
    pri = {i: (t, g) for f, i, t, g in zip(sl.family, sl.index, sl.tau_c, sl.gamma0) if f == "prismatic"}
    assert pri == {1: (1.5, 0.5), 2: (1.5, 0.5), 3: (9.0, 0.1)}                  # default gamma0 + override
    assert all(g == 2.0 for f, g in zip(sl.family, sl.gamma0) if f == "basal")
    assert all(g == 0.5 and t == 3.0 for f, g, t in zip(sl.family, sl.gamma0, sl.tau_c) if f == "pyr_a")
    off = [r for r in sl.records if not r["active"]]
    assert len(off) == 1 and (off[0]["family"], off[0]["index"]) == ("basal", 2)


def test_deactivated_system_is_really_the_one_chosen():
    full = pv.build_slip_systems("hcp", ALL_HCP, 1.587)
    cut = pv.build_slip_systems("hcp", ALL_HCP, 1.587, {("pyr_ca_I", 5): dict(active=False)})
    rec = [r for r in full.records if (r["family"], r["index"]) == ("pyr_ca_I", 5)][0]
    assert len(cut) == len(full) - 1
    gone = [i for i in range(len(full)) if not any(np.allclose(full.B[i], b) and np.allclose(full.N[i], n)
                                                  for b, n in zip(cut.B, cut.N))]
    assert len(gone) == 1 and np.allclose(full.B[gone[0]], rec["b"]) and full.label[gone[0]] == rec["label"]


def test_family_switched_off_and_validation():
    fam = {"basal": 1.0, "prismatic": dict(tau_c=1.0, active=False), "pyr_ca_I": 3.0}
    sl = pv.build_slip_systems("hcp", fam, 1.587)
    assert "prismatic" not in sl.family and len(sl) == 15
    for bad_ov in ({("basal", 0): {}}, {("basal", 4): {}}, {("pyr_a", 1): {}},
                   {("basal", 1): dict(foo=1)}):
        with pytest.raises(KeyError):
            pv.build_slip_systems("hcp", fam, 1.587, bad_ov)
    with pytest.raises(KeyError):
        pv.build_slip_systems("hcp", {"basal": dict(gamma0=1.0)}, 1.587)         # tau_c missing
    with pytest.raises(KeyError):
        pv.build_slip_systems("hcp", {"basal": dict(tau_c=1.0, bar=2)}, 1.587)   # unknown parameter
    with pytest.raises(ValueError):
        pv.build_slip_systems("hcp", {"basal": dict(tau_c=1.0, active=False)}, 1.587)   # nothing active


def test_deactivating_reduces_independent_modes():
    """<a> slip (basal, prismatic, pyramidal <a>) gives only 4 modes: the 5th needs a <c+a> system."""
    def rank(ov):
        return pv.build_slip_systems("hcp", ALL_HCP, 1.587, ov).n_independent()
    off = lambda fam, n: {(fam, j): dict(active=False) for j in range(1, n + 1)}
    assert rank({}) == 5
    assert rank({**off("pyr_ca_I", 12), **off("pyr_ca_II", 6)}) == 4        # all <c+a> off: <a> slip only
    assert rank(off("pyr_ca_I", 12)) == 5                                   # pyr_ca_II alone is enough
    assert rank({**off("pyr_ca_I", 12), **{("pyr_ca_II", j): dict(active=False) for j in range(2, 7)}}) == 5   # ONE <c+a> system
    assert rank({**off("basal", 3), **off("prismatic", 3), **off("pyr_a", 6)}) == 5   # <c+a> alone


def test_systems_table_lists_all_including_unlisted():
    rec = pv.systems_records("hcp", {"basal": 1.0}, 1.587, include_unlisted=True)
    assert len(rec) == 30 and sum(r["active"] for r in rec) == 3
    txt = pv.systems_table(rec)
    assert "(0 0 0 1)[2 -1 -1 0]" in txt and txt.count("\n") == 30
