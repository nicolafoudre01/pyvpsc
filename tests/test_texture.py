from pathlib import Path

import numpy as np
import pytest
import pyvpsc as pv

TEX = Path(__file__).resolve().parent.parent / "textures" / "SOR_random_500.csv"


def test_read_real_file():
    t = pv.read_texture(TEX)
    assert len(t) == 500 and t.shape == (1.0, 1.0, 1.0)
    assert abs(t.weights.sum() - 1) < 1e-12
    assert t.euler_deg[:, 2].max() <= 60 and t.euler_deg[:, 0].max() <= 180      # hexagonal zone


def test_bunge_matrices():
    g = pv.bunge_matrix([[0, 0, 0], [90, 0, 0], [0, 90, 0], [30, 40, 50]])
    assert np.allclose(g[0], np.eye(3))
    assert np.allclose(g[1], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])                 # rotation about z by phi1
    assert np.allclose(g[2], [[1, 0, 0], [0, 0, 1], [0, -1, 0]])                 # rotation about x by Phi
    assert np.abs(np.einsum("gij,gkj->gik", g, g) - np.eye(3)).max() < 1e-14
    assert np.allclose(np.linalg.det(g), 1)


def test_R_is_transpose_of_g():
    t = pv.read_texture(TEX)
    assert np.allclose(t.R[3], t.g[3].T)


def test_parser_variants(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("2. 1. 1.\n10, 20, 30, 1\n40; 50; 60; 3\n")
    t = pv.read_texture(f)
    assert t.shape == (2.0, 1.0, 1.0) and np.allclose(t.weights, [0.25, 0.75])   # weights normalised
    f.write_text("1 1 1\n10 20 30\n40 50 60\n")                                  # no weight column
    assert np.allclose(pv.read_texture(f).weights, [0.5, 0.5])
    f.write_text("1 1 1 20 90 90\n10 20 30 1\n")
    with pytest.raises(NotImplementedError):
        pv.read_texture(f)
    f.write_text("1 1\n10 20 30 1\n")
    with pytest.raises(ValueError):
        pv.read_texture(f)
