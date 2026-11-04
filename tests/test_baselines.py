import numpy as np

from pertrieve.baselines import ahlmann_eltze, combine, ridge


def test_ridge_recovers_linear_map_and_intercept_for_missing():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)); W = rng.normal(size=(8, 5))
    Y = X @ W + 1.0
    Xq = np.vstack([rng.normal(size=(3, 8)), np.zeros((1, 8))])
    P = ridge(X, Y, Xq, lambdas=(1e-8,))[1e-8]
    np.testing.assert_allclose(P[:3], Xq[:3] @ W + 1.0, atol=1e-5)
    np.testing.assert_allclose(P[3], Y.mean(0))


def test_ahlmann_eltze_grid_and_shapes():
    rng = np.random.default_rng(1)
    X, Y, Xq = rng.normal(size=(60, 12)), rng.normal(size=(60, 20)), rng.normal(size=(4, 12))
    out = ahlmann_eltze(X, Y, Xq, ks=(3, 5), lambdas=(0.1, 1.0))
    assert set(out) == {(3, 0.1), (3, 1.0), (5, 0.1), (5, 1.0)}
    assert all(P.shape == (4, 20) for P in out.values())


def test_combine_unit_rows_and_missing_blocks():
    a = np.array([[1.0, 0.0], [0.0, 1.0]]); b = np.array([[0.0, 1.0], [0.0, 0.0]])
    X = combine([a, b])
    np.testing.assert_allclose(np.linalg.norm(X[0]), 1.0)
    assert not X[1].any()
