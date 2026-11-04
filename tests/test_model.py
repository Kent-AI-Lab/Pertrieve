"""Unit tests that need no data download: pytest tests/"""
from pathlib import Path

import numpy as np
import pytest

from pertrieve import Pertrieve, knn_predict, soft_knn
from pertrieve.compose import decompose
from pertrieve.metrics import per_perturbation
from pertrieve.stats import holm, paired

REPO = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def unit(n, d):
    X = rng.normal(size=(n, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def toy_model(d=8, r=2):
    return Pertrieve(w=np.array([0.8, 0.2]), tau=0.05, U=rng.normal(size=(d, r)), V=rng.normal(size=(d, r)),
                     scale={"a": 0.9}, representations=("x", "y"))


def test_weights_are_convex():
    m = toy_model()
    Q, K = [unit(5, 8), unit(5, 4)], [unit(30, 8), unit(30, 4)]
    A = m.weights(Q, K)
    assert A.shape == (5, 30) and np.all(A >= 0) and np.allclose(A.sum(1), 1)


def test_prediction_is_scaled_combination_of_references():
    m = toy_model()
    Q, K, D = [unit(3, 8), unit(3, 4)], [unit(20, 8), unit(20, 4)], rng.normal(size=(20, 50))
    assert np.allclose(m.predict(Q, K, D, "a"), 0.9 * m.weights(Q, K) @ D)


def test_missing_first_representation_gives_mean_reference():
    m = toy_model()
    Q = [np.zeros((1, 8)), unit(1, 4)]
    K, D = [unit(20, 8), unit(20, 4)], rng.normal(size=(20, 50))
    assert np.allclose(m.predict(Q, K, D, "a")[0], 0.9 * D.mean(0))


def test_soft_knn_concentrates_on_nearest_reference():
    m = soft_knn(tau=0.01, dim=8, n_rep=1, rank=2)
    K = unit(10, 8)
    A = m.weights([K[[3]]], [K])
    assert A.argmax() == 3


def test_knn_and_metrics():
    E, D = unit(10, 8), rng.normal(size=(10, 30))
    P = knn_predict(E[:2], E, D, k=1, fallback=np.zeros(30))
    assert np.allclose(P, D[:2])
    m = per_perturbation(D, D)
    assert np.allclose(m.pearson_delta, 1) and np.allclose(m.disc_cos, 1) and np.allclose(m.mse, 0)


def test_decompose_recovers_additive_effects():
    d = {g: rng.normal(size=20) for g in "ABC"}
    rows = [(["A"], d["A"]), (["B"], d["B"]), (["A", "C"], d["A"] + d["C"]), (["B", "C"], d["B"] + d["C"])]
    genes, X = decompose(rows, lam=1e-8)
    assert np.allclose(X[genes.index("C")], d["C"], atol=1e-5)


def test_stats():
    r = paired(np.ones(50) + 0.1, np.ones(50), np.arange(50) // 2)
    assert np.isclose(r["diff"], 0.1) and r["n_genes"] == 25
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


@pytest.mark.parametrize("f", sorted((REPO / "weights/heldout/gd/unseen").glob("pertrieve_s*.npz")))
def test_released_weights(f):
    m = Pertrieve.load(f)
    assert m.n_parameters == 65_541
    assert set(m.scale) == {"rpe1", "k562", "hepg2", "jurkat"}


def test_paired_lw_matches_pooled_for_one_line():
    from pertrieve.stats import paired_lw
    a, b = rng.normal(size=60), rng.normal(size=60)
    genes = np.array([f"g{i % 30}" for i in range(60)])
    one = paired_lw(a, b, genes, np.array(["l"] * 60), B=200)
    assert np.isclose(one["diff"], one["pooled"]) and one["lo"] <= one["diff"] <= one["hi"] and one["n_lines"] == 1


def test_paired_lw_weights_lines_equally():
    from pertrieve.stats import paired_lw
    # line A: 90 rows with difference +1, line B: 10 rows with difference -1 -> pooled +0.8, line-weighted 0
    d = np.r_[np.ones(90), -np.ones(10)]
    r = paired_lw(d, np.zeros(100), np.array([f"g{i}" for i in range(100)]), np.array(["A"] * 90 + ["B"] * 10), B=100)
    assert np.isclose(r["diff"], 0.0) and np.isclose(r["pooled"], 0.8) and r["per_line"] == "A:+1.0000;B:-1.0000"


def test_untrained_model_and_scale_fit():
    from pertrieve import untrained
    m = untrained(("x", "y"), dim=8, w=[1, 1])
    K, D = [unit(20, 8), unit(20, 4)], rng.normal(size=(20, 50))
    s = m.fit_scale(K, D, "a")
    S = m.scores(K, K); np.fill_diagonal(S, -1e9)
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True); P = A @ D
    assert np.isclose(s, (P * D).sum() / (P * P).sum()) and m.scale["a"] == s and m.n_parameters == 2 + 1 + 2 * 8 * 32
