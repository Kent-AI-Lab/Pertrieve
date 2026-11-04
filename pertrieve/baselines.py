"""Closed-form literature baselines on gene representations (no iterative training).

  ridge   GenePert (Chen & Zou 2024): ridge regression representation -> response change, with intercept.
  ae      Ahlmann-Eltze, Huber & Anders (2025) linear model Y ~ G W P^T + b: G = top-K principal axes of the
          training responses (readout genes), P = top-K principal components of the representation (fitted on
          the training genes, queries projected), W by ridge, b = mean training response.
Queries without a representation receive the intercept (mean training response). Each function returns one
prediction per hyper-parameter value so that the setting can be chosen on validation genes.
"""
from __future__ import annotations

import numpy as np

RIDGE_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
AE_KS = (10, 50)
AE_LAMBDAS = (1e-1, 1.0, 10.0)


def combine(blocks: list[np.ndarray]) -> np.ndarray:
    """Concatenation of several L2-normalized representations, scaled by 1/sqrt(#blocks) (so rows stay unit
    norm). Genes missing any block get a zero row (no representation)."""
    X = np.hstack(blocks) / np.sqrt(len(blocks))
    X[~np.all([b.any(1) for b in blocks], 0)] = 0
    return X


def ridge(Xtr: np.ndarray, Ytr: np.ndarray, Xq: np.ndarray, lambdas=RIDGE_LAMBDAS) -> dict:
    """{lambda: [n_queries, G]}. Fitted on the training genes that have a representation (dual form); the
    intercept is the mean response of all training genes."""
    ok = Xtr.any(1)
    xm, ym = Xtr[ok].mean(0), Ytr.mean(0)
    Xc, Yc = Xtr[ok] - xm, Ytr[ok] - ym
    Kxx, Kq = Xc @ Xc.T, (Xq - xm) @ Xc.T
    out = {}
    for lam in lambdas:
        P = ym + Kq @ np.linalg.solve(Kxx + lam * np.eye(len(Kxx)), Yc)
        P[~Xq.any(1)] = ym
        out[lam] = P
    return out


def ahlmann_eltze(Xtr: np.ndarray, Ytr: np.ndarray, Xq: np.ndarray, ks=AE_KS, lambdas=AE_LAMBDAS) -> dict:
    """{(K, lambda): [n_queries, G]}."""
    b = Ytr.mean(0); Yc = Ytr - b
    ok = Xtr.any(1)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    xm = Xtr[ok].mean(0)
    _, _, Ex = np.linalg.svd(Xtr[ok] - xm, full_matrices=False)
    out = {}
    for K in ks:
        G = Vt[:K].T
        Ptr, Pq = (Xtr[ok] - xm) @ Ex[:K].T, (Xq - xm) @ Ex[:K].T
        Z = Yc[ok] @ G
        for lam in lambdas:
            W = np.linalg.solve(Ptr.T @ Ptr + lam * np.eye(K), Ptr.T @ Z)
            P = b + (Pq @ W) @ G.T
            P[~Xq.any(1)] = b
            out[(K, lam)] = P
    return out
