"""Gene combinations: additive composition and joint decomposition (closed form, no gradient training)."""
from __future__ import annotations

import numpy as np


def additive(effects: dict, pair) -> np.ndarray:
    """Delta_hat(A+B) = Delta_A + Delta_B, with per-gene effects from measurement or from Pertrieve."""
    a, b = pair
    return effects[a] + effects[b]


def decompose(rows, lam: float = 1e-2):
    """Per-gene effects d_g fitted jointly to training singles and doubles by ridge least squares,
        min sum_A ||Delta_A - d_A||^2 + sum_(A,B) ||Delta_(A+B) - d_A - d_B||^2 + lam sum_g ||d_g||^2.
    rows: list of (genes, delta) with genes a list of one or two gene names. Returns (genes, [n_genes, G])."""
    gl = sorted({g for gs, _ in rows for g in gs})
    gi = {g: i for i, g in enumerate(gl)}
    X = np.zeros((len(rows), len(gl)))
    for r, (gs, _) in enumerate(rows):
        for g in gs:
            X[r, gi[g]] = 1
    Y = np.stack([d for _, d in rows])
    return gl, np.linalg.solve(X.T @ X + lam * np.eye(len(gl)), X.T @ Y)
