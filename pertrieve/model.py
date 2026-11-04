"""Pertrieve inference: weight a library of measured perturbation responses and combine them.

For a query gene q and a reference perturbation k measured in the same cellular context c,

    s(q, k) = (1 / tau) * sum_m w_m cos(e_q^(m), e_k^(m)) + (U^T e_q^(1))^T (V^T e_k^(1))
    alpha_qk = softmax_k s(q, k)
    Delta_hat_q = gamma_c * sum_k alpha_qk Delta_k

where e^(m) are frozen, L2-normalized gene representations (representation 1 carries the low-rank bilinear
term) and Delta_k = mean expression of perturbation k minus the control mean of context c. Missing
representations are zero vectors, so their cosine terms vanish.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _softmax(S: np.ndarray) -> np.ndarray:
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


@dataclass
class Pertrieve:
    """A trained Pertrieve model.

    w            mixture weight per representation (length M)
    tau          temperature of the cosine mixture
    U, V         [d_1, r] projections of the first representation (bilinear term)
    scale        output scale gamma per context, e.g. {"k562": 0.90}
    representations  names of the M representations, in the order of w (the first carries U, V)
    uniform_if_missing  which missing representations make a query fall back to uniform weights (the mean
                 reference response): "first" (Replogle/Nadig models) or "all" (other benchmarks)
    """

    w: np.ndarray
    tau: float
    U: np.ndarray
    V: np.ndarray
    scale: dict = field(default_factory=dict)
    representations: tuple = ("gobert", "genept", "esm2", "scgpt")
    uniform_if_missing: str = "first"
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Pertrieve":
        """Load a model saved as <path>.npz (arrays) + <path>.json (metadata)."""
        path = Path(path)
        z = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(w=z["w"].astype(np.float64), tau=float(z["tau"]), U=z["U"].astype(np.float64),
                   V=z["V"].astype(np.float64), scale={k: float(v) for k, v in meta.pop("scale").items()},
                   representations=tuple(meta.pop("representations")),
                   uniform_if_missing=meta.pop("uniform_if_missing"), meta=meta)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), w=self.w.astype(np.float32), tau=np.float32(self.tau),
                            U=self.U.astype(np.float32), V=self.V.astype(np.float32))
        meta = dict(self.meta, scale=self.scale, representations=list(self.representations),
                    uniform_if_missing=self.uniform_if_missing)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")

    @property
    def n_parameters(self) -> int:
        return int(self.w.size + 1 + self.U.size + self.V.size)

    # ------------------------------------------------------------------ inference
    def scores(self, Q: list[np.ndarray], K: list[np.ndarray]) -> np.ndarray:
        """Unnormalized scores [n_query, n_ref]. Q, K: one [n, d_m] array per representation (rows
        L2-normalized, all-zero row = representation missing), in the order of self.representations."""
        assert len(Q) == len(K) == len(self.w)
        S = sum(self.w[m] * (Q[m] @ K[m].T) for m in range(len(self.w))) / self.tau
        S = S + (Q[0] @ self.U) @ (K[0] @ self.V).T
        miss = [~np.any(q != 0, axis=1) for q in Q]
        fallback = miss[0] if self.uniform_if_missing == "first" else np.all(miss, axis=0)
        S[fallback] = 0.0
        return S

    def weights(self, Q: list[np.ndarray], K: list[np.ndarray]) -> np.ndarray:
        """Retrieval weights alpha [n_query, n_ref]; every row sums to one."""
        return _softmax(self.scores(Q, K))

    def predict(self, Q: list[np.ndarray], K: list[np.ndarray], D: np.ndarray, context: str) -> np.ndarray:
        """Predicted response changes [n_query, G] = gamma_c * alpha @ D (D: [n_ref, G] reference responses)."""
        return self.scale[context] * self.weights(Q, K) @ D

    def fit_scale(self, K: list[np.ndarray], D: np.ndarray, context: str) -> float:
        """Set gamma_c by least squares on leave-one-out predictions of the library itself (each reference predicted
        from the others), as done after training; returns the fitted scale."""
        S = self.scores(K, K)
        np.fill_diagonal(S, -1e9)
        P = _softmax(S) @ D
        self.scale[context] = float((P * D).sum() / (P * P).sum())
        return self.scale[context]


def untrained(representations, dim: int, w=None, tau: float = 0.05, rank: int = 32, uniform_if_missing: str = "all") -> Pertrieve:
    """The model at its initialization (no training): cosine mixture with weights w (default: all weight on the first
    representation), temperature tau, bilinear term zero; scales are fitted afterward with fit_scale."""
    M = len(representations)
    w = np.eye(1, M)[0] if w is None else np.asarray(w, float)
    return Pertrieve(w=w, tau=tau, U=np.zeros((dim, rank)), V=np.zeros((dim, rank)), representations=tuple(representations),
                     uniform_if_missing=uniform_if_missing, meta=dict(name="untrained"))


def soft_knn(tau: float = 0.05, dim: int = 1024, n_rep: int = 4, rank: int = 32) -> Pertrieve:
    """The untrained model: softmax-weighted GoBERT neighbors (w = (1, 0, ..., 0), U = 0, gamma = 1)."""
    w = np.zeros(n_rep); w[0] = 1.0
    return Pertrieve(w=w, tau=tau, U=np.zeros((dim, rank)), V=np.zeros((dim, rank)), scale={},
                     uniform_if_missing="first", meta={"name": "soft GoBERT kNN"})


def knn_predict(q_emb: np.ndarray, ref_emb: np.ndarray, D: np.ndarray, k: int, fallback: np.ndarray) -> np.ndarray:
    """Hard kNN: unweighted mean response of the k references with the highest cosine similarity.
    References without a representation are never neighbors; queries without one get `fallback`. Similarities are
    rounded to 1e-10 before sorting, so ties (references with identical representations) are broken by library order
    and the neighbors do not depend on the BLAS summation order (thread count)."""
    has_ref = np.any(ref_emb != 0, axis=1)
    E, Dk = ref_emb[has_ref], D[has_ref]
    out = np.empty((len(q_emb), D.shape[1]))
    for i, e in enumerate(q_emb):
        if not np.any(e != 0):
            out[i] = fallback
            continue
        o = np.argsort(-np.round(E @ e, 10), kind="stable")[:k]
        out[i] = Dk[o].mean(0)
    return out
