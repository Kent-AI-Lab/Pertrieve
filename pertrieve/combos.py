"""Condition-level inference for benchmarks with gene combinations (GEARS, scGenePT and Scouter formats).

A condition ("A+B", "A+ctrl") is represented by the sum of its genes' raw vectors, L2-normalized. Variants:
  plain         library = training conditions; query = the test condition
  additive      library = training singles; A+B -> Delta(A) + Delta(B), measured single if it is a training
                condition, Pertrieve's prediction for the gene otherwise
  split_joint / split_double   per-gene effects fitted in closed form on training singles and doubles (joint) or
                doubles only (compose.decompose); Pertrieve predicts genes outside the fitted library
A test condition whose genes have neither a fitted/measured effect nor any representation gets the mean
training response.
"""
from __future__ import annotations

import numpy as np

from .compose import decompose


def genes_of(c):
    return [g for g in c.split("+") if g != "ctrl"]


def pearson_delta(p, t):
    a, b = p - p.mean(), t - t.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return 0.0 if d == 0 else float((a * b).sum() / d)


class RawRepresentation:
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.v = dict(zip(z["genes"].astype(str), z["X"].astype(np.float64)))
        self.dim = z["X"].shape[1]

    def has(self, c):
        return any(g in self.v for g in genes_of(c))

    def matrix(self, conds):
        out = np.zeros((len(conds), self.dim))
        for i, c in enumerate(conds):
            vs = [self.v[g] for g in genes_of(c) if g in self.v]
            if vs:
                s = np.sum(vs, axis=0); n = np.linalg.norm(s)
                out[i] = s / n if n > 0 else 0
        return out


def predict(model, reps, variant, train, test, delta, context):
    has = lambda c: any(r.has(c) for r in reps)  # noqa: E731
    Q = lambda cl: [r.matrix(cl) for r in reps]  # noqa: E731
    tr = [c for c in train if c != "ctrl"]
    if variant == "plain":
        keys = [c for c in tr if has(c)]
        return model.predict(Q(test), Q(keys), np.stack([delta[c] for c in keys]), context)
    if variant == "additive":
        keys = [c for c in tr if has(c) and len(genes_of(c)) == 1]
        D = np.stack([delta[c] for c in keys])
        obs = {genes_of(c)[0]: delta[c] for c in tr if len(genes_of(c)) == 1}
    else:
        rows = [c for c in tr if variant == "split_joint" or len(genes_of(c)) == 2]
        bank_genes, bank = decompose([(genes_of(c), delta[c]) for c in rows])
        keep = np.array([has(g) for g in bank_genes])
        keys, D = [g for g, k in zip(bank_genes, keep) if k], bank[keep]
        obs = dict(zip(bank_genes, bank))
    mean_delta = np.mean([delta[c] for c in tr], axis=0)
    gq = sorted({g for c in test for g in genes_of(c)})
    gd = dict(zip(gq, model.predict(Q(gq), Q(keys), D, context)))
    out = []
    for c in test:
        gs = genes_of(c)
        ok = all(g in obs or has(g) for g in gs)
        out.append(np.sum([obs[g] if g in obs else gd[g] for g in gs], axis=0) if ok else mean_delta)
    return np.stack(out)


