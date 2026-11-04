#!/usr/bin/env python3
"""Pertrieve predictions for CellSimBench (Miller et al., Nat. Biotechnol. 2026), with the benchmark's own folds.

Writes one predictions.h5ad per (variant, seed, fold) in the format of CellSimBench's precomputed models
(one row per test condition, X = predicted mean expression, obs: condition, covariate); score them with the
unchanged CellSimBench scorer (`cellsimbench benchmark`, model type `precomputed`).
Inputs (see DATA.md): data/cellsimbench/<ds>/{data.npz, representations/*.npz}, splits/cellsimbench/<ds>/fold<k>.*
A condition's representation is the sum of its genes' L2-normalized vectors, L2-normalized. Variants: plain
(library = training conditions); for combination datasets also additive and split (pertrieve.combos semantics,
library of training singles / jointly decomposed per-gene effects), and auto = the variant with the best
validation score (Pearson delta + L2 discrimination) in each fold. Requires anndata.

  python scripts/run_cellsimbench.py --ds wessels23
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, load_representations
from pertrieve.compose import decompose
from pertrieve.data import read_genes

REPO = Path(__file__).resolve().parents[1]


def is_ctrl(c):
    return c == "control" or "ctrl" in c


def genes_of(c):
    return [g for g in c.split("+") if g and not is_ctrl(g)]


def val_score(P, R):
    Pc = P - P.mean(1, keepdims=True); Rc = R - R.mean(1, keepdims=True)
    pear = (Pc * Rc).sum(1) / np.sqrt((Pc ** 2).sum(1) * (Rc ** 2).sum(1) + 1e-30)
    d = (P ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None] - 2 * P @ R.T
    return float(pear.mean()) + float((1 - (d < np.diag(d)[:, None]).sum(1) / len(P)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", default=str(REPO / "predictions/cellsimbench"))
    a = ap.parse_args()
    import anndata as ad

    d = Path(a.data) / "cellsimbench" / a.ds
    z = np.load(d / "data.npz", allow_pickle=True)
    ctrl, cov, var_names = z["ctrl"].astype(np.float64), str(z["covariate"]), z["var_names"].astype(str)
    delta = {c: m.astype(np.float64) - ctrl for c, m in zip(z["conds"].astype(str), z["mean"])}
    combo = any(len(genes_of(c)) == 2 for c in delta)
    W = REPO / "weights/cellsimbench" / a.ds
    reps = load_representations(d / "representations", ("gobert", "genept", "esm2", "scgpt"))
    rep_index = [r.index for r in reps]

    def vecs(items, of=genes_of):
        out = []
        for r, idx in zip(reps, rep_index):
            M = np.zeros((len(items), r.dim))
            for i, c in enumerate(items):
                vs = [r.X[idx[g]] for g in of(c) if g in idx and np.any(r.X[idx[g]] != 0)]
                if vs:
                    s = np.sum(vs, 0); n = np.linalg.norm(s)
                    M[i] = s / n if n > 0 else 0
            out.append(M)
        return out

    sd = REPO / "splits/cellsimbench" / a.ds
    folds = sorted(int(f.name.split(".")[0][4:]) for f in sd.glob("fold*.test.txt"))
    seeds = sorted({int(f.stem.split("_s")[-1]) for f in (W / "plain").glob("fold*_s*.npz")})
    for s in seeds:
        for k in folds:
            tr, va, te = (read_genes(sd / f"fold{k}.{p}.txt") for p in ("train", "val", "test"))
            Rval = np.stack([delta[c] for c in va])
            D = np.stack([delta[c] for c in tr])
            m = Pertrieve.load(W / "plain" / f"fold{k}_s{s}")
            preds = {"plain": {p: m.predict(vecs(q), vecs(tr), D, a.ds) for p, q in (("val", va), ("test", te))}}
            if combo:
                singles = {genes_of(c)[0]: delta[c] for c in tr if len(genes_of(c)) == 1}
                gl, bank = decompose([(genes_of(c), delta[c]) for c in tr])
                banks = {"additive": (sorted(singles), np.stack([singles[g] for g in sorted(singles)]), singles),
                         "split": (gl, bank, dict(zip(gl, bank)))}
                one = lambda g: [g]  # noqa: E731
                for var, (kg, KD, obsd) in banks.items():
                    m = Pertrieve.load(W / var / f"fold{k}_s{s}")
                    preds[var] = {}
                    for p, cl in (("val", va), ("test", te)):
                        gq = sorted({g for c in cl for g in genes_of(c)})
                        gd = dict(zip(gq, m.predict(vecs(gq, one), vecs(kg, one), KD, a.ds)))
                        preds[var][p] = np.stack([np.sum([obsd[g] if g in obsd else gd[g] for g in genes_of(c)], 0)
                                                  if genes_of(c) else D.mean(0) for c in cl])
                best = max(preds, key=lambda v: val_score(preds[v]["val"], Rval))
                preds["auto"] = preds[best]
                print(f"{a.ds} seed {s} fold {k}: auto -> {best}", flush=True)
            for var, P in preds.items():
                o = Path(a.out) / a.ds / f"retrieval_{var}" / f"s{s}" / f"split_fold_{k}"
                o.mkdir(parents=True, exist_ok=True)
                obs = pd.DataFrame({"condition": te, "covariate": cov}, index=[f"{cov}_{c}" for c in te])
                pa = ad.AnnData(X=(ctrl[None] + P["test"]).astype(np.float32), obs=obs)
                pa.var_names = var_names
                pa.write_h5ad(o / "predictions.h5ad")
        print(f"{a.ds} seed {s}: predictions for {len(folds)} folds written", flush=True)


if __name__ == "__main__":
    main()
