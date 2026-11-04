#!/usr/bin/env python3
"""Closed-form baselines on State's few-shot splits, matched to scripts/predict_fewshot.py (same libraries, same
validation genes, same scorer).

  ridge_all4   GenePert ridge on the four gene representations (GoBERT, GenePT, ESM2, scGPT, concatenated)
  ridge_xline  ridge regression from the cross-cell-line profile (pertrieve.data.CrossLineProfile)
  ae_xline     Ahlmann-Eltze linear model with the cross-cell-line profile as the gene representation
  slim_string_sel / slim_string_default   official SLIM (`pip install` RasmussenLab/SLIM) with its STRING
               embeddings, single-context (line c's library only); only with --slim-string <h5>
Hyper-parameters are chosen per line and seed on the seeded validation genes (splits/fewshot/<line>.val_s<seed>.txt;
library without them) by Pearson delta + cosine discrimination, then refit on the full library. Writes
<out>/<model>_s<seed>_<line>.npz (scored by evaluate_fewshot.py) and the validation predictions of the chosen
setting to <out>/val/<model>_s<seed>_<line>.npz (used by fuse_fewshot.py).

  python scripts/fewshot_baselines.py [--slim-string data/gene_string_embeddings.v0.3.h5]
"""
import argparse
from pathlib import Path

import numpy as np

from pertrieve import Pseudobulk, load_representations
from pertrieve.baselines import ahlmann_eltze, combine, ridge
from pertrieve.data import CONTROL, CrossLineProfile, read_genes
from pertrieve.metrics import clip_delta, per_perturbation

LINES = ("k562", "rpe1", "hepg2", "jurkat")
SEEDS = (0, 1, 2)
SLIM_KS = (5, 10, 20, 50, 100, 200)
SLIM_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
REPO = Path(__file__).resolve().parents[1]


def slim_fitter(h5, pb, line):
    """fit(K, lam, train, queries) -> deltas, calling the unmodified slim.model.BilinearModel through a
    DatasetAdapter over the line's pseudobulk rows (one row per perturbation plus the control)."""
    import functools

    import anndata as ad
    import pandas as pd
    import slim.model
    from slim.config import SlimConfig
    from slim.datasets.base import DatasetAdapter
    from slim.model import BilinearModel

    # speed only: read the embedding file once (SLIM's loader, identical values; callers get copies)
    cached = functools.lru_cache(maxsize=None)(slim.model.load_h5_embeddings)

    def load_h5_embeddings(path, dim=None):
        g2i, emb = cached(str(path), dim)
        return dict(g2i), emb.copy()
    slim.model.load_h5_embeddings = load_h5_embeddings

    genes = pb.genes(line); mu = pb.control(line)
    obs = pd.DataFrame({"condition": genes + [CONTROL]}); obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=np.vstack([pb.mean(line, genes), mu[None]]), obs=obs)

    class Adapter(DatasetAdapter):
        condition_col, control_tag = "condition", CONTROL

        def __init__(self, train, test):
            self._train, self._test = list(train), list(test)

        def load(self):
            return self

        @property
        def adata(self):
            return adata

        def train_perts(self):
            return self._train

        def val_perts(self):
            return []

        def test_perts(self):
            return self._test

    def fit(K, lam, train, queries):
        m = BilinearModel(str(h5), SlimConfig(condition_col="condition", control_tag=CONTROL, K=K, lambda_reg=lam))
        m.fit(Adapter(train, queries))
        p = m.predict(queries)
        return np.stack([p[g] for g in queries]) - mu
    return fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--slim-string", default=None, help="SLIM's STRING embedding file (enables the SLIM baselines)")
    ap.add_argument("--out", default=str(REPO / "predictions/fewshot"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    reps = load_representations(Path(a.data) / "representations", ("gobert", "genept", "esm2", "scgpt"))
    out = Path(a.out); (out / "val").mkdir(parents=True, exist_ok=True)
    for c in LINES:
        held = set(read_genes(REPO / f"splits/fewshot/{c}.test.txt"))
        test = sorted(g for g in pb.genes(c) if g in held)
        lib = [g for g in pb.genes(c) if g not in held]
        ctrl, D = pb.control(c), pb.delta(c, lib)
        prof = CrossLineProfile(pb, c, LINES)
        feats = {"xline": (prof.matrix(lib), prof.matrix(test)),
                 "all4": (combine([r.matrix(lib) for r in reps]), combine([r.matrix(test) for r in reps]))}
        models = {"ae_xline": ("xline", ahlmann_eltze), "ridge_xline": ("xline", ridge), "ridge_all4": ("all4", ridge)}
        if a.slim_string:
            slim = slim_fitter(a.slim_string, pb, c)
        for s in SEEDS:
            val = set(read_genes(REPO / f"splits/fewshot/{c}.val_s{s}.txt"))
            vi = np.array([g in val for g in lib]); fi = ~vi
            vgenes = [g for g in lib if g in val]
            score = lambda P: (lambda m: m.pearson_delta + m.disc_cos)(per_perturbation(clip_delta(P, ctrl), D[vi]).mean())  # noqa: E731
            for name, (feat, fit) in models.items():
                Xl, Xt = feats[feat]
                V = fit(Xl[fi], D[fi], Xl[vi])
                h = max(V, key=lambda h: score(V[h]))
                np.savez_compressed(out / "val" / f"{name}_s{s}_{c}.npz", genes=np.array(vgenes), delta=V[h].astype(np.float32),
                                    ctrl=ctrl, setting=str(h))
                P = fit(Xl, D, Xt)[h]
                np.savez_compressed(out / f"{name}_s{s}_{c}.npz", genes=np.array(test), delta=P.astype(np.float32), ctrl=ctrl,
                                    setting=str(h))
                print(f"{c} {name} s{s}: setting {h}", flush=True)
            if a.slim_string:
                fl = [g for g, v in zip(lib, vi) if not v]
                sc = {(K, lam): score(slim(K, lam, fl, vgenes)) for K in SLIM_KS for lam in SLIM_LAMBDAS}
                runs = [("slim_string_sel", max(sc, key=sc.get))] + ([("slim_string_default", (10, 0.1))] if s == 0 else [])
                for name, (K, lam) in runs:
                    P = slim(K, lam, lib, test)
                    np.savez_compressed(out / f"{name}_s{s}_{c}.npz", genes=np.array(test), delta=P.astype(np.float32),
                                        ctrl=ctrl, setting=str((K, lam)))
                    print(f"{c} {name} s{s}: K={K} lambda={lam}", flush=True)


if __name__ == "__main__":
    main()
