#!/usr/bin/env python3
"""SLIM-STRING (Table 1 / Supplementary Table 9) on the held-out-gene splits, with SLIM's released code.

SLIM's model is not re-implemented: slim.model.BilinearModel.fit / .predict are called as released
(RasmussenLab/SLIM, commit 5a7e9ade). Only the data access is adapted, through SLIM's DatasetAdapter extension
point: the adapter's AnnData holds one row per perturbation (our pseudobulk) plus the control row, which SLIM's
own pseudobulking averages back to the same per-perturbation means. Per cell line, the training perturbations
are that line's measured genes except all held-out genes (as for every other model).

  default    K = 10, lambda = 0.1 (SLIM's defaults)
  selected   (K, lambda) in {5,10,20,50,100,200} x {0.01,0.1,1,10,100}, chosen on the RPE1 queries by
             Pearson delta + cosine discrimination (the criterion of every other model). The settings selected
             for the paper are in splits/heldout/slim_string_selected.json and are used by default;
             --reselect repeats the selection.
SLIM's PCA basis uses scikit-learn's randomized SVD without a fixed random state, so unseeded runs differ
slightly (about 1e-3 relative in the predictions) and near-tied settings can swap during selection. This wrapper
seeds NumPy's global generator (--seed) before every fit so that runs repeat exactly; the paper's values come
from an unseeded run and agree to about +-0.001 in Pearson.
Writes slim_string_{default,selected}_s0_<line>.npz in the format of predict_heldout.py, scored by
scripts/evaluate_heldout.py.

Requirements: SLIM (pip install "git+https://github.com/RasmussenLab/SLIM@5a7e9ade") and its STRING gene
embeddings, data/gene_string_embeddings.v0.3.h5 in the SLIM repository (cite Hu et al. 2026 when using them).

  python scripts/baseline_slim_heldout.py --split unseen --protocol gd --string-h5 SLIM/data/gene_string_embeddings.v0.3.h5
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from slim.config import SlimConfig
from slim.datasets.base import DatasetAdapter
from slim.model import BilinearModel

from pertrieve import CONTROL, Pseudobulk, heldout_split
from pertrieve.metrics import clip_delta, per_perturbation

LINES = ("rpe1", "k562", "hepg2", "jurkat")
KS = (5, 10, 20, 50, 100, 200)
LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0)
REPO = Path(__file__).resolve().parents[1]


class PseudobulkAdapter(DatasetAdapter):
    """One cell line's pseudobulk rows as a SLIM dataset (SLIM README, "Usage B")."""
    condition_col = "condition"
    control_tag = CONTROL

    def __init__(self, M, genes, ctrl, train, test):
        obs = pd.DataFrame({"condition": list(genes) + [CONTROL]}); obs.index = obs.index.astype(str)
        self._adata = ad.AnnData(X=np.vstack([M, ctrl[None]]), obs=obs)
        self._train, self._test = list(train), list(test)

    def load(self):
        return self

    @property
    def adata(self):
        return self._adata

    def train_perts(self):
        return self._train

    def val_perts(self):
        return []

    def test_perts(self):
        return self._test

    def format_condition(self, pert):
        return pert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--split", default="unseen", choices=["unseen", "cluster", "cluster_genept"])
    ap.add_argument("--protocol", default="gd", choices=["gd", "full"])
    ap.add_argument("--string-h5", required=True, help="SLIM's data/gene_string_embeddings.v0.3.h5")
    ap.add_argument("--reselect", action="store_true", help="select (K, lambda) on RPE1 instead of using the stored setting")
    ap.add_argument("--seed", type=int, default=0, help="NumPy global seed set before every SLIM fit")
    ap.add_argument("--out", default=str(REPO / "predictions/heldout"))
    a = ap.parse_args()

    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    sp = heldout_split(REPO / "splits/heldout", a.split, a.protocol)
    out = Path(a.out) / a.protocol / a.split
    out.mkdir(parents=True, exist_ok=True)

    def fit_predict(c, K, lam):
        genes = pb.genes(c); ctrl = pb.control(c)
        lib = [g for g in genes if g not in sp["heldout"]]
        q = sorted(g for g in genes if g in sp["queries"][c])
        adapter = PseudobulkAdapter(pb.mean(c, genes), genes, ctrl, lib, q)
        model = BilinearModel(a.string_h5, SlimConfig(condition_col="condition", control_tag=CONTROL, K=K, lambda_reg=lam))
        np.random.seed(a.seed)
        model.fit(adapter)
        pred = model.predict(q)
        return q, np.stack([pred[g] for g in q]) - ctrl

    if a.reselect:
        ctrl = pb.control("rpe1")
        val = {}
        for K in KS:
            for lam in LAMBDAS:
                q, P = fit_predict("rpe1", K, lam)
                m = per_perturbation(clip_delta(P, ctrl), pb.delta("rpe1", q)).mean()
                val[f"{K}_{lam}"] = float(m.pearson_delta + m.disc_cos)
        best = max(val, key=val.get)
        (out / "slim_string_selected.selection.json").write_text(json.dumps(dict(chosen=best, rpe1_score=val), indent=1))
        K, lam = int(best.split("_")[0]), float(best.split("_")[1])
    else:
        s = json.loads((REPO / "splits/heldout/slim_string_selected.json").read_text()).get(a.protocol, {}).get(a.split)
        if s is None:  # SLIM was not run on this protocol / split for the paper
            print(f"{a.protocol}/{a.split}: no stored SLIM setting; skipped (use --reselect to run it)", flush=True)
            return
        K, lam = s["K"], s["lam"]
    for name, (k, l) in {"slim_string_default": (10, 0.1), "slim_string_selected": (K, lam)}.items():
        for c in LINES[1:]:
            q, P = fit_predict(c, k, l)
            np.savez_compressed(out / f"{name}_s0_{c}.npz", genes=np.array(q), delta=P.astype(np.float32), ctrl=pb.control(c))
        print(f"{a.protocol}/{a.split}/{name}: K={k}, lambda={l}", flush=True)


if __name__ == "__main__":
    main()
