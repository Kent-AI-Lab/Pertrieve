#!/usr/bin/env python3
"""Pertrieve under SLIM's official single-gene protocol (GEARS-format Replogle K562 / RPE1 essential screens,
GEARS "simulation" splits with split seeds 1-5; test = unseen single genes).

Single-context setting: the library is the training conditions of the dataset. Inputs:
  data/slim_protocol/<dataset>.npz          per-condition mean expression of the GEARS AnnData (conds, mean)
  splits/slim_protocol/<dataset>/seed<s>.{train,test}.txt
Metrics as in SLIM's evaluation, on clipped mean profiles: Pearson(pred - ctrl, real - ctrl) over all genes,
cosine discrimination within the test set, MSE. Pertrieve: mean over its three seeds per condition.
SLIM–STRING: predictions of SLIM's official pipeline (K = 10, lambda = 0.1; mean of its synthetic population per
condition), data/slim_protocol/slim_official/<dataset>_seed<s>.npz, scored with the same code (row omitted if absent).
Outputs results/slim_protocol/{per_condition,summary}.csv.

  python scripts/run_slim_protocol.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, knn_predict, load_representations
from pertrieve.data import read_genes
from pertrieve.metrics import clip_delta, per_perturbation

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("replogle_k562_essential", "replogle_rpe1_essential")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "results/slim_protocol"))
    a = ap.parse_args()
    rows = []
    for ds in DATASETS:
        z = np.load(Path(a.data) / "slim_protocol" / f"{ds}.npz", allow_pickle=True)
        idx = {c: i for i, c in enumerate(z["conds"].astype(str))}
        M = z["mean"].astype(np.float64); ctrl = M[idx["ctrl"]]
        knn_k = {int(k): int(v) for k, v in pd.read_csv(REPO / f"splits/slim_protocol/{ds}/knn_k.csv").values}
        for s in range(1, 6):
            tr, te = (read_genes(REPO / f"splits/slim_protocol/{ds}/seed{s}.{p}.txt") for p in ("train", "test"))
            D = M[[idx[f"{g}+ctrl"] for g in tr]] - ctrl; R = M[[idx[f"{g}+ctrl"] for g in te]] - ctrl
            preds = {"train mean": [np.repeat(D.mean(0)[None], len(te), 0)]}
            models = [Pertrieve.load(f) for f in sorted((REPO / "weights/slim_protocol" / ds).glob(f"split{s}_s[0-9].npz"))]
            reps = load_representations(Path(a.data) / "representations", models[0].representations)
            K, Q = [r.matrix(tr) for r in reps], [r.matrix(te) for r in reps]
            preds["Pertrieve"] = [m.predict(Q, K, D, ds) for m in models]
            preds["GoBERT kNN"] = [knn_predict(Q[0], K[0], D, knn_k[s], D.mean(0))]
            f = Path(a.data) / "slim_protocol" / "slim_official" / f"{ds}_seed{s}.npz"
            if f.exists():
                z = np.load(f, allow_pickle=True)
                assert list(z["genes"].astype(str)) == list(te)
                preds["SLIM-STRING"] = [z["pred_mean"].astype(np.float64) - ctrl]
            for name, Ps in preds.items():
                m = sum(per_perturbation(clip_delta(P, ctrl), R)[["pearson_delta", "disc_cos", "mse"]] for P in Ps) / len(Ps)
                rows.append(m.assign(dataset=ds, split_seed=s, method=name, condition=te))
    P = pd.concat(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "per_condition.csv", index=False)
    by_split = P.groupby(["dataset", "method", "split_seed"])[["pearson_delta", "disc_cos", "mse"]].mean()
    S = by_split.groupby(["dataset", "method"]).agg(["mean", "std"])
    S.columns = [f"{m}_{s}" for m, s in S.columns]
    S = S[["pearson_delta_mean", "pearson_delta_std", "disc_cos_mean", "mse_mean"]]  # the columns of Supplement Table 62
    S.to_csv(out / "summary.csv")
    pd.set_option("display.width", 200)
    print(S[["pearson_delta_mean", "pearson_delta_std", "disc_cos_mean", "mse_mean"]].round(4).to_string())


if __name__ == "__main__":
    main()
