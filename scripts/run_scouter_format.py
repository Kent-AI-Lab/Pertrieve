#!/usr/bin/env python3
"""Pertrieve on Scouter's benchmarks (Adamson, Norman, Wessels) with Scouter's five splits and metrics.

Inputs (see DATA.md): data/scouter_format/<dataset>/{means.npz, degs.npz, representations/<name>.npz} and
splits/scouter_format/<dataset>/split<k>.{train,test}.txt, split<k>.subgroups.tsv. Variants as in
scripts/run_gears_format.py (plain, additive, split_joint, split_double; pertrieve.combos).
Metrics (Scouter's evaluation on mean profiles): on the top-20 non-dropout DE genes of the condition (excluding
the perturbed genes) DE NormMSE = MSE(true, pred) / MSE(true, ctrl) and DE Pearson = Pearson(true - ctrl,
pred - ctrl). Outputs results/scouter_format/<dataset>/{per_condition,summary}_<variant>.csv.

  python scripts/run_scouter_format.py --dataset norman --variant split_joint
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve
from pertrieve.combos import RawRepresentation, predict
from pertrieve.data import read_genes

REPO = Path(__file__).resolve().parents[1]


def metrics(t, p, ctrl):
    mse = lambda a, b: float(np.mean((a - b) ** 2))  # noqa: E731
    return mse(t, p) / mse(t, ctrl), float(np.corrcoef(t - ctrl, p - ctrl)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--dataset", required=True, choices=["adamson", "norman", "wessels"])
    ap.add_argument("--variant", default="plain", choices=["plain", "additive", "split_joint", "split_double"])
    ap.add_argument("--out", default=str(REPO / "results/scouter_format"))
    a = ap.parse_args()
    d = Path(a.data) / "scouter_format" / a.dataset
    z = np.load(d / "means.npz", allow_pickle=True)
    conds, var = z["conds"].astype(str), z["var_names"].astype(str)
    pb = dict(zip(conds, z["mean"].astype(np.float64)))
    degs = np.load(d / "degs.npz", allow_pickle=True)
    ctrl = pb["ctrl"]; delta = {c: m - ctrl for c, m in pb.items()}
    sd = REPO / "splits/scouter_format" / a.dataset
    wdir = REPO / "weights/scouter_format" / a.dataset / a.variant
    rows = []
    for split in range(1, 6):
        train, test = read_genes(sd / f"split{split}.train.txt"), read_genes(sd / f"split{split}.test.txt")
        sub = pd.read_csv(sd / f"split{split}.subgroups.tsv", sep="\t").set_index("condition").subgroup
        models = [Pertrieve.load(f) for f in sorted(wdir.glob(f"split{split}_s[0-9].npz"))]
        reps = [RawRepresentation(d / "representations" / f"{n}.npz") for n in models[0].representations]
        for m in models:
            P = predict(m, reps, a.variant, train, test, delta, a.dataset)
            for c, p in zip(test, P):
                de = np.setdiff1d(degs[c], np.where(np.isin(var, c.split("+"))))
                nm, pr = metrics(pb[c][de], (ctrl + p)[de], ctrl[de])
                rows.append(dict(split=split, seed=m.meta["seed"], condition=c, subgroup=sub[c], NormMSE=nm, Pearson=pr))
    R = pd.DataFrame(rows)
    out = Path(a.out) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / f"per_condition_{a.variant}.csv", index=False)
    cols = ["NormMSE", "Pearson"]
    S = pd.concat([R.groupby(["seed", "subgroup"])[cols].mean(), R.groupby("seed")[cols].mean().assign(subgroup="all").set_index("subgroup", append=True)])
    S.to_csv(out / f"summary_{a.variant}.csv")
    print(f"{a.dataset} {a.variant} (mean over seeds, 5 splits pooled)\n{S.groupby('subgroup').mean().round(4).to_string()}")


if __name__ == "__main__":
    main()
