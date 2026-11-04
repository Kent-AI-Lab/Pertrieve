#!/usr/bin/env python3
"""Pertrieve on the GEARS-format benchmarks released with GEARS and scGenePT (Norman, Adamson), with the
benchmarks' own simulation splits.

Inputs (see DATA.md): data/gears_format/<bench>/means.npz (per-condition mean expression; 'ctrl' = control),
data/gears_format/<bench>/representations/<name>.npz (raw gene vectors), splits/gears_format/<bench>/split<k>.*
A condition's representation is the sum of its genes' raw vectors, L2-normalized. Variants:
  plain     library = training conditions; query = the test condition
  additive  library = training singles; A+B -> Delta(A) + Delta(B), using the measured single when it is a
            training condition and Pertrieve's prediction for the gene otherwise
  split_joint / split_double   per-gene effects fitted in closed form on training singles and doubles (joint) or
            doubles only (pertrieve.compose.decompose); Pertrieve predicts genes outside the fitted library
Metric: Pearson(pred - ctrl, true - ctrl) over all genes (GEARS "pearson_delta"), per test condition and test
subgroup (unseen_single, combo_seen0/1/2). Outputs results/gears_format/<bench>/{per_condition,summary}.csv.

  python scripts/run_gears_format.py --bench gears_norman --variant additive
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve
from pertrieve.combos import RawRepresentation, pearson_delta, predict
from pertrieve.data import read_genes

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--bench", required=True, choices=["gears_norman", "scgenept_norman", "scgenept_adamson"])
    ap.add_argument("--variant", default="plain", choices=["plain", "additive", "split_joint", "split_double"])
    ap.add_argument("--out", default=str(REPO / "results/gears_format"))
    a = ap.parse_args()
    d = Path(a.data) / "gears_format" / a.bench
    z = np.load(d / "means.npz", allow_pickle=True)
    pb = dict(zip(z["conds"].astype(str), z["mean"].astype(np.float64)))
    ctrl = pb["ctrl"]; delta = {c: m - ctrl for c, m in pb.items()}
    rows = []
    wdir = REPO / "weights/gears_format" / a.bench / a.variant
    for f in sorted(wdir.glob("split*_s0.npz")):
        split = int(f.stem.split("_")[0][5:])
        sd = REPO / "splits/gears_format" / a.bench
        train, test = read_genes(sd / f"split{split}.train.txt"), [c for c in read_genes(sd / f"split{split}.test.txt") if c != "ctrl"]
        sub = pd.read_csv(sd / f"split{split}.subgroups.tsv", sep="\t").set_index("condition").subgroup
        models = [Pertrieve.load(g) for g in sorted(wdir.glob(f"split{split}_s[0-9].npz"))]
        reps = [RawRepresentation(d / "representations" / f"{n}.npz") for n in models[0].representations]
        for m in models:
            P = predict(m, reps, a.variant, train, test, delta, a.bench)
            for c, p in zip(test, P):
                rows.append(dict(split=split, seed=m.meta["seed"], condition=c, subgroup=sub.get(c, "unseen_single"),
                                 pearson_delta=pearson_delta(p, delta[c])))
    R = pd.DataFrame(rows)
    out = Path(a.out) / a.bench; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / f"per_condition_{a.variant}.csv", index=False)
    per_seed = R.groupby("seed").pearson_delta.mean()
    S = R.groupby(["seed", "subgroup"]).pearson_delta.mean().unstack().assign(all=per_seed)
    S.to_csv(out / f"summary_{a.variant}.csv")
    print(f"{a.bench} {a.variant}: mean over seeds\n{S.mean().round(4).to_string()}\nseed SD of 'all': {per_seed.std():.4f}")


if __name__ == "__main__":
    main()
