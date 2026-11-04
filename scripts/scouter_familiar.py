#!/usr/bin/env python3
"""Unseen singles on Scouter's Norman benchmark split by evidence source (Supplement Table 40).

A test single is "familiar" if its gene occurs in a training double of the same split (the joint decomposition can
estimate its effect from those doubles) and "unfamiliar" if the gene occurs in no training condition (its effect can
only come from retrieval). Reads the per-condition results of scripts/run_scouter_format.py --dataset norman
(every variant found) and Scouter's published per-condition results (data/scouter_format/norman/scouter_published.csv,
from Scouter's reproducibility repository, results/norman.csv, Method == "Scouter"; Pearson = 1 - "1-PCC").
Metrics: DE NormMSE / DE Pearson on the top-20 DE genes, mean over conditions (Pertrieve: seeds averaged).
Output results/scouter_format/norman/familiar.csv.

  python scripts/scouter_familiar.py
"""
import argparse
from pathlib import Path

import pandas as pd

from pertrieve.data import read_genes

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ("plain", "additive", "split_double", "split_joint")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--results", default=str(REPO / "results/scouter_format/norman"))
    a = ap.parse_args()
    sd = REPO / "splits/scouter_format/norman"
    part = {}
    for split in range(1, 6):
        train = read_genes(sd / f"split{split}.train.txt")
        in_doubles = {g for c in train for g in c.split("+") if g != "ctrl" and "ctrl" not in c.split("+")}
        sub = pd.read_csv(sd / f"split{split}.subgroups.tsv", sep="\t")
        for c in sub.condition[sub.subgroup == "single"]:
            gene = next(g for g in c.split("+") if g != "ctrl")
            part[(c, split)] = "familiar" if gene in in_doubles else "unfamiliar"
    sc = pd.read_csv(Path(a.data) / "scouter_format/norman/scouter_published.csv")
    sc = sc[sc.subgroup == "single"].assign(Pearson=lambda x: 1 - x["1-PCC"], method="Scouter (published)")
    frames = [sc[["condition", "split", "method", "NormMSE", "Pearson"]]]
    for v in VARIANTS:
        f = Path(a.results) / f"per_condition_{v}.csv"
        if f.exists():
            r = pd.read_csv(f)
            r = r[r.subgroup == "single"].groupby(["condition", "split"])[["NormMSE", "Pearson"]].mean().reset_index()
            frames.append(r.assign(method=f"Pertrieve, {v}"))
    R = pd.concat(frames, ignore_index=True)
    R["part"] = [part[(c, s)] for c, s in zip(R.condition, R.split)]
    rows = []
    for m, g in R.groupby("method", sort=False):
        g = g.set_index(["condition", "split"])
        for p, x in [("all singles", g), *g.groupby("part")]:
            rows.append(dict(method=m, part=p, n=len(x), NormMSE=x.NormMSE.mean(), Pearson=x.Pearson.mean()))
    T = pd.DataFrame(rows)
    T.to_csv(Path(a.results) / "familiar.csv", index=False)
    T["v"] = T.NormMSE.map("{:.3f}".format) + " / " + T.Pearson.map("{:.3f}".format)
    print("DE NormMSE / DE Pearson (n:", T.groupby("part").n.first().to_dict(), ")")
    print(T.pivot(index="method", columns="part", values="v").reindex(T.method.unique())
          [["all singles", "familiar", "unfamiliar"]].to_string())


if __name__ == "__main__":
    main()
