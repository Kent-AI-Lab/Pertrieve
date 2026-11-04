#!/usr/bin/env python3
"""Pertrieve on PerturbQA (Wu et al., ICLR 2025): differential-expression (de) and direction (dir) tasks in
K562, RPE1, HepG2 and Jurkat, with the official train/test pairs.

Here retrieval weights combine the reference perturbations' labels instead of their expression responses:
  score(p, g) = sum_j a_pj H_jg L_jg / sum_j a_pj H_jg
with a_pj the Pertrieve weights over training perturbations j, L the training labels and H = 1 where pair (j, g)
is labeled. A test pair falls back to the gene's training label frequency when the perturbation has no
representation or no labeled reference carries weight. Metric: AUC per gene, averaged over genes (the official
perturbqa.eval.auc_per_gene; genes with a single label class are skipped).
Inputs: data/perturbqa/datasets/<line>-<task>.csv (PerturbQA release; columns pert, gene, label, split) and
data/perturbqa/representations/{GoBERT,GenePT_v2}.npz. Output results/perturbqa/auc.csv.

  python scripts/run_perturbqa.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from pertrieve import Pertrieve, load_representations

REPO = Path(__file__).resolve().parents[1]
LINES = ("k562", "rpe1", "hepg2", "jurkat")


def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_per_gene(genes, pred, true):
    df = pd.DataFrame(dict(g=genes, p=pred, t=true))
    vals = [auc(d.t.to_numpy(), d.p.to_numpy()) for _, d in df.groupby("g", sort=False) if 0 < d.t.sum() < len(d)]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "results/perturbqa"))
    a = ap.parse_args()
    rows = []
    for c in LINES:
        for task in ("de", "dir"):
            df = pd.read_csv(Path(a.data) / f"perturbqa/datasets/{c}-{task}.csv")
            tr, te = df[df.split == "train"], df[df.split == "test"].reset_index(drop=True)
            M = tr.pivot_table(index="pert", columns="gene", values="label", aggfunc="mean")
            freq = tr.groupby("gene").label.mean()
            gcol = {g: i for i, g in enumerate(M.columns)}
            for f in sorted((REPO / "weights/perturbqa" / c).glob(f"{task}_s[0-9].npz")):
                m = Pertrieve.load(f)
                reps = load_representations(Path(a.data) / "perturbqa/representations", m.representations)
                has = lambda ps: np.any([np.any(r.matrix(ps) != 0, axis=1) for r in reps], axis=0)  # noqa: E731
                lib = [p for p, h in zip(M.index, has(list(M.index))) if h]
                L = M.loc[lib].to_numpy(); H = ~np.isnan(L); L = np.nan_to_num(L)
                tep = sorted(te.pert.unique())
                A = m.weights([r.matrix(tep) for r in reps], [r.matrix(lib) for r in reps])
                S, C = (A @ (H * L)) / (A @ H + 1e-6), A @ H
                row = {p: i for i, (p, h) in enumerate(zip(tep, has(tep))) if h}
                r = np.array([row.get(p, -1) for p in te.pert]); col = np.array([gcol.get(g, -1) for g in te.gene])
                pred = np.array([freq.get(g, 0.5) for g in te.gene])
                ok = (r >= 0) & (col >= 0)
                ok[ok] = C[r[ok], col[ok]] > 1e-4
                pred[ok] = S[r[ok], col[ok]]
                rows.append(dict(line=c, task=task, seed=m.meta["seed"], AUC=auc_per_gene(te.gene, pred, te.label.to_numpy())))
    R = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "auc.csv", index=False)
    T = R.groupby(["task", "line"]).AUC.mean().unstack()
    T["mean"] = T.mean(axis=1)
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
