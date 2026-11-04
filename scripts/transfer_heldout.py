#!/usr/bin/env python3
"""Cross-cell-line transfer of the learned similarity (Fig. 3; Supplement Tables 34-35, Fig. S4; specified before testing).

For each held-out line L, similarity parameters {w, tau, U, V} were trained on the gd libraries of the other
three lines only (L never a training query or key; epoch selected on the RPE1 selection genes, K562 selection genes
when L = RPE1) and frozen; the per-line output scales were then fitted by leave-one-out least squares on each line's
library. Released as weights/heldout_line/<split>/transfer_hold<line>_s<seed> (5 seeds) with the matching
initialization weights/heldout_line/<split>/init_hold<line> (soft GoBERT kNN with the same per-line scales).
This script applies the transfer model to the evaluation genes of L with L's own library and compares it with (i) the
initialization and (ii) the released model trained on all four lines of the same seed (weights/heldout/gd/<split>/pertrieve_s<seed>).
Response Pearson averaged over seeds (the metric of Tables 34-35); paired differences transfer minus initialization per
line (gene bootstrap) and over the four lines (line-weighted, pertrieve.stats.paired_lw), transfer minus the four-line
model over the four lines; Holm over these tests. Outputs <out>/<split>/{summary,per_line,paired}.csv.

  python scripts/transfer_heldout.py --split unseen
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, Pseudobulk, heldout_split, load_representations
from pertrieve.metrics import clip_delta, per_perturbation
from pertrieve.stats import holm, paired_lw

REPO = Path(__file__).resolve().parents[1]
LINES = ("rpe1", "k562", "hepg2", "jurkat")
SEEDS = (0, 1, 2, 3, 4)
METS = ["pearson_delta"]
PAIRED_COLS = ["model", "vs", "line", "metric", "diff", "lo", "hi", "p_wilcoxon", "p_holm", "n_genes", "n_rows", "n_lines", "lines_all_positive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--split", default="unseen", choices=["unseen", "cluster", "cluster_genept"])
    ap.add_argument("--out", default=str(REPO / "results/heldout_line"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    sp = heldout_split(REPO / "splits/heldout", a.split, "gd")
    ev = set(sp["queries"]["k562"])  # the evaluation genes (60% of the held-out pool), the same for every line
    W = REPO / "weights/heldout_line" / a.split
    per, rows = {}, []
    for c in LINES:
        q = sorted(g for g in pb.genes(c) if g in ev)
        lib = [g for g in pb.genes(c) if g not in sp["heldout"]]
        R, ctrl, D = pb.delta(c, q), pb.control(c), pb.delta(c, lib)
        models = {"init": [Pertrieve.load(W / f"init_hold{c}")],
                  "transfer": [Pertrieve.load(W / f"transfer_hold{c}_s{s}") for s in SEEDS],
                  "full": [Pertrieve.load(REPO / "weights/heldout/gd" / a.split / f"pertrieve_s{s}") for s in SEEDS]}
        reps = load_representations(Path(a.data) / "representations", models["transfer"][0].representations)
        Q, K = [r.matrix(q) for r in reps], [r.matrix(lib) for r in reps]
        for name, ms in models.items():
            m = sum(per_perturbation(clip_delta(mm.predict(Q, K, D, c), ctrl), R) for mm in ms) / len(ms)
            per.setdefault(name, []).append(m.assign(line=c, gene=q))
            rows.append(dict(model=name, line=c, n=len(q), n_seeds=len(ms), **m[METS].mean().to_dict()))
        print(f"{a.split} {c}: {len(q)} evaluation genes, library {len(lib)}", flush=True)
    PL = pd.DataFrame(rows)
    S = PL.groupby("model")[METS].mean().loc[["init", "transfer", "full"]]
    per = {k: pd.concat(v, ignore_index=True) for k, v in per.items()}
    T = []
    for other, lines in (("init", list(LINES) + ["all"]), ("full", ["all"])):
        for c in lines:
            A = per["transfer"] if c == "all" else per["transfer"][per["transfer"].line == c]
            B = per[other] if c == "all" else per[other][per[other].line == c]
            for met in METS:
                sg = -1 if met == "mse" else 1
                T.append(dict(model="transfer", vs=other, line=c, metric=met,
                              **paired_lw(sg * A[met].to_numpy(), sg * B[met].to_numpy(), A.gene.to_numpy(), A.line.to_numpy())))
    T = pd.DataFrame(T); T["p_holm"] = holm(T.p_wilcoxon.fillna(1).to_numpy()); T = T[PAIRED_COLS]
    out = Path(a.out) / a.split; out.mkdir(parents=True, exist_ok=True)
    PL.to_csv(out / "per_line.csv", index=False); S.to_csv(out / "summary.csv"); T.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(4).to_string())
    print(T[T.metric == "pearson_delta"][["vs", "line", "diff", "lo", "hi", "lines_all_positive", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
