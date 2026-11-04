#!/usr/bin/env python3
"""Score held-out-gene predictions (scripts/predict_heldout.py) on K562, HepG2 and Jurkat.

Models: the prediction sets found for the split/protocol, plus the training-free baselines computed here (GoBERT kNN
with the validation-selected k in splits/heldout/knn_k.json, and the context mean). Per-perturbation metrics are
averaged over seeds; table values are means over the three test lines. Paired comparisons with Pertrieve-Retrieval are
line-weighted (pertrieve.stats.paired_lw: mean difference within each line, averaged over lines; gene-clustered
bootstrap intervals, B = 2000; Wilcoxon tests), as in the paper. The outputs hold the models, comparisons and metrics
reported in the paper (REPORTED below: Tables 1-2; Supplement Tables 9-12, 16, 43-44, 46 and 65).
Outputs <out>/<protocol>/<split>/{summary,per_line,paired}.csv.

  python scripts/evaluate_heldout.py --split unseen --protocol gd [--cell-eval]
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pseudobulk, heldout_split, knn_predict, load_representations
from pertrieve.metrics import cell_eval_metrics, clip_delta, per_perturbation, residual_pearson
from pertrieve.stats import paired_lw

TEST_LINES = ("k562", "hepg2", "jurkat")
METRICS = ["pearson_delta", "disc_cos", "mse", "residual_pearson"]
REPO = Path(__file__).resolve().parents[1]
PD, PD_COS, ALL3 = ("pearson_delta",), ("pearson_delta", "disc_cos"), ("pearson_delta", "disc_cos", "mse")
# rows and metrics reported in the paper: summary {model: metrics}, paired {comparator: metrics} (Pertrieve-Retrieval
# minus comparator); "GoBERT kNN" stands for the validation-selected kNN of the split
REPORTED = {
    "gd": dict(summary={"pertrieve": ALL3, "GoBERT kNN": ALL3, "context mean": PD_COS, "genepert_ridge_gobert": PD_COS,
                        "ae_linear_gobert": PD_COS, "ae_linear_all4": PD_COS, "slim_string_default": PD_COS,
                        "slim_string_selected": PD_COS, "soft_knn_init": PD_COS, "only_gobert": PD_COS, "no_gobert": PD_COS,
                        "only_genept": PD, "only_esm2": PD, "only_scgpt": PD, "only_random": PD, "only_gobert_shuf": PD},
               paired={"GoBERT kNN": ("pearson_delta", "mse", "residual_pearson"), "ae_linear_all4": PD,
                       "no_gobert": PD_COS, "only_gobert": PD, "soft_knn_init": PD}),
    "full": dict(summary={"pertrieve": ALL3, "GoBERT kNN": ALL3, "ablation_no_mixture": ALL3, "ablation_no_bilinear": ALL3,
                          "ablation_temperature_only": ALL3},
                 paired={"GoBERT kNN": PD}),
}
OVERRIDE = {  # split-specific entries replace the protocol default
    ("full", "cluster_genept"): dict(summary={"pertrieve": ("pearson_delta", "mse"), "GoBERT kNN": ("pearson_delta", "mse")}, paired={}),
    ("gd", "cluster"): dict(paired={**REPORTED["gd"]["paired"], "soft_knn_init": ("pearson_delta", "residual_pearson")}),
    ("full", "cluster"): dict(paired={"GoBERT kNN": PD, "soft_knn_init": ("residual_pearson",)})}
PAIRED_COLS = ["model", "vs", "metric", "diff", "lo", "hi", "p_wilcoxon", "n_genes", "n_rows", "n_lines"]


def reported(protocol, split, kind):
    return OVERRIDE.get((protocol, split), {}).get(kind, REPORTED[protocol][kind])


def base(name):
    return "GoBERT kNN" if name.startswith("GoBERT kNN") else name


def trim(df, protocol, split):
    """Keep the models reported in the paper; blank the metrics the paper does not report for a model."""
    keep = reported(protocol, split, "summary")
    df = df[[base(m) in keep for m in df.model]].copy()
    for c in METRICS:
        df[c] = [v if c in keep[base(m)] else np.nan for m, v in zip(df.model, df[c])]
    return df.drop(columns=[c for c in METRICS if df[c].isna().all()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--split", default="unseen", choices=["unseen", "cluster", "cluster_genept"])
    ap.add_argument("--protocol", default="gd", choices=["gd", "full"])
    ap.add_argument("--pred", default=str(REPO / "predictions/heldout"))
    ap.add_argument("--out", default=str(REPO / "results/heldout"))
    ap.add_argument("--reference", default="pertrieve", help="model compared against all others")
    ap.add_argument("--cell-eval", action="store_true", help="also score Pertrieve seed 0 with the official cell-eval package")
    a = ap.parse_args()

    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    sp = heldout_split(REPO / "splits/heldout", a.split, a.protocol)
    k = json.loads((REPO / "splits/heldout/knn_k.json").read_text())[a.protocol][a.split]
    gobert = load_representations(Path(a.data) / "representations", ["gobert"])[0]
    pdir = Path(a.pred) / a.protocol / a.split
    stems = sorted({re.sub(r"_s\d+_\w+$", "", f.stem) for f in pdir.glob("*_s[0-9]_*.npz")})

    per, rows = {}, []
    for c in TEST_LINES:
        lib = [g for g in pb.genes(c) if g not in sp["heldout"]]
        q = sorted(g for g in pb.genes(c) if g in sp["queries"][c])
        ctrl, R, D = pb.control(c), pb.delta(c, q), pb.delta(c, lib)
        cm = pb.context_mean(c, lib)
        C = np.repeat(clip_delta(cm, ctrl)[None], len(q), 0)
        preds = {f"GoBERT kNN (k={k})": [knn_predict(gobert.matrix(q), gobert.matrix(lib), D, k, cm)], "context mean": [C]}
        for s in stems:
            zs = [np.load(f) for f in sorted(pdir.glob(f"{s}_s[0-9]_{c}.npz"))]
            assert all(list(z["genes"]) == q for z in zs)
            preds[s] = [z["delta"].astype(np.float64) for z in zs]
        for name, Ps in preds.items():
            ms = []
            for P in Ps:
                P = clip_delta(P, ctrl)
                m = per_perturbation(P, R); m["residual_pearson"] = residual_pearson(P, R, C); ms.append(m)
            m = sum(ms) / len(ms)
            per.setdefault(name, []).append(m.assign(line=c, gene=q))
            rows.append(dict(model=name, line=c, n=len(q), n_seeds=len(Ps), **m[METRICS].mean().to_dict()))
        if a.cell_eval and a.reference in preds:
            P = clip_delta(preds[a.reference][0], ctrl)
            with tempfile.TemporaryDirectory() as tmp:
                ce = cell_eval_metrics(c, q, P + ctrl, R + ctrl, ctrl, tmp)
            print(f"cell-eval {c} {a.reference}_s0:", {kk: round(v, 6) for kk, v in ce.items()},
                  "| numpy:", per_perturbation(P, R)[["pearson_delta", "disc_cos", "mse"]].mean().round(6).to_dict())

    out = Path(a.out) / a.protocol / a.split
    out.mkdir(parents=True, exist_ok=True)
    PL = pd.DataFrame(rows)
    S = PL.groupby("model")[METRICS + ["n_seeds"]].mean().sort_values("pearson_delta").reset_index()
    trim(PL, a.protocol, a.split).to_csv(out / "per_line.csv", index=False)
    S = trim(S, a.protocol, a.split).set_index("model"); S.to_csv(out / "summary.csv")
    per = {n: pd.concat(v, ignore_index=True) for n, v in per.items()}
    T = []
    pairs = reported(a.protocol, a.split, "paired")
    if a.reference in per:
        for other in per:
            if other == a.reference or base(other) not in pairs:
                continue
            for met in pairs[base(other)]:
                sg = -1 if met == "mse" else 1
                T.append(dict(model=a.reference, vs=other, metric=met,
                              **paired_lw(sg * per[a.reference][met].to_numpy(), sg * per[other][met].to_numpy(),
                                          per[a.reference].gene.to_numpy(), per[a.reference].line.to_numpy())))
    T = pd.DataFrame(T, columns=PAIRED_COLS)[PAIRED_COLS]; T.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n== {a.protocol} / {a.split}: mean over {', '.join(TEST_LINES)}\n{S.round(4).to_string()}")
    if len(T):
        print(T[["vs", "metric", "diff", "lo", "hi", "p_wilcoxon"]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
