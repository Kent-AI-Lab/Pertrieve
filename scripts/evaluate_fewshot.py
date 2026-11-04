#!/usr/bin/env python3
"""Score few-shot predictions (scripts/predict_fewshot.py) with the unified pseudobulk scorer of the paper.

All models are scored on the complete official test sets with the same clipped pseudobulk metrics; per-
perturbation metrics are averaged over seeds and table values are means over the four lines. Baselines computed
here: context mean (cell-weighted mean response of the line's library) and perturb mean (the gene's mean
response in the other lines). Predictions of other models can be added with --extra NAME=PATTERN, where PATTERN
contains {line} and points to npz files with `genes` and `delta` (prediction minus the control mean), e.g.
--extra "State=preds/state_{line}.npz". If such a file also holds `ctrl_log1p` (the model's predicted control, as for
State's released output), the model is scored a second time as "NAME (measured control)" = delta + ctrl_log1p minus the
measured control, the alternative reference of Supplement Table 7. Paired
comparisons are line-weighted (pertrieve.stats.paired_lw), as in the paper. The outputs hold the models, comparisons and
metrics reported in the paper (REPORTED below: Tables 3-4; Supplement Tables 7, 18, 19, 21 and 65).
Outputs <out>/{summary,per_line,paired}.csv.

  python scripts/evaluate_fewshot.py [--extra NAME=PATTERN ...]
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pseudobulk
from pertrieve.data import read_genes
from pertrieve.metrics import clip_delta, per_perturbation, residual_pearson
from pertrieve.stats import paired_lw

LINES = ("k562", "rpe1", "hepg2", "jurkat")
METRICS = ["pearson_delta", "disc_cos", "mse", "residual_pearson"]
REPO = Path(__file__).resolve().parents[1]
PD, ALL3, ALL4 = ("pearson_delta",), ("pearson_delta", "disc_cos", "mse"), tuple(METRICS)
# rows and metrics reported in the paper: summary {model: metrics}; paired {comparator: metrics}, the primary model
# (xline+ae_xline) minus comparator
REPORTED = dict(
    summary={"State": ALL4, "State (measured control)": ALL3, "gene": ALL4, "ridge_xline": ALL4, "ae_xline": ALL4, "xline": ALL4,
             "xline+ae_xline": ALL4, "xline+ridge_xline": ALL4, "xline+ae_xline_perline": ALL4, "xline+ridge_xline_perline": ALL4,
             "ridge_all4": ALL4, "slim_string_sel": ALL4, "ridge_xline+ae_xline": PD, "xline0+ae_xline": PD, "xonly": ALL3,
             "perturb mean": ALL3, "context mean": PD},
    paired={"State": ("pearson_delta", "mse"), "xline": PD, "ae_xline": PD, "ridge_xline+ae_xline": PD, "xline0+ae_xline": PD,
            "xline0": PD})
PAIRED_COLS = ["model", "vs", "metric", "diff", "lo", "hi", "p_wilcoxon", "n_genes", "n_rows", "n_lines"]


def trim(df, extra=()):
    """Keep the models reported in the paper (and any --extra model); blank the metrics the paper does not report."""
    keep = {**{e: ALL4 for e in extra}, **{f"{e} (measured control)": ALL3 for e in extra}, **REPORTED["summary"]}
    df = df[df.model.isin(list(keep))].copy()
    for c in METRICS:
        df[c] = [v if c in keep[m] else np.nan for m, v in zip(df.model, df[c])]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--pred", default=str(REPO / "predictions/fewshot"))
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--reference", default="xline+ae_xline", help="model the paired comparisons are made against")
    ap.add_argument("--out", default=str(REPO / "results/fewshot"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    stems = sorted({re.sub(r"_s\d+_\w+$", "", Path(f).stem) for f in glob.glob(f"{a.pred}/*_s[0-9]_*.npz")})
    extra = dict(e.split("=", 1) for e in a.extra)
    per, rows = {}, []
    for c in LINES:
        held = set(read_genes(REPO / f"splits/fewshot/{c}.test.txt"))
        test = sorted(g for g in pb.genes(c) if g in held)
        lib = [g for g in pb.genes(c) if g not in held]
        ctrl, R = pb.control(c), pb.delta(c, test)
        C = np.repeat(clip_delta(pb.context_mean(c, lib), ctrl)[None], len(test), 0)
        pm, n = np.zeros_like(R), np.zeros(len(test))
        for o in LINES:
            if o == c:
                continue
            ok = np.array([g in set(pb.genes(o)) for g in test])
            pm[ok] += pb.delta(o, [g for g, k in zip(test, ok) if k]); n[ok] += 1
        preds = {"context mean": [C], "perturb mean": [pm / np.maximum(n, 1)[:, None]]}
        for s in stems:
            zs = [np.load(f) for f in sorted(glob.glob(f"{a.pred}/{s}_s[0-9]_{c}.npz"))]
            assert all(list(z["genes"]) == test for z in zs)
            preds[s] = [z["delta"].astype(np.float64) for z in zs]
        for name, pat in extra.items():
            z = np.load(pat.format(line=c))
            idx = {g: i for i, g in enumerate(z["genes"].astype(str))}
            preds[name] = [z["delta"][[idx[g] for g in test]].astype(np.float64)]
            if "ctrl_log1p" in z.files:  # generative model: also relative to the measured control (Supplement Table 7)
                preds[f"{name} (measured control)"] = [preds[name][0] + z["ctrl_log1p"].astype(np.float64)[None] - ctrl]
        for name, Ps in preds.items():
            ms = []
            for P in Ps:
                P = clip_delta(P, ctrl); m = per_perturbation(P, R); m["residual_pearson"] = residual_pearson(P, R, C); ms.append(m)
            m = sum(ms) / len(ms)
            per.setdefault(name, []).append(m.assign(line=c, gene=test))
            rows.append(dict(model=name, line=c, n=len(test), n_seeds=len(Ps), **m[METRICS].mean().to_dict()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    PL = pd.DataFrame(rows)
    S = trim(PL.groupby("model")[METRICS + ["n_seeds"]].mean().reset_index(), extra).sort_values("pearson_delta").set_index("model")
    trim(PL, extra).to_csv(out / "per_line.csv", index=False); S.to_csv(out / "summary.csv")
    per = {k: pd.concat(v, ignore_index=True) for k, v in per.items()}
    T = []
    for other in per:
        mets = REPORTED["paired"].get(other, METRICS if other in extra and other not in REPORTED["summary"] else ())
        if other == a.reference or a.reference not in per or not mets:
            continue
        for met in mets:
            sg = -1 if met == "mse" else 1
            T.append(dict(model=a.reference, vs=other, metric=met, **paired_lw(sg * per[a.reference][met].to_numpy(), sg * per[other][met].to_numpy(),
                                                                              per[a.reference].gene.to_numpy(), per[a.reference].line.to_numpy())))
    pd.DataFrame(T, columns=PAIRED_COLS)[PAIRED_COLS].to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(4).to_string())
    print(trim(PL, extra).pivot_table(index="model", columns="line", values="pearson_delta").round(3).to_string())


if __name__ == "__main__":
    main()
