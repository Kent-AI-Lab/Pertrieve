#!/usr/bin/env python3
"""Paired statistics on CellSimBench scores (Supplement Section N, Tables 41-42).

Input per dataset: the per-perturbation score file of one `cellsimbench benchmark` run (detailed_metrics.csv; columns
model, perturbation, metric, value) that contains Pertrieve's predictions (3 seeds, written by run_cellsimbench.py)
and the benchmark's own models. data/cellsimbench/<ds>/scores.csv.gz holds the runs used in the paper (the 13 metrics
only).

Protocol (fixed before computing):
  * Perturbation set: per dataset, the perturbations scored for Pertrieve (all seeds) and for every non-oracle model
    (fMLP x4, GEARS, linear or additive, control_mean, dataset_mean); every mean is computed on that one set.
  * Pertrieve: `plain` on single-gene datasets, `auto` on combination datasets (Norman, Wessels, Sunshine);
    per-perturbation mean over seeds.
  * Pertrieve vs the best non-oracle competitor per (dataset, metric), chosen by the competitor means on that set.
  * Two-sided paired Wilcoxon signed-rank test over perturbations; Holm over all tests. Calibrated metrics = the five
    with dynamic range fraction >= 0.29 in the CellSimBench paper.
Outputs results/cellsimbench_stats/{comparison_A.csv (Table 42), counts.csv (Table 41), coverage.csv}.

  python scripts/cellsimbench_stats.py                                  # the 7 datasets of Supplement Table 41
  python scripts/cellsimbench_stats.py --scores norman19=outputs/benchmark_all_norman19/<run>/detailed_metrics.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from pertrieve.stats import holm

REPO = Path(__file__).resolve().parents[1]
TABLE5 = ["replogle22k562", "replogle22rpe1", "nadig25hepg2", "nadig25jurkat", "adamson16", "norman19", "wessels23"]
COMBO = {"norman19", "wessels23"}
M13 = ["mse", "wmse", "nir", "pearson_deltactrl", "pearson_deltactrl_degs", "pearson_deltapert", "pearson_deltapert_degs",
       "r2_deltactrl", "r2_deltactrl_degs", "weighted_r2_deltactrl", "r2_deltapert", "r2_deltapert_degs",
       "weighted_r2_deltapert"]
CALIB = ["wmse", "weighted_r2_deltactrl", "nir", "pearson_deltapert_degs", "weighted_r2_deltapert"]
LOWER = {"mse", "wmse"}
ORACLES = {"technical_duplicate", "interpolated_duplicate", "ground_truth"}
SEEDS = (0, 1, 2)



def compare(ds, f):
    d = pd.read_csv(f)
    d["model"] = d.model.str.replace("^pre_", "", regex=True)
    d = d[d.metric.isin(M13)]
    ours_m = [f"retrieval_{'auto' if ds in COMBO else 'plain'}_s{s}" for s in SEEDS]
    comps = sorted(m for m in d.model.unique() if m not in ORACLES and not m.startswith("retrieval"))
    persets = {m: set(d[d.model == m].perturbation) for m in d.model.unique()}
    missing = [m for m in ours_m if m not in persets]
    if missing:
        raise SystemExit(f"{ds}: no scores for {missing} in {f}")
    common = set.intersection(*[persets[m] for m in ours_m + comps])
    W = d[d.perturbation.isin(common)].pivot_table(index=["perturbation", "metric"], columns="model", values="value")
    ours = W[ours_m].mean(axis=1)
    cov = dict(dataset=ds, n_all=len(set(d.perturbation)), n_common=len(common), competitors=",".join(comps))
    A = []
    for met in M13:
        o = ours.xs(met, level="metric").sort_index()
        sign = -1 if met in LOWER else 1
        cm = {c: W[c].xs(met, level="metric").sort_index() for c in comps}
        valid = [c for c in comps if cm[c].notna().any()]  # e.g. Pearson of a constant prediction is undefined
        best = max(valid, key=lambda c: sign * cm[c].mean())
        dd = sign * (o - cm[best]).to_numpy()
        dd = dd[~np.isnan(dd)]
        A.append(dict(dataset=ds, metric=met, competitor=best, n=len(dd), pertrieve=o.mean(), comp=cm[best].mean(),
                      win=bool(np.mean(dd) > 0), frac_perts_better=float((dd > 0).mean()),
                      p=float(wilcoxon(dd).pvalue) if np.any(dd != 0) else 1.0, calibrated=met in CALIB))
    return A, cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--ds", nargs="+", default=TABLE5, help="dataset names (default: the 7 datasets of the paper)")
    ap.add_argument("--scores", nargs="*", default=[], help="DS=PATH to a detailed_metrics.csv (overrides the data dir)")
    ap.add_argument("--out", default=str(REPO / "results/cellsimbench_stats"))
    a = ap.parse_args()
    ds_list = a.ds
    paths = {ds: Path(a.data) / "cellsimbench" / ds / "scores.csv.gz" for ds in ds_list}
    paths.update(dict(s.split("=", 1) for s in a.scores))
    A, C = [], []
    for ds in ds_list:
        r = compare(ds, paths[ds])
        A += r[0]; C.append(r[1])
    A = pd.DataFrame(A)
    A["p_holm"] = holm(A.p)

    def counts(df, lab):  # Table 41: favorable means, favorable and unfavorable with Holm p < 0.05
        return dict(family=lab, n_tests=len(df), wins=int(df.win.sum()), wins_holm=int((df.win & (df.p_holm < .05)).sum()),
                    losses_holm=int((~df.win & (df.p_holm < .05)).sum()))
    K = pd.DataFrame([counts(A[A.dataset == ds], ds) for ds in ds_list]
                     + [counts(A, f"all {len(ds_list)} datasets x 13 metrics (Holm over {len(A)})")])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    A.drop(columns="p").to_csv(out / "comparison_A.csv", index=False); pd.DataFrame(C).to_csv(out / "coverage.csv", index=False)
    K.to_csv(out / "counts.csv", index=False)
    pd.set_option("display.width", 200)
    print(K.to_string(index=False))
    T = A[A.calibrated].assign(v=lambda x: x.pertrieve.round(4).astype(str) + np.where(x.win, "+", "-")
                               + np.where(x.win & (x.p_holm < .05), "*", ""))
    print("\nCalibrated metrics (+ better than the best non-oracle competitor, * Holm p < 0.05):")
    print(T.pivot(index="dataset", columns="metric", values="v").reindex(ds_list)[CALIB].to_string())


if __name__ == "__main__":
    main()
