#!/usr/bin/env python3
"""Compare results/ with the reference values in expected/ (the numbers reported in the paper).

Held-out genes: expected/heldout_{summary,paired}.csv and expected/heldout_baselines_{summary,paired}.csv against
results/heldout/ (scripts/reproduce_heldout.sh; SLIM rows only if SLIM predictions were made).
Other benchmarks: expected/{other_benchmarks,fewshot_extra,fewshot_de}.csv against whichever of results/{fewshot,
fewshot_de,slim_protocol,gears_format,scouter_format,perturbqa}/ exist, and expected/{ptrag_summary,ptrag_paired,
scouter_familiar,cellsimbench_counts}.csv against results/{ptrag,scouter_format/norman/familiar.csv,
cellsimbench_stats}/, and expected/{heldout_line_*,fewshot_bio_*,fewshot_confirm_*,magnitude_*}.csv
against results/{heldout_line,fewshot_bio,fewshot_confirm,magnitude}/ (scripts/reproduce_other.sh), and
expected/retrieval_summary.csv against results/retrieval_analysis/ (scripts/analyze_retrieval.py), and
expected/combo_selected.csv against results/combo_selected/ (scripts/select_combo_variant.py). Repeated outer
splits: expected/repeat_*.csv against results/repeat/ (SLIM rows and outer splits only if they were run). Results that
were not produced are skipped. Rows that a result file holds but expected/ does not (the outputs are restricted to
the rows reported in the paper) are listed as EXTRA.

  python scripts/check_expected.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KEYS = {"summary": ["protocol", "split", "model"], "paired": ["protocol", "split", "model", "vs", "metric"]}
OTHER_KEYS = ["benchmark", "dataset", "variant", "group", "metric"]


def compare(m, keys, cols, label):
    bad = 0
    for c in cols:
        if c not in m:
            continue
        e, g = m[c], m[f"{c}_got"] if f"{c}_got" in m else pd.Series(np.nan, index=m.index)
        d = (e - g).abs()
        miss = g.isna() & e.notna()
        off = (d > 1e-4 * (0.1 if "mse" in str(c).lower() else 1)) | (e.isna() & g.notna())
        bad += int(miss.sum() + off.sum())
        for _, r in m[miss | off].iterrows():
            print(f"MISMATCH {label} {[r[k] for k in keys]} {c}: expected {r[c]:.6g}, got {g[_]}")
    return bad


def extra(exp, got, keys, label):
    """Rows of a result file that expected/ does not hold."""
    x = got[keys].drop_duplicates().merge(exp[keys].drop_duplicates(), on=keys, how="left", indicator=True)
    x = x[x._merge == "left_only"]
    for _, r in x.iterrows():
        print(f"EXTRA {label} {[r[k] for k in keys]}")
    return len(x)


def check_heldout():
    bad = 0
    for kind, keys in KEYS.items():
        exp = pd.read_csv(REPO / f"expected/heldout_{kind}.csv")
        base = pd.read_csv(REPO / f"expected/heldout_baselines_{kind}.csv")
        if not any((REPO / "predictions/heldout").glob("*/*/slim_string_*_s0_*.npz")):  # SLIM not installed / not run
            slim = base.model.str.startswith("slim_")
            if "vs" in base:
                slim |= base.vs.astype(str).str.startswith("slim_")
            base = base[~slim]
        exp = pd.concat([exp, base], ignore_index=True)
        got = []
        for f in sorted((REPO / "results/heldout").glob(f"*/*/{kind}.csv")):
            proto, split = f.parts[-3], f.parts[-2]
            got.append(pd.read_csv(f).assign(protocol=proto, split=split).dropna(axis=1, how="all"))
        got = pd.concat(got)
        if not any((REPO / "predictions/heldout").glob("*/*/slim_string_*_s0_*.npz")):
            got = got[~got.model.astype(str).str.startswith("slim_")]
        m = exp.merge(got, on=keys, how="left", suffixes=("", "_got"))
        bad += compare(m, keys, ["pearson_delta", "disc_cos", "mse"] if kind == "summary" else ["diff", "lo", "hi"],
                       f"heldout {kind}")
        bad += extra(exp, got, keys, f"heldout {kind}")
        print(f"heldout {kind}: {len(m)} rows checked")
    return bad


def collect_other(res):
    """Long table (benchmark, dataset, variant, group, metric, value) of the other benchmarks' results."""
    rows = []

    def add(bench, ds, var, group, metric, value):
        if pd.isna(value):  # a metric the paper does not report for this row
            return
        rows.append(dict(benchmark=bench, dataset=ds, variant=var, group=group, metric=metric, value=float(value)))

    f = res / "fewshot/summary.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            for c in ("pearson_delta", "disc_cos", "mse"):
                add("fewshot", "replogle_nadig", r["model"], "all", c, r[c])
    f = res / "slim_protocol/summary.csv"
    if f.exists():
        for _, r in pd.read_csv(f).iterrows():
            for c in ("pearson_delta_mean", "pearson_delta_std", "disc_cos_mean", "mse_mean"):
                add("slim_protocol", r["dataset"], r["method"], "all", c, r[c])
    for f in sorted(res.glob("gears_format/*/summary_*.csv")):
        S = pd.read_csv(f).drop(columns="seed").mean()
        for g, v in S.items():
            add("gears_format", f.parent.name, f.stem[len("summary_"):], g, "pearson_delta", v)
    for f in sorted(res.glob("scouter_format/*/summary_*.csv")):
        S = pd.read_csv(f).drop(columns="seed").groupby("subgroup").mean()
        for g, r in S.iterrows():
            for c in ("NormMSE", "Pearson"):
                add("scouter_format", f.parent.name, f.stem[len("summary_"):], g, c, r[c])
    f = res / "fewshot_de/summary.csv"
    if f.exists():
        for v, r in pd.read_csv(f).set_index("variant").iterrows():
            for c, x in r.items():
                if np.issubdtype(type(x), np.number) and not np.isnan(x):
                    add("fewshot_de", "replogle_nadig", v, "all", c, x)
    f = res / "perturbqa/auc.csv"
    if f.exists():
        for (task, line), v in pd.read_csv(f).groupby(["task", "line"]).AUC.mean().items():
            add("perturbqa", line, task, "all", "AUC", v)
    return pd.DataFrame(rows, columns=OTHER_KEYS[:3] + ["group", "metric", "value"])


def check_other():
    exp = pd.concat([pd.read_csv(REPO / f"expected/{n}.csv") for n in ("other_benchmarks", "fewshot_extra", "fewshot_de")],
                    ignore_index=True)
    got = collect_other(REPO / "results")
    exp = exp[exp.benchmark.isin(set(got.benchmark))]
    m = exp.merge(got, on=OTHER_KEYS, how="left", suffixes=("", "_got"))
    bad = compare(m, OTHER_KEYS, ["value"], "other") + extra(exp, got, OTHER_KEYS, "other")
    for b, n in m.groupby("benchmark").size().items():
        print(f"{b}: {n} rows checked")
    return bad


TABLES = [  # expected file, result file (a glob: the parent directory name becomes the first key), keys, value columns
    ("ptrag_summary", "ptrag/summary.csv", ["method"], ["pearson_delta", "disc_cos", "disc_l2", "mse", "residual_pearson"]),
    ("ptrag_paired", "ptrag/paired.csv", ["model", "vs", "metric"], ["diff", "lo", "hi"]),
    ("scouter_familiar", "scouter_format/norman/familiar.csv", ["method", "part"], ["NormMSE", "Pearson"]),
    ("fewshot_paired", "fewshot/paired.csv", ["model", "vs", "metric"], ["diff", "lo", "hi"]),
    ("heldout_line_summary", "heldout_line/*/summary.csv", ["split", "model"], ["pearson_delta", "disc_cos", "mse"]),
    ("heldout_line_paired", "heldout_line/*/paired.csv", ["split", "vs", "line", "metric"], ["diff", "lo", "hi"]),
    ("fewshot_bio_breadth_means", "fewshot_bio/breadth/group_means.csv", ["analysis", "stratum", "model"], ["pearson", "de_sign", "de_mse", "auprc"]),
    ("fewshot_bio_breadth_paired", "fewshot_bio/breadth/paired.csv", ["analysis", "stratum", "vs", "metric"], ["diff", "lo", "hi"]),
    ("fewshot_bio_hallmark_means", "fewshot_bio/hallmark/group_means.csv", ["stratum", "model"], ["spearman", "top5_sign", "top5_overlap"]),
    ("fewshot_bio_hallmark_paired", "fewshot_bio/hallmark/paired.csv", ["stratum", "vs", "metric"], ["diff", "lo", "hi"]),
    ("fewshot_confirm_summary", "fewshot_confirm/*/summary.csv", ["task", "model"], ["pearson_delta", "disc_cos", "mse", "residual_pearson"]),
    ("fewshot_confirm_paired", "fewshot_confirm/*/paired.csv", ["task", "model", "vs", "metric"], ["diff", "lo", "hi"]),
    ("repeat_per_split", "repeat/per_split.csv", ["task", "outer_split", "model"], ["pearson_delta"]),
    ("repeat_paired", "repeat/paired_per_split.csv", ["task", "outer_split", "vs"], ["diff", "lo", "hi"]),
    ("magnitude_summary", "magnitude/summary.csv", ["task", "model"], ["spearman_norm"]),
    ("magnitude_paired", "magnitude/paired.csv", ["task", "model", "vs"], ["diff", "lo", "hi"]),
    ("retrieval_summary", "retrieval_analysis/summary.csv", ["split", "quantity"], ["trained", "untrained", "library_share", "diff", "lo", "hi"]),
    ("combo_selected", "combo_selected/summary.csv", ["benchmark", "metric"], ["value"]),
]
OPTIONAL = {"repeat_per_split": "model", "repeat_paired": "vs"}  # SLIM rows and unrun outer splits are skipped
COUNTS = {"table5_7": "cellsimbench_stats/counts.csv"}


def check_tables():
    bad, checked = 0, False
    for name, res, keys, cols in TABLES:
        files = sorted((REPO / "results").glob(res))
        if not files or not (REPO / f"expected/{name}.csv").exists():
            continue
        got = pd.concat([pd.read_csv(f).assign(**{keys[0]: f.parent.name}) if "*" in res else pd.read_csv(f) for f in files],
                        ignore_index=True)
        exp = pd.read_csv(REPO / f"expected/{name}.csv")
        if name in OPTIONAL:
            exp = exp.merge(got[["task", "outer_split"]].drop_duplicates(), on=["task", "outer_split"])
            if not got[OPTIONAL[name]].str.startswith("SLIM").any():
                exp = exp[~exp[OPTIONAL[name]].str.startswith("SLIM")]
        m = exp.merge(got, on=keys, how="left", suffixes=("", "_got"))
        bad += compare(m, keys, cols, name) + extra(exp, got, keys, name); checked = True
        print(f"{name}: {len(m)} rows checked")
    exp = pd.read_csv(REPO / "expected/cellsimbench_counts.csv")
    for tag, res in COUNTS.items():
        if not (REPO / "results" / res).exists():
            continue
        e = exp[exp.datasets == tag].drop(columns="datasets")
        g = pd.read_csv(REPO / "results" / res)
        m = e.merge(g, on="family", how="left", suffixes=("", "_got")); bad += extra(e, g, ["family"], f"cellsimbench counts {tag}")
        for c in [c for c in e.columns if c != "family"]:
            off = m[c] != m[f"{c}_got"]
            bad += int(off.sum())
            for _, r in m[off].iterrows():
                print(f"MISMATCH cellsimbench counts {tag} {r.family} {c}: expected {r[c]}, got {r[f'{c}_got']}")
        print(f"cellsimbench counts ({tag}): {len(m)} rows checked"); checked = True
    return bad, checked


def main():
    bad, checked = 0, False
    if any((REPO / "results/heldout").glob("*/*/summary.csv")):
        bad += check_heldout(); checked = True
    if not collect_other(REPO / "results").empty:
        bad += check_other(); checked = True
    b, c = check_tables(); bad += b; checked |= c
    if not checked:
        sys.exit("no results found; run scripts/reproduce_heldout.sh or scripts/reproduce_other.sh first")
    print("all results match" if bad == 0 else f"{bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
