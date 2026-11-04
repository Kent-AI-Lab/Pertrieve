#!/usr/bin/env python3
"""Few-shot performance by response breadth (Fig. 5; Supplement Section J, Tables 26-28).

Official few-shot test sets, four lines. Strata from the measured data only: the number of true DE genes of each
perturbation (cell-eval Wilcoxon test on the real test cells, per-perturbation BH, FDR < 0.05; data/fewshot_bio/real_de),
per line: fewer than 10 DE genes = "minimal", the rest split at tertiles into narrow / medium / broad. Sensitivity:
FDR < 0.05 and |log2 FC| > log2(1.5) ("de_rule_fc"), FDR < 0.05 after BH over the whole line ("de_rule_globalBH"), and
perturbations with a cell count inside the line's interquartile range ("cells_iqr"); counts in data/fewshot_bio/
real_de_counts.csv and cells_umi_per_pert.csv.
Models (predictions/fewshot, three seeds averaged; scripts/reproduce_other.sh): Pertrieve (xline+ae_xline),
Linear cross-cell-line (ae_xline), Pertrieve-Retrieval (xline), Untrained retrieval (xline0), Linear + untrained retrieval
(xline0+ae_xline) and the State checkpoint (data/fewshot_bio/state_predictions). Metrics per perturbation on the clipped pseudobulk change: Pearson over all genes;
on its true DE genes the fraction with the measured sign (de_sign), the MSE and (>= 10 DE genes) the Pearson; AUPRC of
the true DE genes ranked by |predicted change|. Statistics: line-weighted paired differences (pertrieve.stats.paired_lw);
Holm over the 18 primary tests (Pertrieve vs State and vs Linear; Pearson, de_sign, AUPRC; narrow/medium/broad).
Paired comparisons as in Supplement Table 28: Pertrieve minus State, Linear, Linear + untrained retrieval and
Pertrieve-Retrieval on Pearson, de_sign, AUPRC and DE MSE, for the primary strata and the three sensitivity rules
(group means: primary strata, Table 27).
Outputs <out>/{per_query.csv.gz, strata.csv, group_means.csv, group_means_per_line.csv, paired.csv}.

  python scripts/breadth_fewshot.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from pertrieve import Pseudobulk
from pertrieve.metrics import clip_delta
from pertrieve.stats import holm, paired_lw

REPO = Path(__file__).resolve().parents[1]
LINES = ("k562", "rpe1", "hepg2", "jurkat")
SEEDS = (0, 1, 2)
MODELS = {"Pertrieve": "xline+ae_xline", "Linear cross-cell-line": "ae_xline", "Pertrieve-Retrieval": "xline", "State": None,
          "Untrained retrieval": "xline0", "Linear + untrained retrieval": "xline0+ae_xline"}
PAIRED_VS = ("State", "Linear cross-cell-line", "Linear + untrained retrieval", "Pertrieve-Retrieval")  # Table 28
PAIRED_COLS = ["analysis", "stratum", "vs", "metric", "n_units", "diff", "lo", "hi", "p_wilcoxon", "n_genes", "n_rows", "n_lines",
               "lines_all_positive", "primary", "p_holm"]
MET = ["pearson", "de_sign", "de_mse", "auprc", "de_pearson", "pos_frac"]
REF = "Pertrieve"


def metrics(P, R, de_idx):
    rows = []
    for i in range(len(R)):
        p, r = P[i], R[i]; d = de_idx[i]; n = len(d)
        pear = np.corrcoef(p, r)[0, 1] if p.std() > 0 and r.std() > 0 else np.nan
        y = np.zeros(len(r), bool); y[d] = True
        ap = average_precision_score(y, np.abs(p)) if 0 < n < len(r) else np.nan
        if n > 0:
            sgn = float(np.mean(np.sign(p[d]) == np.sign(r[d]))); mse_de = float(np.mean((p[d] - r[d]) ** 2))
            pear_de = (np.corrcoef(p[d], r[d])[0, 1] if n >= 10 and p[d].std() > 0 and r[d].std() > 0 else np.nan)
        else:
            sgn = mse_de = pear_de = np.nan
        rows.append(dict(pearson=pear, auprc=ap, de_sign=sgn, de_mse=mse_de, de_pearson=pear_de, n_de=n, pos_frac=n / len(r)))
    return pd.DataFrame(rows)


def strata(key, col):
    s = pd.Series("minimal", index=key.index)
    for _, g in key.groupby("line"):
        ok = g[g[col] >= 10]
        # cut-points on the percent scale, as pandas 2.x Series.quantile computes them (the released strata); pandas 3
        # computes 2/3 directly, which can move a count equal to a cut-point into the neighboring stratum
        q = np.percentile(ok[col].to_numpy(dtype=float), np.array([1 / 3, 2 / 3]) * 100.0)
        s.loc[ok.index] = np.where(ok[col] <= q[0], "narrow", np.where(ok[col] <= q[1], "medium", "broad"))
    return s


def load_models(pb, pred, bio, c, genes, ctrl):
    """Per-model list of prediction arrays (one per seed) for the test genes of line c."""
    st = np.load(bio / "state_predictions" / f"{c}.npz"); si = {g: i for i, g in enumerate(st["genes"].astype(str))}
    out = {}
    for name, v in MODELS.items():
        if name == "State":
            out[name] = [st["delta"][[si[g] for g in genes]].astype(float)]
        else:
            zs = [np.load(pred / f"{v}_s{s}_{c}.npz") for s in SEEDS]
            assert all(list(z["genes"].astype(str)) == genes for z in zs), v
            out[name] = [z["delta"].astype(float) for z in zs]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--pred", default=str(REPO / "predictions/fewshot"))
    ap.add_argument("--out", default=str(REPO / "results/fewshot_bio/breadth"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz"); bio = Path(a.data) / "fewshot_bio"; pred = Path(a.pred)
    hvg = np.loadtxt(bio / "hvg_genes.txt", dtype=str); gi = {g: i for i, g in enumerate(hvg)}; assert len(hvg) == pb.M.shape[1]
    CNT = pd.read_csv(bio / "real_de_counts.csv").set_index(["line", "target"])
    CELLS = pd.read_csv(bio / "cells_umi_per_pert.csv").set_index(["line", "gene"])
    per = []
    for c in LINES:
        ctrl = pb.control(c)
        genes = list(np.load(pred / f"xline_s0_{c}.npz")["genes"].astype(str)); R = pb.delta(c, genes)
        D = pd.read_csv(bio / "real_de" / f"{c}_test_de.csv.gz")
        DE = {t: np.array([gi[g] for g in g_.gene]) for t, g_ in D.groupby("target")}
        de_idx = [DE.get(g, np.array([], int)) for g in genes]
        for name, Ps in load_models(pb, pred, bio, c, genes, ctrl).items():
            for P in Ps:
                assert P.shape == R.shape
            m = sum(metrics(clip_delta(P, ctrl), R, de_idx) for P in Ps) / len(Ps)
            m["n_de"] = [len(d) for d in de_idx]; m["pos_frac"] = m.n_de / len(hvg)
            per.append(m.assign(model=name, line=c, gene=genes))
        print(f"{c}: {len(genes)} test perturbations", flush=True)
    per = pd.concat(per, ignore_index=True)
    key = per[per.model == REF][["line", "gene", "n_de"]].copy()
    for col in ("n_cells", "n_de_fc", "n_de_global"):
        key[col] = [CNT.loc[(l, g), col] for l, g in zip(key.line, key.gene)]
    key["umi_mean"] = [CELLS.loc[(l, g), "umi_mean"] for l, g in zip(key.line, key.gene)]
    assert (key.n_de == [CNT.loc[(l, g), "n_de"] for l, g in zip(key.line, key.gene)]).all()
    key["stratum"] = strata(key, "n_de"); key["stratum_fc"] = strata(key, "n_de_fc"); key["stratum_global"] = strata(key, "n_de_global")
    per = per.merge(key.drop(columns="n_de"), on=["line", "gene"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_query.csv.gz", index=False)
    rows = []
    for (c, s), g in key.groupby(["line", "stratum"]):
        rows.append(dict(line=c, stratum=s, n=len(g), n_de_min=int(g.n_de.min()), n_de_max=int(g.n_de.max()), n_de_median=float(g.n_de.median()),
                         cells_median=float(g.n_cells.median()), umi_median=float(g.umi_mean.median())))
    for c, g in key.groupby("line"):
        rows.append(dict(line=c, stratum="ALL", n=len(g), n_de_min=int(g.n_de.min()), n_de_max=int(g.n_de.max()), n_de_median=float(g.n_de.median()),
                         cells_median=float(g.n_cells.median()), umi_median=float(g.umi_mean.median()),
                         spearman_nde_cells=g[["n_de", "n_cells"]].corr("spearman").iloc[0, 1], spearman_nde_umi=g[["n_de", "umi_mean"]].corr("spearman").iloc[0, 1],
                         n_de0=int((g.n_de == 0).sum()), n_de_lt10=int((g.n_de < 10).sum())))
    pd.DataFrame(rows).to_csv(out / "strata.csv", index=False)

    def summarise(per, scol, tag):
        G = per.groupby([scol, "model", "line"])[MET].mean().reset_index()
        M = G.groupby([scol, "model"])[MET].mean().reset_index()
        M["n_units"] = per[per.model == REF].groupby(scol).size().reindex(M[scol]).to_numpy()
        M["analysis"] = tag; G["analysis"] = tag
        return M.rename(columns={scol: "stratum"}), G.rename(columns={scol: "stratum"})

    def paired(per, scol, tag, strata_list):
        T = []
        A = per[per.model == REF].set_index(["line", "gene"]).sort_index()
        for other in PAIRED_VS:
            B = per[per.model == other].set_index(["line", "gene"]).sort_index(); assert A.index.equals(B.index)
            for s in strata_list:
                m = (A[scol] == s).to_numpy()
                for met in ("pearson", "de_sign", "de_mse", "auprc"):
                    sg = -1 if met == "de_mse" else 1
                    x, y = A[met].to_numpy()[m], B[met].to_numpy()[m]; ok = np.isfinite(x) & np.isfinite(y)
                    if ok.sum() < 20:
                        continue
                    r = paired_lw(sg * x[ok], sg * y[ok], A.index.get_level_values(1)[m][ok], A.index.get_level_values(0)[m][ok])
                    T.append(dict(analysis=tag, stratum=s, vs=other, metric=met, n_units=int(ok.sum()), **r))
        T = pd.DataFrame(T)
        prim = T.vs.isin(["State", "Linear cross-cell-line"]) & T.metric.isin(["pearson", "de_sign", "auprc"]) & T.stratum.isin(["narrow", "medium", "broad"]) & (T.analysis == "primary")
        T["primary"] = prim; T["p_holm"] = np.nan
        if prim.any():
            T.loc[prim, "p_holm"] = holm(T.loc[prim, "p_wilcoxon"].fillna(1).to_numpy())
        return T[PAIRED_COLS]

    M1, G1 = summarise(per, "stratum", "primary"); T1 = paired(per, "stratum", "primary", ["narrow", "medium", "broad", "minimal"])
    iqr = pd.Series(False, index=per.index)
    for c, g in key.groupby("line"):
        lo, hi = np.percentile(g.n_cells.to_numpy(dtype=float), [25.0, 75.0]); sel = set(g[(g.n_cells >= lo) & (g.n_cells <= hi)].gene)
        iqr |= (per.line == c) & per.gene.isin(sel)
    T2 = paired(per[iqr], "stratum", "cells_iqr", ["narrow", "medium", "broad"])
    T3 = paired(per, "stratum_fc", "de_rule_fc", ["narrow", "medium", "broad"])
    T4 = paired(per, "stratum_global", "de_rule_globalBH", ["narrow", "medium", "broad"])
    M1.to_csv(out / "group_means.csv", index=False); G1.to_csv(out / "group_means_per_line.csv", index=False)
    pd.concat([T1, T2, T3, T4]).to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(M1[M1.model.isin(["State", "Linear cross-cell-line", REF])].round(3).to_string(index=False))
    print(T1[T1.primary][["stratum", "vs", "metric", "n_units", "diff", "lo", "hi", "lines_all_positive", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
