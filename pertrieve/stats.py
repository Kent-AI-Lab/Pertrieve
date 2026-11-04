"""Paired comparisons used in the paper: line-weighted (paired_lw, the paper's tables) and pooled (paired) gene-clustered
bootstrap intervals, Wilcoxon tests, Holm correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired(a, b, genes, B: int = 2000, seed: int = 0) -> dict:
    """Paired difference a - b over perturbations (higher = better; flip the sign of MSE before calling).

    Rows of the same gene (e.g. the same gene in several test lines) are resampled together:
    95% percentile interval from B bootstrap resamples of genes, and a two-sided Wilcoxon signed-rank test
    on the per-gene mean differences."""
    d = pd.Series(np.asarray(a, float) - np.asarray(b, float)).groupby(np.asarray(genes)).agg(["sum", "count"])
    s, n = d["sum"].to_numpy(), d["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (B, len(s)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    gmean = s / n
    nz = gmean[gmean != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) > 10 else np.nan
    return dict(diff=float(s.sum() / n.sum()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                frac_better=float((gmean > 0).mean()), p_wilcoxon=p, n_genes=int(len(s)), n_rows=int(n.sum()))


def holm(p) -> np.ndarray:
    """Holm-adjusted p-values."""
    p = np.asarray(p, float)
    o = np.argsort(p); m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[o])
    out = np.empty(m); out[o] = np.minimum(adj, 1.0)
    return out


def paired_lw(a, b, genes, lines, B: int = 2000, seed: int = 0, chunk: int = 250) -> dict:
    """Line-weighted paired difference a - b (higher = better; flip the sign of MSE before calling).

    Estimand of the paper's tables: the mean difference within each cell line, averaged over lines with
    equal weight. Interval: 95% percentile bootstrap over genes (all rows of a gene, in every line, are resampled
    together; B resamples, seed 0). `pooled` is the plain mean over rows (the estimand of `paired`). Wilcoxon test on
    the per-gene mean differences. `per_line` lists the per-line mean differences; `lines_all_positive` says whether
    every line favors a."""
    d = np.asarray(a, float) - np.asarray(b, float)
    genes, lines = np.asarray(genes).astype(str), np.asarray(lines).astype(str)
    gu, gi = np.unique(genes, return_inverse=True)
    lu, li = np.unique(lines, return_inverse=True)
    S = np.zeros((len(gu), len(lu))); N = np.zeros((len(gu), len(lu)))
    np.add.at(S, (gi, li), d); np.add.at(N, (gi, li), 1.0)
    point = float(np.mean(S.sum(0) / N.sum(0)))
    idx = np.random.default_rng(seed).integers(0, len(gu), (B, len(gu)))
    bs = np.empty(B)
    for i in range(0, B, chunk):
        j = idx[i:i + chunk]
        bs[i:i + chunk] = (S[j].sum(1) / N[j].sum(1)).mean(1)
    gmean = S.sum(1) / N.sum(1)
    nz = gmean[gmean != 0]
    per_line = {l: float(S[:, k].sum() / N[:, k].sum()) for k, l in enumerate(lu)}
    frac_units = float(np.mean([np.mean(d[li == k] > 0) for k in range(len(lu))]))
    return dict(diff=point, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), pooled=float(d.mean()),
                frac_genes_better=float((gmean > 0).mean()), frac_units_better_lw=frac_units,
                p_wilcoxon=float(wilcoxon(nz).pvalue) if len(nz) > 10 else np.nan, n_genes=int(len(gu)), n_rows=int(len(d)),
                n_lines=int(len(lu)), lines_all_positive=bool(all(v > 0 for v in per_line.values())),
                per_line=";".join(f"{k}:{v:+.4f}" for k, v in per_line.items()))
