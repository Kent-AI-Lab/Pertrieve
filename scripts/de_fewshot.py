#!/usr/bin/env python3
"""Differential-expression recovery on State's few-shot test sets after validation calibration (Supplement Table 20).

Pertrieve predicts mean responses; for DE they are turned into pseudo-cells and scored with the official cell-eval
DE metrics, as State's predictions are.
  pseudo-cells  for a perturbation with n measured cells, n control cells of the line are drawn (seeded) and every
                feature is multiplied by t_f / m_f (t = clip(gain * delta + control mean, 0), m = mean of the drawn
                cells), so zeros stay zero and the pseudobulk equals t; features with m_f = 0 < t_f get + t_f.
  cells         all cells of the evaluated genes plus all controls of the line, obsm['X_hvg'] clipped to [0, 14],
                from State's preprocessed Replogle/Nadig AnnData (--h5ad).
  scoring       cell-eval profile "full" with the settings of `state tx predict` (pdex, exp_post_agg, log1p);
                FDR by one Benjamini-Hochberg over the whole perturbation x gene table, the convention of the
                official State results (cell-eval/pdex >= 0.3 apply BH per perturbation).
  calibration   the gain is chosen per line, seed and variant on the seeded validation genes, from the
                validation fits (predictions/fewshot/val/: scripts/predict_fewshot.py --val, and for the primary
                fused model xline+ae_xline scripts/fuse_fewshot.py with the test fusion weight), by the largest
                overlap_at_N + pr_auc over (1, 1.25, 1.5, 2, 2.5, 3, 4) (ties -> smaller gain). Test genes are
                not used.
Steps: --step calibrate writes <out>/calibration.csv and <out>/gains.csv; --step test scores the test predictions
with those gains and writes <out>/per_run.csv, per_line.csv, summary.csv (means over the four cell lines). Requires cell-eval.

  python scripts/de_fewshot.py --h5ad replogle.h5ad --step calibrate
  python scripts/de_fewshot.py --h5ad replogle.h5ad --step test
"""
import argparse
import shutil
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from pertrieve import Pseudobulk
from pertrieve.data import CONTROL, read_genes

LINES = ("k562", "rpe1", "hepg2", "jurkat")
SEEDS = (0, 1, 2)
GAINS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
SKIP = ["pearson_edistance", "clustering_agreement"]
TABLE = ["overlap_at_N", "precision_at_N", "pr_auc", "roc_auc", "de_direction_match", "de_spearman_lfc_sig", "mse",
         "de_nsig_counts_pred"]  # last: genes called significant per perturbation (#DE column of the Supplement)
REPO = Path(__file__).resolve().parents[1]


def read_cells(h5ad, line, genes):
    """All cells of `genes` + all controls of `line`; obs (gene, cell_line, gem_group) and X_hvg clipped [0, 14]."""
    with h5py.File(h5ad, "r") as f:
        def cat(k):
            g = f[f"obs/{k}"]
            c = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]])
            return c[g["codes"][:]]
        cl, gene, gem = cat("cell_line"), cat("gene"), f["obs/gem_group"][:]
        rows = np.flatnonzero((cl == line) & (np.isin(gene, list(genes)) | (gene == CONTROL)))
        H = f["obsm/X_hvg"]
        X = np.empty((len(rows), H.shape[1]), np.float32)
        pos, step = 0, 50000
        for s in range(rows[0], rows[-1] + 1, step):
            sel = rows[(rows >= s) & (rows < s + step)]
            if len(sel):
                X[pos: pos + len(sel)] = H[s: s + step][sel - s]
                pos += len(sel)
    np.clip(X, 0.0, 14.0, out=X)
    obs = pd.DataFrame({"gene": gene[rows], "cell_line": line, "gem_group": gem[rows].astype(str)})
    obs.index = obs.index.astype(str)
    return obs, X


def pseudo_cells(obs, X, genes, T, seed=0):
    """Pseudo-cells whose per-perturbation mean equals T (rows aligned with `genes`); control rows unchanged."""
    ctrl = np.flatnonzero(obs["gene"].to_numpy() == CONTROL)
    Xc, P = X[ctrl], X.copy()
    gidx = {g: i for i, g in enumerate(genes)}
    codes, uniq = pd.factorize(obs["gene"].to_numpy())
    line = obs["cell_line"].iloc[0]
    for j, g in enumerate(uniq):
        if g == CONTROL:
            continue
        r = np.flatnonzero(codes == j)
        t = T[gidx[g]].astype(np.float64)
        rng = np.random.default_rng([seed, abs(zlib.crc32(line.encode())), abs(zlib.crc32(g.encode()))])
        S = Xc[rng.integers(0, len(ctrl), size=len(r))].astype(np.float64)
        m = S.mean(0); pos = m > 0
        S[:, pos] *= t[pos] / m[pos]
        S[:, ~pos] += t[~pos]
        P[r] = np.clip(S, 0, 14).astype(np.float32)
    return P


def global_bh_metrics(outdir, line):
    """DE metrics with one BH over the whole table (official convention); other metrics from cell-eval."""
    import polars as pl
    from cell_eval._pipeline import MetricPipeline
    from cell_eval._types import initialize_de_comparison
    from scipy.stats import false_discovery_control

    schema = {"target": pl.Utf8, "feature": pl.Utf8}

    def bh(frame):
        p = frame["p_value"].to_numpy().astype(np.float64); fdr = np.full_like(p, np.nan); ok = np.isfinite(p)
        fdr[ok] = false_discovery_control(p[ok])
        return frame.with_columns(pl.Series("fdr", fdr))
    comp = initialize_de_comparison(real=bh(pl.read_csv(outdir / f"{line}_real_de.csv", schema_overrides=schema)),
                                    pred=bh(pl.read_csv(outdir / f"{line}_pred_de.csv", schema_overrides=schema)))
    pipe = MetricPipeline(profile="full"); pipe.compute_de_metrics(comp)
    de = pipe.get_agg_results().to_pandas().set_index("statistic").loc["mean"]
    agg = pd.read_csv(outdir / f"{line}_agg_results.csv", index_col=0).loc["mean"]
    return {k: float(de[k]) if k in de.index else float(agg[k]) for k in agg.index}


def evaluate(obs, X, genes, T, real_de, workdir, line, threads):
    """cell-eval of pseudo-cells for target profiles T; real DE read from (or written to) `real_de`."""
    import anndata as ad
    from cell_eval import MetricsEvaluator

    P = pseudo_cells(obs, X, genes, T)
    workdir.mkdir(parents=True, exist_ok=True)
    ev = MetricsEvaluator(adata_pred=ad.AnnData(X=P, obs=obs.copy()), adata_real=ad.AnnData(X=X, obs=obs),
                          de_real=str(real_de) if real_de.exists() else None, control_pert=CONTROL, pert_col="gene",
                          outdir=str(workdir), prefix=line, pdex_kwargs=dict(exp_post_agg=True, is_log1p=True),
                          num_threads=threads)
    ev.compute(profile="full", metric_configs={}, skip_metrics=SKIP)
    if not real_de.exists():
        real_de.parent.mkdir(parents=True, exist_ok=True); shutil.copy(workdir / f"{line}_real_de.csv", real_de)
    if not (workdir / f"{line}_real_de.csv").exists():
        shutil.copy(real_de, workdir / f"{line}_real_de.csv")
    m = global_bh_metrics(workdir, line)
    shutil.rmtree(workdir)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True, help="State's preprocessed Replogle/Nadig AnnData (replogle.h5ad)")
    ap.add_argument("--step", choices=["calibrate", "test"], required=True)
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--pred", default=str(REPO / "predictions/fewshot"))
    ap.add_argument("--variants", nargs="+", default=["xline+ae_xline", "xline", "xonly"],
                    help="xline+ae_xline = primary fused model (needs scripts/fuse_fewshot.py first)")
    ap.add_argument("--lines", nargs="+", default=list(LINES))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "results/fewshot_de"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "_".join(a.lines)
    rows = []
    for c in a.lines:
        ctrl = pb.control(c)
        if a.step == "calibrate":
            for s in SEEDS:
                genes = read_genes(REPO / f"splits/fewshot/{c}.val_s{s}.txt")
                obs, X = read_cells(a.h5ad, c, genes)
                assert (obs.groupby("gene").size().loc[genes].to_numpy() == pb.count(c, genes)).all()
                for v in a.variants:
                    z = np.load(Path(a.pred) / "val" / f"{v}_s{s}_{c}.npz")
                    assert list(z["genes"].astype(str)) == genes
                    for gain in GAINS:
                        T = np.clip(gain * z["delta"].astype(np.float64) + ctrl, 0, None)
                        m = evaluate(obs, X, genes, T, out / "real_de" / f"{c}_val_s{s}_real_de.csv",
                                     out / "_tmp" / f"{c}_{v}_s{s}_{gain:g}", c, a.threads)
                        rows.append(dict(line=c, seed=s, variant=v, gain=gain, n_val=len(genes), **{k: m[k] for k in TABLE + ["pearson_delta"]}))
                        print(f"{c} s{s} {v} gain {gain:g}: overlap {m['overlap_at_N']:.4f} pr_auc {m['pr_auc']:.4f}", flush=True)
                        pd.DataFrame(rows).to_csv(out / f"calibration_{tag}.csv", index=False)
        else:
            gains = pd.concat([pd.read_csv(f) for f in sorted(out.glob("gains_*.csv"))]).drop_duplicates(
                ["line", "seed", "variant"], keep="last").set_index(["line", "seed", "variant"]).gain
            genes = sorted(g for g in pb.genes(c) if g in set(read_genes(REPO / f"splits/fewshot/{c}.test.txt")))
            obs, X = read_cells(a.h5ad, c, genes)
            assert (obs.groupby("gene").size().loc[genes].to_numpy() == pb.count(c, genes)).all()
            for v in a.variants:
                for s in SEEDS:
                    z = np.load(Path(a.pred) / f"{v}_s{s}_{c}.npz")
                    assert list(z["genes"].astype(str)) == genes
                    g = float(gains.loc[(c, s, v)])
                    T = np.clip(g * z["delta"].astype(np.float64) + ctrl, 0, None)
                    m = evaluate(obs, X, genes, T, out / "real_de" / f"{c}_test_real_de.csv", out / "_tmp" / f"{c}_{v}_s{s}", c, a.threads)
                    rows.append(dict(line=c, variant=v, seed=s, gain=g, **{k: m[k] for k in TABLE + ["pearson_delta", "discrimination_score_l2"]}))
                    print(f"{c} {v} s{s} gain {g:g}: " + " ".join(f"{k} {m[k]:.4f}" for k in TABLE), flush=True)
                    pd.DataFrame(rows).to_csv(out / f"per_run_{tag}.csv", index=False)
    if a.step == "calibrate":
        df = pd.DataFrame(rows); df["score"] = df.overlap_at_N + df.pr_auc
        ch = df.loc[df.sort_values("gain").groupby(["line", "seed", "variant"])["score"].idxmax()]
        ch[["line", "seed", "variant", "gain", "score"]].to_csv(out / f"gains_{tag}.csv", index=False)
        print(ch[["line", "seed", "variant", "gain"]].to_string(index=False))
    else:
        R = pd.concat([pd.read_csv(f) for f in sorted(out.glob("per_run_*.csv"))]).drop_duplicates(["line", "variant", "seed"], keep="last")
        R.to_csv(out / "per_run.csv", index=False)
        PL = R.groupby(["variant", "line"])[TABLE].mean(); PL.to_csv(out / "per_line.csv")
        S = PL.groupby("variant").mean(); S.to_csv(out / "summary.csv")  # LFC rho: lines where it is defined
        print(S.round(3).to_string())


if __name__ == "__main__":
    main()
