#!/usr/bin/env python3
"""Pertrieve vs PT-RAG (Di Francesco et al. 2026) on PT-RAG's own few-shot splits and gene space (Supplement Section I,
Table 23; PT-RAG rows of Table 7).

Split (splits/ptrag/<line>.{test,val}.txt, from the PT-RAG repository's datasets/repogle_nadig*.toml): in the
target line the test perturbations are held out and the val perturbations are used only for model selection; the
library of line c is every other perturbation of c. Scored perturbations = those PT-RAG released predictions for
(test genes with data in that line, minus a few without a GenePT vector).
Data (see DATA.md): data/ptrag/pseudobulk.npz = means of replogle.h5ad X over all cells for PT-RAG's 2,000 output
genes (equal to the means of PT-RAG's adata_real); data/ptrag/ptrag_predictions/<line>.npz = PT-RAG's released test
predictions (HF difra00/PT-RAG, eval_last.ckpt/adata_pred.h5ad), averaged over predicted cells per perturbation.
Methods (the rows of Table 23): PT-RAG; the context mean; Pertrieve-Retrieval with the cross-cell-line profile
(weights/ptrag/<line>/xline_s<seed>, 3 seeds, metrics averaged over seeds per perturbation); and Pertrieve, the
retrieval fused with the cross-cell-line linear model on PT-RAG's split (test predictions written by
scripts/confirm_fusion.py --task ptrag to predictions/fewshot_confirm/ptrag/ae+xline_s<seed>_<line>.npz, scored here
when present). PT-RAG generates cells, so its response is the mean of its predicted cells minus the mean of its
predicted control cells (the cell-eval convention used for State; Supplement Section B.1); "PT-RAG (measured control)"
subtracts the measured control instead (Supplement Table 7). Scoring as for the few-shot benchmark (pertrieve.metrics
on clip(delta + ctrl, 0)). Paired statistics, Pertrieve minus PT-RAG: pertrieve.stats.paired_lw (line-weighted;
perturbations of the same gene resampled together), Holm over the four metrics.
Outputs results/ptrag/{per_perturbation,per_line,summary,paired}.csv.

  python scripts/run_ptrag.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, Pseudobulk, load_representations
from pertrieve.data import CrossLineProfile, read_genes
from pertrieve.metrics import clip_delta, per_perturbation
from pertrieve.stats import holm, paired_lw

REPO = Path(__file__).resolve().parents[1]
LINES = ("k562", "rpe1", "hepg2", "jurkat")
MET = ["pearson_delta", "disc_cos", "disc_l2", "mse"]
RETRIEVAL, FUSED = "Pertrieve-Retrieval, cross-cell-line", "Pertrieve"
PAIRED_COLS = ["model", "vs", "metric", "diff", "lo", "hi", "p_wilcoxon", "p_holm", "n_genes", "n_rows", "n_lines"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--out", default=str(REPO / "results/ptrag"))
    ap.add_argument("--fused", default=str(REPO / "predictions/fewshot_confirm/ptrag"), help="fused predictions of confirm_fusion.py")
    a = ap.parse_args()
    d = Path(a.data)
    pb = Pseudobulk(d / "ptrag/pseudobulk.npz")
    rep_dir = d / "representations"
    rows = []
    for c in LINES:
        held, val_all = set(read_genes(REPO / f"splits/ptrag/{c}.test.txt")), set(read_genes(REPO / f"splits/ptrag/{c}.val.txt"))
        z = np.load(d / f"ptrag/ptrag_predictions/{c}.npz", allow_pickle=True)
        zp = [str(p) for p in z["perts"]]
        genes = set(pb.genes(c))
        test = sorted(p for p in zp if p != "non-targeting")
        assert set(test) <= held & genes
        lib = sorted(genes - held - val_all)
        mu, D, real = pb.control(c), pb.delta(c, lib), pb.delta(c, test)
        n = pb.count(c, lib).astype(float)
        cm = (pb.mean(c, lib) * n[:, None]).sum(0) / n.sum() - mu  # cell-weighted mean profile of the library
        pi = {p: i for i, p in enumerate(zp)}
        pm = z["pred_mean"].astype(np.float64)
        preds = {"PT-RAG": [pm[[pi[p] for p in test]] - pm[pi["non-targeting"]]],  # PT-RAG's own predicted control
                 "PT-RAG (measured control)": [pm[[pi[p] for p in test]] - mu],
                 "context mean": [np.tile(cm, (len(test), 1))], RETRIEVAL: []}
        for f in sorted((REPO / "weights/ptrag" / c).glob("xline_s[0-9].npz")):
            m = Pertrieve.load(f)
            gene_reps = iter(load_representations(rep_dir, [x for x in m.representations if x != "xline_profile"]))
            reps = [CrossLineProfile(pb, c, LINES) if x == "xline_profile" else next(gene_reps) for x in m.representations]
            preds[RETRIEVAL].append(m.predict([r.matrix(test) for r in reps], [r.matrix(lib) for r in reps], D, c))
        fused = sorted(Path(a.fused).glob(f"ae+xline_s[0-9]_{c}.npz"))
        for f in fused:
            z = np.load(f); fi = {str(g): i for i, g in enumerate(z["genes"])}
            preds.setdefault(FUSED, []).append(z["delta"][[fi[p] for p in test]].astype(np.float64))
        print(f"[{c}] scored {len(test)}, library {len(lib)}; fused seeds {len(fused)}", flush=True)
        for name, Ps in preds.items():
            ms = [per_perturbation(clip_delta(X, mu), real)[MET] for X in Ps]
            rows.append((sum(ms) / len(ms)).assign(line=c, method=name, perturbation=test, n_seeds=len(Ps)))
    R = pd.concat(rows, ignore_index=True)
    R.loc[R.method == "PT-RAG (measured control)", "disc_l2"] = np.nan  # Table 7 reports Pearson, cosine discrimination, MSE
    per_line = R.groupby(["method", "line"])[MET].mean()
    summ = per_line.groupby("method").mean()  # each line weighted equally
    summ["n_perturbations"] = R.groupby("method").size()
    P = []
    if FUSED in set(R.method):
        A = R[R.method == FUSED].set_index(["line", "perturbation"])
        B = R[R.method == "PT-RAG"].set_index(["line", "perturbation"]).loc[A.index]
        for m in MET:
            sgn = -1 if m == "mse" else 1
            st = paired_lw(sgn * A[m].to_numpy(), sgn * B[m].to_numpy(), A.index.get_level_values("perturbation"),
                           A.index.get_level_values("line"))
            if m == "mse":
                st.update(diff=-st["diff"], lo=-st["hi"], hi=-st["lo"])
            P.append(dict(model=FUSED, vs="PT-RAG", metric=m, **st))
    P = pd.DataFrame(P, columns=PAIRED_COLS)
    if len(P):
        P["p_holm"] = holm(P.p_wilcoxon.fillna(1).to_numpy())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "per_perturbation.csv", index=False); per_line.to_csv(out / "per_line.csv")
    summ.to_csv(out / "summary.csv"); P[PAIRED_COLS].to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
    print(P[["model", "vs", "metric", "diff", "lo", "hi", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
