#!/usr/bin/env python3
"""Hallmark set-score recovery by response-breadth stratum (Supplement Section J, Table 29).

MSigDB Hallmark 2020 sets with at least 15 genes in the 2,000-gene space (data/fewshot_bio/MSigDB_Hallmark_2020.txt; 38
sets). Set score of a perturbation = mean clipped pseudobulk change over the set's covered genes, the same formula for the
measured and the predicted response (a transcript-level summary, not a pathway-activity estimate). Per perturbation:
Spearman correlation of measured and predicted scores across the sets; sign agreement on the 5 sets with the largest
measured |score|; overlap of those 5 with the 5 largest predicted |scores|. Per set (Pertrieve and State): Pearson
across perturbations within each line, averaged over the four lines. Strata, models, seeds and statistics as
scripts/breadth_fewshot.py (run it first: its per_query.csv.gz provides the strata). Paired comparisons: Pertrieve minus
State and minus Linear on all three metrics and strata, and minus Linear + untrained retrieval on the Spearman over all
strata (Section 5.4; Supplement Table 65).
Outputs <out>/{per_query.csv.gz, per_set.csv, sets.csv, group_means.csv, group_means_per_line.csv, paired.csv}.

  python scripts/hallmark_fewshot.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pertrieve import Pseudobulk
from pertrieve.metrics import clip_delta
from pertrieve.stats import paired_lw

REPO = Path(__file__).resolve().parents[1]
LINES = ("k562", "rpe1", "hepg2", "jurkat")
MET = ["spearman", "top5_sign", "top5_overlap"]
REF = "Pertrieve"
PAIRED = {"State": {s: MET for s in ("narrow", "medium", "broad", "minimal", "ALL")},
          "Linear cross-cell-line": {s: MET for s in ("narrow", "medium", "broad", "minimal", "ALL")},
          "Linear + untrained retrieval": {"ALL": ["spearman"]}}
PAIRED_COLS = ["stratum", "vs", "metric", "n_units", "diff", "lo", "hi", "p_wilcoxon", "n_genes", "n_rows", "n_lines"]


def set_metrics(SP, ST):
    rows = []
    for i in range(len(ST)):
        t, p = ST[i], SP[i]; top = np.argsort(-np.abs(t))[:5]; ptop = set(np.argsort(-np.abs(p))[:5])
        rows.append(dict(spearman=spearmanr(t, p).correlation, top5_sign=float(np.mean(np.sign(p[top]) == np.sign(t[top]))),
                         top5_overlap=len(set(top) & ptop) / 5.0))
    return pd.DataFrame(rows)


def main():
    from breadth_fewshot import load_models  # same models and prediction files
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--pred", default=str(REPO / "predictions/fewshot"))
    ap.add_argument("--breadth", default=str(REPO / "results/fewshot_bio/breadth/per_query.csv.gz"))
    ap.add_argument("--out", default=str(REPO / "results/fewshot_bio/hallmark"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz"); bio = Path(a.data) / "fewshot_bio"; pred = Path(a.pred)
    hvg = np.loadtxt(bio / "hvg_genes.txt", dtype=str); gi = {g: i for i, g in enumerate(hvg)}
    SETS = {}
    for l in open(bio / "MSigDB_Hallmark_2020.txt"):
        p = l.rstrip("\n").split("\t"); idx = sorted({gi[g] for g in p[2:] if g in gi})
        if len(idx) >= 15:
            SETS[p[0]] = np.array(idx)
    names = list(SETS); M = np.zeros((len(hvg), len(names)))
    for j, n in enumerate(names):
        M[SETS[n], j] = 1.0 / len(SETS[n])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(set=names, covered=[len(SETS[n]) for n in names])).to_csv(out / "sets.csv", index=False)
    strata = pd.read_csv(a.breadth); strata = strata[strata.model == REF][["line", "gene", "stratum"]]
    per, perset = [], []
    for c in LINES:
        ctrl = pb.control(c); genes = list(np.load(pred / f"xline_s0_{c}.npz")["genes"].astype(str)); R = pb.delta(c, genes); ST = R @ M
        for name, Ps in load_models(pb, pred, bio, c, genes, ctrl).items():
            SPs = [clip_delta(P, ctrl) @ M for P in Ps]
            m = sum(set_metrics(SP, ST) for SP in SPs) / len(SPs); per.append(m.assign(model=name, line=c, gene=genes))
            SPm = sum(SPs) / len(SPs)
            for j, n in enumerate(names):
                perset.append(dict(line=c, model=name, set=n, covered=len(SETS[n]),
                                   pearson_across_perts=(np.corrcoef(ST[:, j], SPm[:, j])[0, 1] if SPm[:, j].std() > 1e-9 else np.nan),
                                   sign_agree=float(np.mean(np.sign(SPm[:, j]) == np.sign(ST[:, j])))))
        print(f"{c}: {len(genes)} test perturbations, {len(names)} sets", flush=True)
    per = pd.concat(per, ignore_index=True).merge(strata, on=["line", "gene"]); per.to_csv(out / "per_query.csv.gz", index=False)
    PS = pd.DataFrame(perset); PS = PS[PS.model.isin([REF, "State"])].groupby(["model", "set", "covered"])[["pearson_across_perts", "sign_agree"]].mean()
    PS.reset_index().to_csv(out / "per_set.csv", index=False)  # mean over the four lines
    G = per.groupby(["stratum", "model", "line"])[MET].mean().reset_index(); G.to_csv(out / "group_means_per_line.csv", index=False)
    Mn = G.groupby(["stratum", "model"])[MET].mean().reset_index()
    A = per.groupby(["model", "line"])[MET].mean().groupby("model").mean().reset_index().assign(stratum="ALL")
    Mn = pd.concat([Mn, A]); Mn.to_csv(out / "group_means.csv", index=False)
    T = []
    Ap = per[per.model == REF].set_index(["line", "gene"]).sort_index()
    for other, by_stratum in PAIRED.items():
        Bm = per[per.model == other].set_index(["line", "gene"]).sort_index(); assert Ap.index.equals(Bm.index)
        for s, mets in by_stratum.items():
            m = np.ones(len(Ap), bool) if s == "ALL" else (Ap.stratum == s).to_numpy()
            for met in mets:
                x, y = Ap[met].to_numpy()[m], Bm[met].to_numpy()[m]; ok = np.isfinite(x) & np.isfinite(y)
                T.append(dict(stratum=s, vs=other, metric=met, n_units=int(ok.sum()),
                              **paired_lw(x[ok], y[ok], Ap.index.get_level_values(1)[m][ok], Ap.index.get_level_values(0)[m][ok])))
    T = pd.DataFrame(T)[PAIRED_COLS]; T.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(Mn[Mn.model.isin(["State", "Linear cross-cell-line", REF])].round(3).to_string(index=False))
    print(T[T.vs.isin(["State", "Linear cross-cell-line"])][["stratum", "vs", "metric", "n_units", "diff", "lo", "hi"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
