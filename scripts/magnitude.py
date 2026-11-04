#!/usr/bin/env python3
"""Per-query magnitude (Section 3.3; Supplement Section A.2 "Per-query magnitude" and Table 65).

The output scale gamma_c is one scalar per line, but the norm of the unscaled prediction depends on the references a
query retrieves. Magnitude = L2 norm over the 2,000 genes of the clipped predicted response and of the measured
response. Per line: norms averaged over seeds, then the Spearman correlation across the test perturbations; the result
is the equal-weight mean over lines.
  few-shot   official test sets, three seeds: Pertrieve (xline+ae_xline), its linear predictor (ae_xline) and its
             retrieval component (xline), from predictions/fewshot/ (written by the few-shot steps of reproduce_other.sh)
  held-out   Pertrieve-Retrieval seeds 0-2 on the evaluation genes of the three splits of Table 1, from
             predictions/heldout/gd/<split>/ (written by reproduce_heldout.sh)
Differences Pertrieve - component: 95% percentile bootstrap over genes (all rows of a gene, in every line, resampled
together; B = 1,000, seed 0) of the line-mean Spearman difference.
Outputs results/magnitude/{summary,paired}.csv.

  python scripts/magnitude.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pertrieve import Pseudobulk
from pertrieve.metrics import clip_delta

REPO = Path(__file__).resolve().parents[1]
FEWSHOT = {"Pertrieve": "xline+ae_xline", "Linear predictor (ae_xline)": "ae_xline", "Cross-cell-line retrieval (xline)": "xline"}
FS_LINES = ("k562", "rpe1", "hepg2", "jurkat")
HO_LINES = ("k562", "hepg2", "jurkat")


def line_mean_spearman(d, col):
    return float(np.mean([spearmanr(x[col], x.true)[0] for _, x in d.groupby("line", sort=False)]))


def boot(d, a, b, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    genes = d.gene.unique(); idx = {g: np.flatnonzero(d.gene.values == g) for g in genes}
    diff = lambda x: line_mean_spearman(x, a) - line_mean_spearman(x, b)  # noqa: E731
    bs = [diff(d.iloc[np.concatenate([idx[g] for g in rng.choice(genes, len(genes))])]) for _ in range(B)]
    return diff(d), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--pred", default=str(REPO / "predictions"))
    ap.add_argument("--out", default=str(REPO / "results/magnitude"))
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    pred = Path(a.pred)

    def norms(f, genes, ctrl):
        z = np.load(f); assert list(z["genes"].astype(str)) == genes, f
        return np.linalg.norm(clip_delta(z["delta"].astype(np.float64), ctrl), axis=1)

    sets = {}
    rows = []
    for c in FS_LINES:
        genes = list(np.load(pred / f"fewshot/xline_s0_{c}.npz")["genes"].astype(str)); ctrl = pb.control(c)
        d = pd.DataFrame(dict(gene=genes, line=c, true=np.linalg.norm(pb.delta(c, genes), axis=1)))
        for name, v in FEWSHOT.items():
            d[name] = np.mean([norms(pred / f"fewshot/{v}_s{s}_{c}.npz", genes, ctrl) for s in range(3)], 0)
        rows.append(d)
    sets["fewshot (official test sets)"] = pd.concat(rows, ignore_index=True)
    for split in ("unseen", "cluster", "cluster_genept"):
        rows = []
        for c in HO_LINES:
            genes = list(np.load(pred / f"heldout/gd/{split}/pertrieve_s0_{c}.npz")["genes"].astype(str)); ctrl = pb.control(c)
            d = pd.DataFrame(dict(gene=genes, line=c, true=np.linalg.norm(pb.delta(c, genes), axis=1)))
            d["Pertrieve-Retrieval"] = np.mean([norms(pred / f"heldout/gd/{split}/pertrieve_s{s}_{c}.npz", genes, ctrl)
                                                for s in range(3)], 0)
            rows.append(d)
        sets[f"held-out ({split})"] = pd.concat(rows, ignore_index=True)

    S = pd.DataFrame([dict(task=t, model=m, spearman_norm=line_mean_spearman(d, m))
                      for t, d in sets.items() for m in d.columns if m not in ("gene", "line", "true")])
    d = sets["fewshot (official test sets)"]
    P = pd.DataFrame([dict(task="fewshot (official test sets)", model="Pertrieve", vs=b,
                           **dict(zip(("diff", "lo", "hi"), boot(d, "Pertrieve", b)))) for b in list(FEWSHOT)[1:]])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "summary.csv", index=False); P.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.round(3).to_string(index=False)); print(P.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
