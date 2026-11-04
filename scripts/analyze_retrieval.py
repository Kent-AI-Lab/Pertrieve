#!/usr/bin/env python3
"""What does Pertrieve retrieve? Retrieval weights of the held-out-gene models (separate selection genes) before and after training.

For every evaluation gene (K562, HepG2, Jurkat), the weights of the trained model (mean over seeds) and of
its untrained initialization (soft GoBERT kNN) are compared on annotations the model never uses:
  Complex Portal human complexes   https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv
  STRING v12 physical links        https://stringdb-downloads.org/download/protein.physical.links.v12.0/9606.protein.physical.links.v12.0.txt.gz
                                   https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
  UniProt -> gene symbol           https://current.geneontology.org/annotations/goa_human.gaf.gz
  HGNC complete set                https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
The expression data use older gene symbols in places (TARS, CARS, ZNRD1, ...); before partners are matched, every symbol
is mapped to its current HGNC symbol (itself if current, else the unique current symbol listing it as a previous symbol,
else as an alias). Reported (Supplement Tables 53-54): weight on annotated partners (complex members; STRING score
>= 700) for queries with at least one partner in the library, their share of the library, and the measured response
correlation of the five highest-weighted references; means and paired differences are line-weighted (mean within each
line, averaged over lines; pertrieve.stats.paired_lw). Also the top references of selected genes.
Outputs <out>/{per_query,summary,top_references}.csv.

  python scripts/analyze_retrieval.py --annotations annotations/ --genes k562:POLR1B k562:MED7 jurkat:CDC123
"""
import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, Pseudobulk, heldout_split, load_representations
from pertrieve.stats import paired_lw

LINES = ("k562", "hepg2", "jurkat")
REPO = Path(__file__).resolve().parents[1]


def symbol_map(path: Path):
    """canon(symbol) -> current HGNC symbol (itself if current, else the unique current symbol listing it as a previous
    symbol, else as an alias, else unchanged)."""
    h = pd.read_csv(path, sep="\t", dtype=str, usecols=["symbol", "alias_symbol", "prev_symbol", "status"], low_memory=False)
    h = h[h.status == "Approved"]
    current = set(h.symbol)
    maps = []
    for col in ("prev_symbol", "alias_symbol"):
        d = defaultdict(set)
        for sym, vals in zip(h.symbol, h[col].fillna("")):
            for v in vals.split("|"):
                v = v.strip()
                if v and v not in current:
                    d[v].add(sym)
        maps.append({k: next(iter(v)) for k, v in d.items() if len(v) == 1})
    prev, alias = maps
    return lambda s: s if s in current else prev.get(s) or alias.get(s) or s


def annotations(d: Path):
    acc2sym = {}
    with gzip.open(d / "goa_human.gaf.gz", "rt") as f:
        for ln in f:
            if ln[0] != "!":
                x = ln.split("\t"); acc2sym.setdefault(x[1], x[2])
    comp, cname = defaultdict(set), {}
    for _, r in pd.read_csv(d / "9606.tsv", sep="\t", dtype=str).iterrows():
        mem = {acc2sym.get(m.split("(")[0].split("-")[0]) for m in str(r.iloc[4]).split("|")} - {None}
        if len(mem) >= 2:
            cname[r.iloc[0]] = r.iloc[1]
            for g in mem:
                comp[g].add(r.iloc[0])
    info = pd.read_csv(d / "9606.protein.info.v12.0.txt.gz", sep="\t", usecols=[0, 1])
    sym = dict(zip(info.iloc[:, 0], info.iloc[:, 1]))
    L = pd.read_csv(d / "9606.protein.physical.links.v12.0.txt.gz", sep=" ")
    L = L[L.combined_score >= 700]
    partners = defaultdict(set)
    for a, b in zip(L.protein1.map(sym), L.protein2.map(sym)):
        partners[a].add(b); partners[b].add(a)
    return comp, cname, partners


def corr_rows(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-30; B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-30
    return A @ B.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--annotations", required=True, help="directory with the four annotation files listed above")
    ap.add_argument("--splits", nargs="+", default=["unseen", "cluster", "cluster_genept"])
    ap.add_argument("--genes", nargs="*", default=["jurkat:CDC123", "k562:GFER", "k562:POLR1B", "k562:MED7", "k562:RFC3"],
                    help="line:gene pairs (random-gene split) whose top references are listed")
    ap.add_argument("--out", default=str(REPO / "results/retrieval_analysis"))
    a = ap.parse_args()
    comp, cname, partners = annotations(Path(a.annotations))
    canon = symbol_map(Path(a.annotations) / "hgnc_complete_set.txt")
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    rows, top = [], []
    for split in a.splits:
        sp = heldout_split(REPO / "splits/heldout", split, "gd")
        wd = REPO / "weights/heldout/gd" / split
        models = [Pertrieve.load(f) for f in sorted(wd.glob("pertrieve_s[0-9].npz"))]
        init = Pertrieve.load(wd / "soft_knn_init_s0.npz")
        reps = load_representations(Path(a.data) / "representations", models[0].representations)
        for c in LINES:
            lib = [g for g in pb.genes(c) if g not in sp["heldout"]]
            q = sorted(g for g in pb.genes(c) if g in sp["queries"][c])
            K = [r.matrix(lib) for r in reps]; Q = [r.matrix(q) for r in reps]
            At = np.mean([m.weights(Q, K) for m in models], 0); Ai = init.weights(Q, K)
            C = corr_rows(pb.delta(c, q), pb.delta(c, lib))
            libc = [canon(k) for k in lib]
            for i, g in enumerate(q):
                gc = canon(g)
                cm = np.array([bool(comp.get(gc, set()) & comp.get(k, set())) for k in libc])
                st = np.array([k in partners.get(gc, set()) for k in libc])
                ti, tt = np.argsort(-Ai[i])[:5], np.argsort(-At[i])[:5]
                rows.append(dict(split=split, line=c, gene=g, n_complex=int(cm.sum()), n_string=int(st.sum()),
                                 mass_complex_trained=At[i][cm].sum(), mass_complex_init=Ai[i][cm].sum(), share_complex=cm.mean(),
                                 mass_string_trained=At[i][st].sum(), mass_string_init=Ai[i][st].sum(), share_string=st.mean(),
                                 top5_corr_trained=C[i, tt].mean(), top5_corr_init=C[i, ti].mean()))
                if split == "unseen" and f"{c}:{g}" in a.genes:
                    for model, A, o in (("untrained", Ai, ti), ("trained", At, tt)):
                        for r, k in enumerate(o):
                            shared = sorted(comp.get(gc, set()) & comp.get(libc[k], set()))
                            top.append(dict(line=c, gene=g, model=model, rank=r + 1, reference=lib[k], weight=A[i, k],
                                            response_corr=C[i, k], string_partner=bool(st[k]),
                                            shared_complex="; ".join(cname[x] for x in shared)))
    D = pd.DataFrame(rows)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "per_query.csv", index=False); pd.DataFrame(top).to_csv(out / "top_references.csv", index=False)
    summ = []
    lw = lambda x, col: float(x.groupby("line")[col].mean().mean())  # noqa: E731  line-weighted mean
    for split, d in D.groupby("split", sort=False):
        for q, x, t, u, share in [("top-5 response correlation", d, "top5_corr_trained", "top5_corr_init", None)] + [
                (f"weight on {ann} partners", d[d[f"n_{ann}"] > 0], f"mass_{ann}_trained", f"mass_{ann}_init", f"share_{ann}")
                for ann in ("complex", "string")]:
            g = paired_lw(x[t].to_numpy(), x[u].to_numpy(), x.gene.to_numpy(), x.line.to_numpy())
            summ.append(dict(split=split, quantity=q, trained=lw(x, t), untrained=lw(x, u),
                             library_share=lw(x, share) if share else np.nan,
                             **{k: g[k] for k in ("diff", "lo", "hi", "p_wilcoxon", "n_genes", "n_rows")}))
    S = pd.DataFrame(summ); S.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(S[["split", "quantity", "trained", "untrained", "library_share", "diff", "lo", "hi", "n_genes"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
