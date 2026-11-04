#!/usr/bin/env python3
"""Repeated outer splits of the held-out-gene tasks (Fig. 2; Supplement Section E, Tables 13-15).

Each task (unseen = random genes, cluster = GoBERT functional clusters, cluster_genept = GenePT clusters) was split
five times by the rule that produced it, with only the random seed changed (splits/repeat/<task>_s<k>.*; outer split
0 is the split of Table 1, identical to splits/heldout/<task>.*). In each split, 40% of the held-out genes are
selection genes (scored in RPE1) and 60% evaluation genes (scored in K562, HepG2 and Jurkat); libraries exclude
every held-out gene, as in scripts/predict_heldout.py.

Models per outer split:
  Pertrieve-Retrieval        weights/repeat/<task>_s<k>/pertrieve_s{0,1,2} (three seeds, metrics averaged over seeds)
  untrained initialization   weights/repeat/<task>_s<k>/soft_knn_init_s0 (soft GoBERT kNN, output scale fitted)
  kNN                        GoBERT kNN, k in {1,2,3,5,7,10,15,20,30,50}
  ridge                      Ahlmann-Eltze linear model on the four representations (pertrieve.baselines, K x lambda grid)
  SLIM-STRING                optional (--slim-string, needs SLIM): SLIM's released code through the adapter of
                             scripts/baseline_slim_heldout.py, (K, lambda) in {5,...,200} x {0.01,...,100}
kNN k and the ridge setting are chosen here on the RPE1 selection genes by Pearson delta + cosine discrimination.
SLIM uses the settings selected for the paper (splits/repeat/selection.json) unless --reselect repeats the grid
(NumPy seeded with 0 before every fit; about 4 min per split).

Score: response Pearson on the clipped prediction (pertrieve.metrics), mean per test line, then equal-weight mean over
K562, HepG2 and Jurkat. Paired differences Pertrieve-Retrieval minus each comparator: pertrieve.stats.paired_lw
within each split; across splits the outer split is the unit (mean, SD, min, max, number of positive splits).
Outputs results/repeat/{per_split,paired_per_split,model_means,across_splits}.csv.

  python scripts/repeat_heldout.py [--tasks unseen cluster cluster_genept] [--outer 0 1 2 3 4] [--slim-string <h5>]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, Pseudobulk, heldout_split, knn_predict, load_representations
from pertrieve.baselines import ahlmann_eltze, combine
from pertrieve.metrics import clip_delta, per_perturbation
from pertrieve.stats import paired_lw

REPO = Path(__file__).resolve().parents[1]
TASKS = ("unseen", "cluster", "cluster_genept")
TEST_LINES = ("k562", "hepg2", "jurkat")
FOUR = ("gobert", "genept", "esm2", "scgpt")
KS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50)
NAMES = {"pertrieve": "Pertrieve-Retrieval", "init": "Untrained initialization", "knn": "kNN (val.-selected k)",
         "ridge4": "Ahlmann-Eltze ridge (four repr.)", "slim": "SLIM-STRING (val.-selected)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--outer", nargs="+", type=int, default=list(range(5)), choices=range(5))
    ap.add_argument("--slim-string", default=None, help="SLIM's data/gene_string_embeddings.v0.3.h5; adds SLIM-STRING")
    ap.add_argument("--reselect", action="store_true", help="repeat SLIM's (K, lambda) selection instead of the paper's")
    ap.add_argument("--out", default=str(REPO / "results/repeat"))
    a = ap.parse_args()

    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    reps = dict(zip(FOUR, load_representations(Path(a.data) / "representations", FOUR)))
    stored = json.loads((REPO / "splits/repeat/selection.json").read_text())
    if a.slim_string:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import baseline_slim_heldout as bsh  # needs SLIM (pip install -e ".[slim]")
        from slim.config import SlimConfig
        from slim.model import BilinearModel
    crit = lambda P, R, ctrl: float(per_perturbation(clip_delta(P, ctrl), R)[["pearson_delta", "disc_cos"]].mean().sum())  # noqa: E731

    per_split, paired = [], []
    for task in a.tasks:
        for k in a.outer:
            name = f"{task}_s{k}"
            sp = heldout_split(REPO / "splits/repeat", name, "gd")

            def setup(c):
                lib = [g for g in pb.genes(c) if g not in sp["heldout"]]
                q = sorted(g for g in pb.genes(c) if g in sp["queries"][c])
                return lib, q, pb.control(c), pb.delta(c, lib), pb.context_mean(c, lib)

            def closed_form(c):
                lib, q, ctrl, D, cm = setup(c)
                knn = {kk: knn_predict(reps["gobert"].matrix(q), reps["gobert"].matrix(lib), D, kk, cm) for kk in KS}
                ae = ahlmann_eltze(combine([reps[r].matrix(lib) for r in FOUR]), D, combine([reps[r].matrix(q) for r in FOUR]))
                return knn, ae

            def slim_fit(c, K, lam):
                genes, ctrl = pb.genes(c), pb.control(c)
                lib, q = [g for g in genes if g not in sp["heldout"]], sorted(g for g in genes if g in sp["queries"][c])
                model = BilinearModel(a.slim_string, SlimConfig(condition_col="condition", control_tag=bsh.CONTROL, K=K, lambda_reg=lam))
                np.random.seed(0)
                model.fit(bsh.PseudobulkAdapter(pb.mean(c, genes), genes, ctrl, lib, q))
                pred = model.predict(q)
                return np.stack([pred[g] for g in q]) - ctrl

            # settings chosen on the RPE1 selection genes
            _, q, ctrl, _, _ = setup("rpe1"); R = pb.delta("rpe1", q)
            knn, ae = closed_form("rpe1")
            kb = max(KS, key=lambda kk: crit(knn[kk], R, ctrl))
            hb = max(ae, key=lambda h: crit(ae[h], R, ctrl))
            st = stored[task][str(k)]
            slim_set = (st["slim_K"], st["slim_lambda"])
            if a.slim_string and a.reselect:
                slim_set = max(((K, lam) for K in bsh.KS for lam in bsh.LAMBDAS), key=lambda s: crit(slim_fit("rpe1", *s), R, ctrl))
            setting = {"knn": str(kb), "ridge4": f"K={hb[0]} lambda={hb[1]:g}", "slim": f"K={slim_set[0]} lambda={slim_set[1]:g}"}
            if (kb, hb) != (st["knn_k"], (st["ridge_K"], st["ridge_lambda"])):
                print(f"  note: {name} selection differs from the paper's ({setting} vs {st})", flush=True)

            models = sorted((REPO / f"weights/repeat/{name}").glob("pertrieve_s[0-9].npz"))
            init = Pertrieve.load(REPO / f"weights/repeat/{name}/soft_knn_init_s0.npz")
            rows = {}
            for c in TEST_LINES:
                lib, q, ctrl, D, cm = setup(c)
                Q, Kr = [reps[r].matrix(q) for r in FOUR], [reps[r].matrix(lib) for r in FOUR]
                knn, ae = closed_form(c)
                preds = {"pertrieve": [Pertrieve.load(f).predict(Q, Kr, D, c) for f in models],
                         "init": [init.predict(Q, Kr, D, c)], "knn": [knn[kb]], "ridge4": [ae[hb]]}
                if a.slim_string:
                    preds["slim"] = [slim_fit(c, *slim_set)]
                R = pb.delta(c, q)
                for m, Ps in preds.items():
                    v = np.mean([per_perturbation(clip_delta(P, ctrl), R)["pearson_delta"].to_numpy() for P in Ps], 0)
                    rows.setdefault(m, []).append(pd.DataFrame(dict(line=c, gene=q, pearson_delta=v)))
            rows = {m: pd.concat(v, ignore_index=True) for m, v in rows.items()}
            for m, d in rows.items():
                per_split.append(dict(task=task, outer_split=k, model=NAMES[m], n_seeds=len(models) if m == "pertrieve" else 1,
                                      setting=setting.get(m, ""), pearson_delta=d.groupby("line").pearson_delta.mean().mean()))
            A = rows["pertrieve"]
            for m, B in rows.items():
                if m == "pertrieve":
                    continue
                assert (A.gene.values == B.gene.values).all() and (A.line.values == B.line.values).all()
                s = paired_lw(A.pearson_delta.to_numpy(), B.pearson_delta.to_numpy(), A.gene.to_numpy(), A.line.to_numpy())
                paired.append(dict(task=task, outer_split=k, vs=NAMES[m], diff=s["diff"], lo=s["lo"], hi=s["hi"],
                                   lines_all_positive=s["lines_all_positive"]))
            print(f"{name}: " + ", ".join(f"{NAMES[m]} {per_split[-len(rows) + i]['pearson_delta']:.4f}" for i, m in enumerate(rows))
                  + f" | kNN k = {kb}, ridge {setting['ridge4']}" + (f", SLIM {setting['slim']}" if a.slim_string else ""), flush=True)

    S, P = pd.DataFrame(per_split), pd.DataFrame(paired)
    M = S.groupby(["task", "model"], sort=False).pearson_delta.agg(["mean", "std", "size"]).reset_index()
    X = P.groupby(["task", "vs"], sort=False)["diff"].agg(mean="mean", sd="std", min="min", max="max",
                                                         n_positive=lambda x: int((x > 0).sum()), n_splits="size").reset_index()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S.to_csv(out / "per_split.csv", index=False); P.to_csv(out / "paired_per_split.csv", index=False)
    M.to_csv(out / "model_means.csv", index=False); X.to_csv(out / "across_splits.csv", index=False)
    pd.set_option("display.width", 200)
    print(M.round(4).to_string(index=False)); print(X.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
