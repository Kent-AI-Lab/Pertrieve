#!/usr/bin/env python3
"""Fusion confirmation under the protocol fixed before testing (Table 4, columns "PT-RAG split" and "CRISPRa";
Supplementary Table 21): the validation-selected fusion of Pertrieve (cross-cell-line variant xline) with the linear
cross-cell-line model, re-run on two further settings after the official few-shot results had been seen.

  --task ptrag   PT-RAG's few-shot split of the Replogle/Nadig screens (2,000 output genes, four lines): test =
                 the perturbations PT-RAG released predictions for, PT-RAG's val genes never enter the library; one
                 model per (line, seed) (weights/ptrag/<line>/xline_s<seed>, the models of scripts/run_ptrag.py).
  --task kaden   CRISPRa transcription-factor screen (Kaden et al. 2025; fibroblast and RPE1, 1,836 factors measured
                 in both lines, 4,985 genes, CellSimBench processing; data/fewshot_confirm/kaden). Five folds per
                 line (splits/fewshot_confirm/kaden): a fold is held out in the target line and its factors are
                 predicted from the other line; per seed 10% of the fold's library are validation genes (the
                 validation fit weights/fewshot_confirm/kaden/<line>/fold<k>/val/xline_s<seed> chooses the epoch, the
                 released model .../fold<k>/xline_s<seed> is the refit on the full library).
Models: xline (Pertrieve-Retrieval with the cross-cell-line profile), xline0 (the same model untrained: cosine weights on the
profile and GoBERT, output scale by leave-one-out least squares, as scripts/predict_fewshot.py --variant xline0),
ae_xline and ridge_xline (closed-form linear cross-cell-line models, setting chosen on the validation genes by Pearson delta
+ cosine discrimination), and the fusions ae+xline (= Pertrieve, the paper's few-shot model), ae+ridge, ae+xline0:
clip(a * linear + (1 - a) * second member), one a per seed from {0, 0.1, ..., 1} chosen on all validation units.
Scoring as scripts/evaluate_fewshot.py (clipped pseudobulk metrics per perturbation, mean over seeds, then over lines);
residual Pearson relative to the context mean of the library used for that perturbation. Paired statistics:
pertrieve.stats.paired_lw (line-weighted, gene-clustered bootstrap), Holm over the reported tests. The outputs hold the
rows of Supplement Tables 21 (Pearson) and 23 (CRISPRa: Pearson, MSE, residual Pearson, cosine discrimination of
Pertrieve, ae and ae + init.).
Writes results/fewshot_confirm/<task>/{summary,per_line,weights,paired}.csv and the test predictions of every model,
predictions/fewshot_confirm/<task>/<model>_s<seed>_<unit>.npz (genes, delta, ctrl; unit = line or line@fold).

  python scripts/confirm_fusion.py --task ptrag
  python scripts/confirm_fusion.py --task kaden
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pertrieve import Pertrieve, Pseudobulk, load_representations, untrained
from pertrieve.baselines import ahlmann_eltze, ridge
from pertrieve.data import CrossLineProfile, read_genes
from pertrieve.metrics import clip_delta, per_perturbation, residual_pearson
from pertrieve.stats import holm, paired_lw

REPO = Path(__file__).resolve().parents[1]
REPS = ("gobert", "genept", "esm2", "scgpt")
SEEDS = (0, 1, 2)
GRID = np.round(np.arange(0, 1.001, 0.1), 2)
METRICS = ["pearson_delta", "disc_cos", "mse", "residual_pearson"]
PD, ALL4 = ("pearson_delta",), tuple(METRICS)
REPORTED = {  # summary {model: metrics}; paired {(model, comparator): metrics}
    "kaden": dict(summary={"ae+xline": ALL4, "ae_xline": ALL4, "ae+xline0": ALL4, "xline": PD, "ae+ridge": PD},
                  paired={("ae+xline", "xline"): PD, ("ae+xline", "ae_xline"): ALL4, ("ae+xline", "ae+ridge"): PD,
                          ("ae+xline", "ae+xline0"): ALL4}),
    "ptrag": dict(summary={m: PD for m in ("ae+xline", "ae_xline", "ae+xline0", "xline", "ae+ridge")},
                  paired={("ae+xline", v): PD for v in ("xline", "ae_xline", "ae+ridge", "ae+xline0")})}
PAIRED_COLS = ["model", "vs", "metric", "diff", "lo", "hi", "p_wilcoxon", "p_holm", "n_genes", "n_rows", "n_lines"]


def trim(df, task):
    keep = REPORTED[task]["summary"]
    df = df[df.model.isin(list(keep))].copy()
    for c in METRICS:
        df[c] = [v if c in keep[m] else np.nan for m, v in zip(df.model, df[c])]
    return df
FUSIONS = {"ae+xline": ("ae_xline", "xline"), "ae+ridge": ("ae_xline", "ridge_xline"), "ae+xline0": ("ae_xline", "xline0")}
TASKS = {"ptrag": dict(lines=("k562", "rpe1", "hepg2", "jurkat"), pb="ptrag/pseudobulk.npz", reps="representations"),
         "kaden": dict(lines=("fibroblast", "rpe1"), pb="fewshot_confirm/kaden/pseudobulk.npz", reps="fewshot_confirm/kaden/representations")}


def crit(P, R, ctrl):
    m = per_perturbation(clip_delta(P, ctrl), R).mean()
    return m.pearson_delta + m.disc_cos


def units(task, pb, data):
    """One record per (line, fold): test genes, full library, and per seed the fit/validation split of the library."""
    out = []
    for c in TASKS[task]["lines"]:
        allg = pb.genes(c)
        if task == "kaden":
            for k in range(5):
                test = sorted(read_genes(REPO / f"splits/fewshot_confirm/kaden/{c}.fold{k}.txt"))
                lib = [g for g in allg if g not in set(test)]
                sp = {}
                for s in SEEDS:
                    vi = np.zeros(len(lib), bool); vi[np.random.default_rng(s).choice(len(lib), len(lib) // 10, replace=False)] = True
                    sp[s] = ([g for g, v in zip(lib, vi) if not v], [g for g, v in zip(lib, vi) if v])
                out.append(dict(line=c, unit=f"{c}@{k}", fold=k, test=test, lib=lib, split=sp, refit=True,
                                weights=lambda s, k=k, c=c: (REPO / f"weights/fewshot_confirm/kaden/{c}/fold{k}/val/xline_s{s}",
                                                             REPO / f"weights/fewshot_confirm/kaden/{c}/fold{k}/xline_s{s}")))
        else:
            held, valset = set(read_genes(REPO / f"splits/ptrag/{c}.test.txt")), set(read_genes(REPO / f"splits/ptrag/{c}.val.txt"))
            z = np.load(data / f"ptrag/ptrag_predictions/{c}.npz", allow_pickle=True)
            test = sorted(str(p) for p in z["perts"] if str(p) != "non-targeting")
            lib = [g for g in allg if g not in held | valset]
            val = sorted(g for g in allg if g in valset)
            out.append(dict(line=c, unit=c, fold=None, test=test, lib=lib, split={s: (lib, val) for s in SEEDS}, refit=False,
                            weights=lambda s, c=c: (REPO / f"weights/ptrag/{c}/xline_s{s}", REPO / f"weights/ptrag/{c}/xline_s{s}")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred", default=None, help="prediction directory (default predictions/fewshot_confirm/<task>)")
    a = ap.parse_args()
    data = Path(a.data); T = TASKS[a.task]; LINES = T["lines"]
    out = Path(a.out or REPO / "results/fewshot_confirm" / a.task); out.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(a.pred or REPO / "predictions/fewshot_confirm" / a.task); pred_dir.mkdir(parents=True, exist_ok=True)
    pb = Pseudobulk(data / T["pb"])
    reps = load_representations(data / T["reps"], REPS)
    names = ("xline_profile",) + REPS
    U = units(a.task, pb, data)
    val_pred, test_pred = {}, {}  # (unit, model, seed) -> (genes, clipped prediction)
    for u in U:
        c = u["line"]; ctrl = pb.control(c); prof = CrossLineProfile(pb, c, LINES)
        feats = lambda genes: [prof.matrix(genes)] + [r.matrix(genes) for r in reps]  # noqa: E731
        Ft, Fl, D = feats(u["test"]), feats(u["lib"]), pb.delta(c, u["lib"])
        for s in SEEDS:
            fit_g, val_g = u["split"][s]
            Ff, Fv, Dfit, Rval = feats(fit_g), feats(val_g), pb.delta(c, fit_g), pb.delta(c, val_g)
            wv, wt = u["weights"](s)
            mv, mt = Pertrieve.load(wv), Pertrieve.load(wt)
            assert tuple(mv.representations) == names
            P = {"xline": (mv.predict(Fv, Ff, Dfit, c), mt.predict(Ft, Fl, D, c))}
            m0 = untrained(names, dim=pb.M.shape[1], w=[1, 1, 0, 0, 0])
            m0.fit_scale(Ff, Dfit, c); pv = m0.predict(Fv, Ff, Dfit, c)
            m0.fit_scale(Fl, D, c); P["xline0"] = (pv, m0.predict(Ft, Fl, D, c))
            for name, fit in (("ae_xline", ahlmann_eltze), ("ridge_xline", ridge)):
                V = fit(Ff[0], Dfit, Fv[0])
                h = max(V, key=lambda h: crit(V[h], Rval, ctrl))
                P[name] = (V[h], fit(Fl[0], D, Ft[0])[h] if u["refit"] else fit(Ff[0], Dfit, Ft[0])[h])
            for name, (pv, pt) in P.items():
                val_pred[(u["unit"], name, s)] = (val_g, clip_delta(pv, ctrl)); test_pred[(u["unit"], name, s)] = clip_delta(pt, ctrl)
        print(f"[{u['unit']}] test {len(u['test'])}, library {len(u['lib'])}", flush=True)
    # fusion weights on the validation genes: one a per seed, mean over units of Pearson delta + cosine discrimination
    weights = {}
    for f, (A, B) in FUSIONS.items():
        for s in SEEDS:
            sc = {w: [] for w in GRID}
            for u in U:
                c = u["line"]; ctrl = pb.control(c)
                (g, PA), (_, PB) = val_pred[(u["unit"], A, s)], val_pred[(u["unit"], B, s)]
                R = pb.delta(c, g)
                for w in GRID:
                    sc[w].append(crit(w * PA + (1 - w) * PB, R, ctrl))
            weights[(f, s)] = max(GRID, key=lambda w: np.mean(sc[w]))
    pd.DataFrame([dict(fusion=f, seed=s, a=w) for (f, s), w in weights.items() if f == "ae+xline"]).to_csv(out / "weights.csv", index=False)  # Table 21
    # test predictions per line (folds merged), per-perturbation metrics averaged over seeds
    per, rows = {}, []
    for c in LINES:
        ctrl = pb.control(c); us = [u for u in U if u["line"] == c]
        test = sorted(g for u in us for g in u["test"]); R = pb.delta(c, test)
        C = np.zeros_like(R)
        for u in us:  # context mean of the library that predicted each test gene
            idx = [i for i, g in enumerate(test) if g in set(u["test"])]
            C[idx] = clip_delta(pb.context_mean(c, u["lib"]), ctrl)
        for name in ["xline", "xline0", "ae_xline", "ridge_xline"] + list(FUSIONS):
            ms = []
            for s in SEEDS:
                if name in FUSIONS:
                    A, B = FUSIONS[name]; w = weights[(name, s)]
                    for u in us:
                        test_pred[(u["unit"], name, s)] = clip_delta(w * test_pred[(u["unit"], A, s)] + (1 - w) * test_pred[(u["unit"], B, s)], ctrl)
                P = merged(us, name, s, test, test_pred)
                for u in us:
                    np.savez_compressed(pred_dir / f"{name}_s{s}_{u['unit']}.npz", genes=np.array(u["test"]),
                                        delta=test_pred[(u["unit"], name, s)].astype(np.float32), ctrl=ctrl,
                                        **({"weight": w} if name in FUSIONS else {}))
                m = per_perturbation(P, R); m["residual_pearson"] = residual_pearson(P, R, C); ms.append(m)
            m = sum(ms) / len(ms)
            per.setdefault(name, []).append(m.assign(line=c, gene=test))
            rows.append(dict(model=name, line=c, n=len(test), **m[METRICS].mean().to_dict()))
    PL = pd.DataFrame(rows)
    summ = trim(PL.groupby("model")[METRICS].mean().reset_index(), a.task).sort_values("pearson_delta").set_index("model")
    trim(PL, a.task).to_csv(out / "per_line.csv", index=False); summ.to_csv(out / "summary.csv")
    per = {k: pd.concat(v, ignore_index=True) for k, v in per.items()}

    Tt = [dict(model=x, vs=y, metric=met, **paired_lw((-1 if met == "mse" else 1) * per[x][met].to_numpy(),
                                                      (-1 if met == "mse" else 1) * per[y][met].to_numpy(),
                                                      per[x].gene.to_numpy(), per[x].line.to_numpy()))
          for (x, y), mets in REPORTED[a.task]["paired"].items() for met in mets]
    Tt = pd.DataFrame(Tt); Tt["p_holm"] = holm(Tt.p_wilcoxon.fillna(1).to_numpy()); Tt = Tt[PAIRED_COLS]
    Tt.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(4).to_string())
    print(pd.DataFrame([dict(fusion=f, seed=s, a=w) for (f, s), w in weights.items() if f == "ae+xline"]).pivot(index="fusion", columns="seed", values="a"))
    print(Tt[["model", "vs", "metric", "diff", "lo", "hi", "p_holm"]].round(5).to_string(index=False))


def merged(us, name, s, test, test_pred):
    """Test predictions of one line with the folds merged, rows in the order of `test`."""
    g = np.concatenate([np.array(u["test"]) for u in us]); d = np.concatenate([test_pred[(u["unit"], name, s)] for u in us])
    o = {x: i for i, x in enumerate(g)}
    return d[[o[x] for x in test]]


if __name__ == "__main__":
    main()
