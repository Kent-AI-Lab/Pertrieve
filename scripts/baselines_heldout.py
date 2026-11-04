#!/usr/bin/env python3
"""Closed-form literature baselines on the held-out-gene splits (Table 1 / Supplementary Table 9):
GenePert ridge and the Ahlmann-Eltze linear model (pertrieve/baselines.py) on gene representations.

Same libraries and queries as scripts/predict_heldout.py: per cell line, the model is fitted on that line's
measured perturbations except all held-out genes. Hyper-parameters (ridge lambda; K and lambda for the linear
model) are chosen on the RPE1 queries only (selection genes under protocol gd) by Pearson delta + cosine
discrimination, then applied to K562, HepG2 and Jurkat. Writes predictions in the format of predict_heldout.py
(<name>_s0_<line>.npz), so scripts/evaluate_heldout.py scores them with the other models, plus
<name>.selection.json.

Models: genepert_ridge_<rep> and ae_linear_<rep>, rep in gobert, genept, esm2, scgpt, random, all4 (the four
representations concatenated). Default: the rows reported in the paper.

  python scripts/baselines_heldout.py --split unseen --protocol gd
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pertrieve import Pseudobulk, heldout_split, load_representations
from pertrieve.baselines import ahlmann_eltze, combine, ridge
from pertrieve.metrics import clip_delta, per_perturbation

LINES = ("rpe1", "k562", "hepg2", "jurkat")
FOUR = ("gobert", "genept", "esm2", "scgpt")
REPS = FOUR + ("random", "all4")
PAPER = ("genepert_ridge_gobert", "ae_linear_gobert", "ae_linear_all4")
REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--split", default="unseen", choices=["unseen", "cluster", "cluster_genept"])
    ap.add_argument("--protocol", default="gd", choices=["gd", "full"])
    ap.add_argument("--models", nargs="+", default=list(PAPER),
                    choices=[f"{m}_{r}" for m in ("genepert_ridge", "ae_linear") for r in REPS])
    ap.add_argument("--out", default=str(REPO / "predictions/heldout"))
    a = ap.parse_args()

    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    sp = heldout_split(REPO / "splits/heldout", a.split, a.protocol)
    reps = dict(zip(REPS[:-1], load_representations(Path(a.data) / "representations", REPS[:-1])))
    out = Path(a.out) / a.protocol / a.split
    out.mkdir(parents=True, exist_ok=True)

    def features(rep, genes):
        return combine([reps[r].matrix(genes) for r in FOUR]) if rep == "all4" else reps[rep].matrix(genes)

    def fit_predict(name, c):
        kind, rep = name.rsplit("_", 1)
        lib = [g for g in pb.genes(c) if g not in sp["heldout"]]
        q = sorted(g for g in pb.genes(c) if g in sp["queries"][c])
        f = ridge if kind == "genepert_ridge" else ahlmann_eltze
        return q, f(features(rep, lib), pb.delta(c, lib), features(rep, q))

    for name in a.models:
        q, cand = fit_predict(name, "rpe1")
        ctrl, R = pb.control("rpe1"), pb.delta("rpe1", q)
        val = {}
        for h, P in cand.items():
            m = per_perturbation(clip_delta(P, ctrl), R).mean()
            val[str(h)] = float(m.pearson_delta + m.disc_cos)
        best = max(val, key=val.get)
        (out / f"{name}.selection.json").write_text(json.dumps(dict(chosen=best, rpe1_score=val), indent=1))
        for c in LINES[1:]:
            q, cand = fit_predict(name, c)
            P = next(P for h, P in cand.items() if str(h) == best)
            np.savez_compressed(out / f"{name}_s0_{c}.npz", genes=np.array(q), delta=P.astype(np.float32), ctrl=pb.control(c))
        print(f"{a.protocol}/{a.split}/{name}: selected {best} on RPE1", flush=True)


if __name__ == "__main__":
    main()
