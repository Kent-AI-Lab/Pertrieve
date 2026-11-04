#!/usr/bin/env python3
"""Validation-selected fusion of Pertrieve xline with a linear cross-cell-line model: prediction = a * linear +
(1 - a) * Pertrieve, a in {0, 0.1, ..., 1}, chosen on the seeded validation genes (validation fits of both models;
scripts/predict_fewshot.py --val and scripts/fewshot_baselines.py) by Pearson delta + cosine discrimination, then
applied once to the test predictions.
The primary few-shot model of the paper is --partner ae_xline --rule pooled (xline+ae_xline: one a per seed for the
four lines); the ridge partner and the per-line rule are exploratory rows. --retrieval xline0 fuses the linear model
with the untrained retrieval control instead (xline0+ae_xline, "linear + untrained retrieval"); --retrieval ridge_xline fuses the
two linear models (ridge_xline+ae_xline, "ae + ridge").
Writes <pred>/<retrieval>+<partner>[_perline]_s<seed>_<line>.npz (scored by evaluate_fewshot.py) and, with the same a, the
fused validation predictions <pred>/val/xline+<partner>[_perline]_s<seed>_<line>.npz (used by de_fewshot.py).

  python scripts/fuse_fewshot.py --partner ae_xline
"""
import argparse
from pathlib import Path

import numpy as np

from pertrieve import Pseudobulk
from pertrieve.metrics import clip_delta, per_perturbation

LINES = ("k562", "rpe1", "hepg2", "jurkat")
SEEDS = (0, 1, 2)
GRID = np.round(np.arange(0, 1.001, 0.1), 2)
REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--pred", default=str(REPO / "predictions/fewshot"))
    ap.add_argument("--partner", default="ae_xline", choices=["ae_xline", "ridge_xline"])
    ap.add_argument("--rule", default="pooled", choices=["pooled", "per-line"])
    ap.add_argument("--retrieval", default="xline", choices=["xline", "xline0", "ridge_xline"], help="second member: retrieval (xline), the untrained control (xline0) or the cross-cell-line ridge (ridge_xline: two linear models)")
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    pred = Path(a.pred)
    curve = {}
    for c in LINES:
        ctrl = pb.control(c)
        for s in SEEDS:
            r, q = (np.load(pred / "val" / f"{m}_s{s}_{c}.npz") for m in (a.retrieval, a.partner))
            assert list(r["genes"]) == list(q["genes"])
            R = pb.delta(c, list(r["genes"].astype(str)))
            P, L = clip_delta(r["delta"].astype(float), ctrl), clip_delta(q["delta"].astype(float), ctrl)
            for w in GRID:
                m = per_perturbation(clip_delta(w * L + (1 - w) * P, ctrl), R).mean()
                curve[(c, s, w)] = m.pearson_delta + m.disc_cos
    name = f"{a.retrieval}+{a.partner}" + ("_perline" if a.rule == "per-line" else "")
    for s in SEEDS:
        pooled = max(GRID, key=lambda w: np.mean([curve[(c, s, w)] for c in LINES]))
        for c in LINES:
            w = pooled if a.rule == "pooled" else max(GRID, key=lambda w: curve[(c, s, w)])
            ctrl = pb.control(c)
            r, q = (np.load(pred / f"{m}_s{s}_{c}.npz") for m in (a.retrieval, a.partner))
            assert list(r["genes"]) == list(q["genes"])
            F = clip_delta(w * clip_delta(q["delta"].astype(float), ctrl) + (1 - w) * clip_delta(r["delta"].astype(float), ctrl), ctrl)
            np.savez_compressed(pred / f"{name}_s{s}_{c}.npz", genes=r["genes"], delta=F.astype(np.float32), ctrl=ctrl, weight=w)
            r, q = (np.load(pred / "val" / f"{m}_s{s}_{c}.npz") for m in (a.retrieval, a.partner))
            F = clip_delta(w * clip_delta(q["delta"].astype(float), ctrl) + (1 - w) * clip_delta(r["delta"].astype(float), ctrl), ctrl)
            np.savez_compressed(pred / "val" / f"{name}_s{s}_{c}.npz", genes=r["genes"], delta=F.astype(np.float32), ctrl=ctrl,
                                weight=w)
            print(f"{name} s{s} {c}: a = {w:g}", flush=True)


if __name__ == "__main__":
    main()
