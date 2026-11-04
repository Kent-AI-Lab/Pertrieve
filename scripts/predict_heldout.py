#!/usr/bin/env python3
"""Predict the responses of held-out genes on the Replogle/Nadig screens with released Pertrieve weights.

For each cell line, the reference library is every measured perturbation of that line except the held-out
genes; the queries are the held-out genes measured in the line (protocol gd: RPE1 selection genes
and K562/HepG2/Jurkat evaluation genes). Writes one npz per (model, seed, line) with genes, delta (raw
prediction, before clipping) and ctrl (control mean).

  python scripts/predict_heldout.py --data data --split unseen --protocol gd --model pertrieve
"""
import argparse
from pathlib import Path

import numpy as np

from pertrieve import Pertrieve, Pseudobulk, heldout_split, load_representations

LINES = ("rpe1", "k562", "hepg2", "jurkat")
REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"), help="directory with replogle_nadig_pseudobulk.npz and representations/")
    ap.add_argument("--split", default="unseen", choices=["unseen", "cluster", "cluster_genept"])
    ap.add_argument("--protocol", default="gd", choices=["gd", "full"])
    ap.add_argument("--model", default="pertrieve", help="weight file stem, e.g. pertrieve, only_gobert, no_gobert")
    ap.add_argument("--weights", default=str(REPO / "weights/heldout"))
    ap.add_argument("--out", default=str(REPO / "predictions/heldout"))
    a = ap.parse_args()

    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    sp = heldout_split(REPO / "splits/heldout", a.split, a.protocol)
    files = sorted((Path(a.weights) / a.protocol / a.split).glob(f"{a.model}_s[0-9].npz"))
    assert files, f"no weights for {a.model} ({a.protocol}/{a.split})"
    out = Path(a.out) / a.protocol / a.split
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        m = Pertrieve.load(f)
        reps = load_representations(Path(a.data) / "representations", m.representations)
        for c in LINES:
            lib = [g for g in pb.genes(c) if g not in sp["heldout"]]
            q = sorted(g for g in pb.genes(c) if g in sp["queries"][c])
            K = [r.matrix(lib) for r in reps]; Q = [r.matrix(q) for r in reps]
            P = m.predict(Q, K, pb.delta(c, lib), c)
            np.savez_compressed(out / f"{f.stem}_{c}.npz", genes=np.array(q), delta=P.astype(np.float32), ctrl=pb.control(c))
        print(f"{a.protocol}/{a.split}/{f.stem}: {m.n_parameters:,} parameters", flush=True)


if __name__ == "__main__":
    main()
