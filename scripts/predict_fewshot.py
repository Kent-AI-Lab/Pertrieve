#!/usr/bin/env python3
"""Pertrieve on State's official few-shot splits (Replogle/Nadig, four cell lines).

In the few-shot task a test gene is unmeasured in the target line but may be measured in the other three.
The library of line c is every perturbation of c except its test genes. Variants:
  gene   gene representations only (GoBERT, GenePT, ESM2, scGPT)
  xline  cross-cell-line profile of the gene (pertrieve.data.CrossLineProfile) + the four gene representations;
         the bilinear term acts on the profile (the Pertrieve part of the primary few-shot model, which
         scripts/fuse_fewshot.py fuses with the cross-cell-line linear model)
  xonly  cross-cell-line profile only
  xline0 the xline model at its initialization (untrained retrieval control: equal cosine weights on the profile and
         GoBERT, no bilinear term, output scale fitted by leave-one-out least squares on the library); no weight file
Writes <out>/<variant>_s<seed>_<line>.npz (genes, delta, ctrl).
--val predicts the seeded validation genes (splits/fewshot/<line>.val_s<seed>.txt) with the validation fits
(weights/fewshot/<line>/val/, fitted without those genes; library without them) and writes <out>/val/; these
predictions choose the DE gain (de_fewshot.py) and the fusion weights (fuse_fewshot.py).

  python scripts/predict_fewshot.py --variant xline [--val]
"""
import argparse
from pathlib import Path

import numpy as np

from pertrieve import Pertrieve, Pseudobulk, load_representations, untrained
from pertrieve.data import CrossLineProfile, read_genes

LINES = ("k562", "rpe1", "hepg2", "jurkat")
REPO = Path(__file__).resolve().parents[1]


def representations(names, pb, line, rep_dir):
    gene_reps = iter(load_representations(rep_dir, [n for n in names if n != "xline_profile"]))
    return [CrossLineProfile(pb, line, LINES) if n == "xline_profile" else next(gene_reps) for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--variant", default="xline", choices=["gene", "xline", "xonly", "xline0"])
    ap.add_argument("--weights", default=str(REPO / "weights/fewshot"))
    ap.add_argument("--out", default=str(REPO / "predictions/fewshot"))
    ap.add_argument("--val", action="store_true", help="validation fits on the validation genes")
    a = ap.parse_args()
    pb = Pseudobulk(Path(a.data) / "replogle_nadig_pseudobulk.npz")
    out = Path(a.out) / ("val" if a.val else ""); out.mkdir(parents=True, exist_ok=True)
    for c in LINES:
        held = set(read_genes(REPO / f"splits/fewshot/{c}.test.txt"))
        test = sorted(g for g in pb.genes(c) if g in held)
        lib = [g for g in pb.genes(c) if g not in held]
        if a.variant == "xline0":
            stems = [f"xline0_s{s}" for s in (0, 1, 2)]
        else:
            stems = [f.stem for f in sorted((Path(a.weights) / c / ("val" if a.val else "")).glob(f"{a.variant}_s[0-9].npz"))]
        for stem in stems:
            lib_s, test_s = lib, test
            if a.val:
                val = set(read_genes(REPO / f"splits/fewshot/{c}.val_s{stem[-1]}.txt"))
                lib_s, test_s = [g for g in pb.genes(c) if g not in held | val], [g for g in pb.genes(c) if g in val]
            if a.variant == "xline0":
                m = untrained(("xline_profile", "gobert", "genept", "esm2", "scgpt"), dim=pb.M.shape[1], w=[1, 1, 0, 0, 0])
                reps = representations(m.representations, pb, c, Path(a.data) / "representations")
                m.fit_scale([r.matrix(lib_s) for r in reps], pb.delta(c, lib_s), c)
            else:
                m = Pertrieve.load(Path(a.weights) / c / ("val" if a.val else "") / stem)
                reps = representations(m.representations, pb, c, Path(a.data) / "representations")
            P = m.predict([r.matrix(test_s) for r in reps], [r.matrix(lib_s) for r in reps], pb.delta(c, lib_s), c)
            np.savez_compressed(out / f"{stem}_{c}.npz", genes=np.array(test_s), delta=P.astype(np.float32), ctrl=pb.control(c))
            print(f"{c} {stem}: {len(test_s)} test genes, library {len(lib_s)}, {m.n_parameters:,} parameters, scale {m.scale[c]:.3f}", flush=True)


if __name__ == "__main__":
    main()
