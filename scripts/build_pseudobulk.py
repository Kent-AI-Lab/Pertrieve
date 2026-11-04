#!/usr/bin/env python3
"""Build replogle_nadig_pseudobulk.npz from State's preprocessed Replogle/Nadig AnnData (replogle.h5ad).

Mean of obsm['X_hvg'] (log1p, 2,000 highly variable genes) over all cells of each (cell_line, gene) group,
including the non-targeting controls. Reads the file in row chunks, so memory stays small.

  python scripts/build_pseudobulk.py --h5ad replogle.h5ad --out data/replogle_nadig_pseudobulk.npz
"""
import argparse

import h5py
import numpy as np
import scipy.sparse as sp


def read_categorical(f, key):
    g = f[f"obs/{key}"]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]])
    return cats, g["codes"][:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args()
    f = h5py.File(a.h5ad, "r")
    cl_cats, cl_codes = read_categorical(f, "cell_line")
    g_cats, g_codes = read_categorical(f, "gene")
    grp = cl_codes.astype(np.int64) * len(g_cats) + g_codes
    uniq, inv = np.unique(grp, return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq)).astype(np.int64)
    H = f["obsm/X_hvg"]
    total = np.zeros((len(uniq), H.shape[1]))
    for s in range(0, len(inv), a.chunk):
        e = min(s + a.chunk, len(inv))
        M = sp.csr_matrix((np.ones(e - s), (inv[s:e], np.arange(e - s))), shape=(len(uniq), e - s))
        total += M @ H[s:e].astype(np.float64)
    np.savez_compressed(a.out, cell_lines=cl_cats[uniq // len(g_cats)], perts=g_cats[uniq % len(g_cats)], n=n,
                        mean_hvg=(total / n[:, None]).astype(np.float32))
    print(f"{len(uniq)} groups written to {a.out}")


if __name__ == "__main__":
    main()
