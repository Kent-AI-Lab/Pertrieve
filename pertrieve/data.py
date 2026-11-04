"""Data access: pseudobulk responses, gene representations and held-out gene lists."""
from __future__ import annotations

from pathlib import Path

import numpy as np

CONTROL = "non-targeting"


class Pseudobulk:
    """Mean expression per (cell line, perturbation) over all cells, as written by scripts/build_pseudobulk.py.

    npz fields: cell_lines [N], perts [N], n [N] (cells per group), mean_hvg [N, G].
    """

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=True)
        self.cell_lines = z["cell_lines"].astype(str)
        self.perts = z["perts"].astype(str)
        self.n = z["n"]
        self.M = z["mean_hvg"].astype(np.float64)
        self.idx = {(c, p): i for i, (c, p) in enumerate(zip(self.cell_lines, self.perts))}

    def genes(self, line: str) -> list[str]:
        """Perturbed genes measured in `line` (sorted, control excluded)."""
        return sorted(p for c, p in zip(self.cell_lines, self.perts) if c == line and p != CONTROL)

    def mean(self, line: str, genes: list[str]) -> np.ndarray:
        return self.M[[self.idx[(line, g)] for g in genes]]

    def count(self, line: str, genes: list[str]) -> np.ndarray:
        return self.n[[self.idx[(line, g)] for g in genes]]

    def control(self, line: str) -> np.ndarray:
        return self.mean(line, [CONTROL])[0]

    def delta(self, line: str, genes: list[str]) -> np.ndarray:
        """Response change: mean expression of each perturbation minus the control mean of the line."""
        return self.mean(line, genes) - self.control(line)

    def context_mean(self, line: str, genes: list[str]) -> np.ndarray:
        """Cell-weighted mean response of the given (training) perturbations."""
        n = self.count(line, genes).astype(np.float64)
        return (self.delta(line, genes) * n[:, None]).sum(0) / n.sum()


class Representation:
    """Gene -> L2-normalized vector. Stored as npz with fields genes [n] and X [n, d]; all-zero rows mean
    that the gene has no representation."""

    def __init__(self, path: str | Path):
        z = np.load(path, allow_pickle=True)
        X = z["X"].astype(np.float64)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        self.X = np.divide(X, nrm, out=np.zeros_like(X), where=nrm > 0)
        self.index = {g: i for i, g in enumerate(z["genes"].astype(str))}
        self.dim = X.shape[1]

    def matrix(self, genes: list[str]) -> np.ndarray:
        out = np.zeros((len(genes), self.dim))
        for i, g in enumerate(genes):
            j = self.index.get(g)
            if j is not None:
                out[i] = self.X[j]
        return out


class NoRepresentation:
    """Placeholder for a representation that a model variant does not use (all genes missing)."""

    dim = 1

    def matrix(self, genes: list[str]) -> np.ndarray:
        return np.zeros((len(genes), 1))


def load_representations(rep_dir: str | Path, names) -> list:
    """One loader per name; "none" marks a slot the model does not use."""
    return [NoRepresentation() if n == "none" else Representation(Path(rep_dir) / f"{n}.npz") for n in names]


def _l2_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.where(n > 0, X / np.where(n > 0, n, 1), 0)


class CrossLineProfile:
    """Few-shot representation of a gene: the mean of its L2-normalized responses in the other cell lines
    (each line weighted equally), L2-normalized; zero if the gene is measured in no other line."""

    def __init__(self, pb: Pseudobulk, target: str, lines=("k562", "rpe1", "hepg2", "jurkat")):
        self.pb, self.others = pb, [o for o in lines if o != target]
        self.measured = {o: set(pb.genes(o)) for o in self.others}
        self.dim = pb.M.shape[1]

    def matrix(self, genes: list[str]) -> np.ndarray:
        P = np.zeros((len(genes), self.dim)); n = np.zeros(len(genes))
        for o in self.others:
            ok = np.array([g in self.measured[o] for g in genes])
            if ok.any():
                P[ok] += _l2_rows(self.pb.delta(o, [g for g, k in zip(genes, ok) if k])); n[ok] += 1
        return _l2_rows(P / np.maximum(n, 1)[:, None])


def read_genes(path: str | Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def heldout_split(split_dir: str | Path, split: str, protocol: str) -> dict:
    """Held-out genes of a Replogle/Nadig split.

    protocol "gd" (held-out genes, separate selection genes): RPE1 selection genes = <split>.select.txt, test genes in K562/HepG2/Jurkat =
    <split>.eval.txt. protocol "full": all held-out genes (<split>.heldout.txt) are queries in every line.
    Training libraries always exclude every held-out gene."""
    d = Path(split_dir)
    held = set(read_genes(d / f"{split}.heldout.txt"))
    if protocol == "gd":
        sel, ev = set(read_genes(d / f"{split}.select.txt")), set(read_genes(d / f"{split}.eval.txt"))
        assert sel | ev == held and not sel & ev
    else:
        sel = ev = held
    return dict(heldout=held, queries={"rpe1": sel, "k562": ev, "hepg2": ev, "jurkat": ev})
