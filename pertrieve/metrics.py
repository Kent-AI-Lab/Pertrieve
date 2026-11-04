"""Per-perturbation metrics on response changes (prediction and truth relative to the same control mean).

These follow the cell-eval "anndata" definitions used throughout the paper:
  pearson_delta  Pearson(pred - ctrl, real - ctrl) across genes
  disc_cos/l2    1 - rank/N of the true perturbation among the N real responses of the same line, ordered by
                 cosine / Euclidean distance to the predicted response (1 = always closest, 0.5 = chance)
  mse            mean squared error of the profile (equal to that of the response change)
Predictions are clipped like the scorer: profile = clip(delta + ctrl, 0) before every metric.
`cell_eval_metrics` runs the official cell-eval package on the same inputs, for an exact cross-check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def clip_delta(P: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    return np.clip(P + ctrl, 0, None) - ctrl


def per_perturbation(P: np.ndarray, R: np.ndarray) -> pd.DataFrame:
    """P, R: [N, G] predicted / real response changes of one line, rows aligned."""
    Pc = P - P.mean(1, keepdims=True); Rc = R - R.mean(1, keepdims=True)
    pear = (Pc * Rc).sum(1) / np.sqrt((Pc ** 2).sum(1) * (Rc ** 2).sum(1) + 1e-30)
    d2 = (P ** 2).sum(1)[:, None] + (R ** 2).sum(1)[None] - 2 * P @ R.T
    disc_l2 = 1 - (d2 < np.diag(d2)[:, None]).sum(1) / len(P)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-30)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-30)
    dc = -(Pn @ Rn.T)
    disc_cos = 1 - (dc < np.diag(dc)[:, None]).sum(1) / len(P)
    mse = ((P - R) ** 2).mean(1)
    return pd.DataFrame(dict(pearson_delta=pear, disc_cos=disc_cos, disc_l2=disc_l2, mse=mse))


def residual_pearson(P: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Residual (perturbation-specific) correlation: Pearson(pred - mu_c, real - mu_c), mu_c = context mean."""
    return per_perturbation(P - C, R - C)["pearson_delta"].to_numpy()


def cell_eval_metrics(line: str, genes, pred_profile: np.ndarray, real_profile: np.ndarray, ctrl: np.ndarray,
                      outdir: str, control: str = "non-targeting") -> dict:
    """Official cell-eval anndata metrics (means over perturbations) on one-row-per-perturbation AnnData
    objects plus one control row. Requires `pip install cell-eval`."""
    import anndata as ad
    from cell_eval import MetricsEvaluator

    obs = pd.DataFrame({"gene": list(genes) + [control], "cell_line": line})
    obs.index = obs.index.astype(str)
    pred = ad.AnnData(X=np.vstack([pred_profile, ctrl[None]]).astype(np.float32), obs=obs)
    real = ad.AnnData(X=np.vstack([real_profile, ctrl[None]]).astype(np.float32), obs=obs.copy())
    ev = MetricsEvaluator(adata_pred=pred, adata_real=real, control_pert=control, pert_col="gene",
                          outdir=outdir, prefix=line, skip_de=True)
    _, agg = ev.compute(profile="anndata", metric_configs={},
                        skip_metrics=["pearson_edistance", "clustering_agreement"], write_csv=False)
    m = agg.to_pandas().set_index("statistic").loc["mean"]
    keys = ("pearson_delta", "discrimination_score_l2", "discrimination_score_cosine", "mse")
    return {k: float(m[k]) for k in keys if k in m}
