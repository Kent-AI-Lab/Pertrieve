#!/usr/bin/env python3
"""Combination variant chosen per run on validation conditions (Supplement Table 37, the three Norman rows).

splits/combo_variant_selection.csv lists, for every run (split x seed) of GEARS-Norman, scGenePT-Norman and
Scouter-Norman, the validation score that the run logged for each of the variants plain, additive and split_joint
(the run's epoch-selection criterion: mean validation pearson_delta on the GEARS-format benchmarks, mean validation
DE Pearson on Scouter's) and the variant with the highest score. This script takes each run's test results for the
chosen variant from the outputs of run_gears_format.py and run_scouter_format.py (all three variants must have been
run) and averages them as those scripts do: mean over test conditions per seed, then over seeds.
Outputs results/combo_selected/summary.csv.

  python scripts/select_combo_variant.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METRICS = {"gears_format": ["pearson_delta"], "scouter_format": ["NormMSE", "Pearson"]}


def main():
    sel = pd.read_csv(REPO / "splits/combo_variant_selection.csv")
    rows = []
    for (fmt, bench), S in sel.groupby(["format", "benchmark"], sort=False):
        res = REPO / "results" / fmt / bench
        R = pd.concat([pd.read_csv(res / f"per_condition_{v}.csv").assign(variant=v) for v in sorted(S.chosen.unique())])
        R = R.merge(S[["split", "seed", "chosen"]], on=["split", "seed"])
        R = R[R.variant == R.chosen]
        counts = ", ".join(f"{v} {n}" for v, n in S.chosen.value_counts().items())
        for m in METRICS[fmt]:
            rows.append(dict(benchmark=f"{fmt}/{bench}", metric=m, value=R.groupby("seed")[m].mean().mean(),
                             runs=len(S), chosen=counts))
    out = REPO / "results/combo_selected"; out.mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame(rows)
    T.to_csv(out / "summary.csv", index=False)
    print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
