#!/usr/bin/env bash
# Reproduce the held-out-gene results (Tables 1-2, representation and component ablations; Fig. 2, repeated outer
# splits) from the released weights, with the closed-form baselines of Table 1. CPU only; about 30 min wall on a 32-core server (7 CPU-hours).
# Usage: bash scripts/reproduce_heldout.sh [data dir]
# SLIM–STRING rows: install SLIM and set SLIM_STRING_H5 to its data/gene_string_embeddings.v0.3.h5 (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-data}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
for proto in gd full; do
  for split in unseen cluster cluster_genept; do
    dir=weights/heldout/$proto/$split
    [ -d "$dir" ] || continue
    for model in $(ls "$dir" | sed -n 's/_s[0-9]\.npz$//p' | sort -u); do
      python scripts/predict_heldout.py --data "$DATA" --split "$split" --protocol "$proto" --model "$model"
    done
    if [ "$proto" = gd ]; then  # the closed-form baselines of Table 1 (protocol gd)
      python scripts/baselines_heldout.py --data "$DATA" --split "$split" --protocol "$proto"
      if [ -n "${SLIM_STRING_H5:-}" ]; then
        python scripts/baseline_slim_heldout.py --data "$DATA" --split "$split" --protocol "$proto" --string-h5 "$SLIM_STRING_H5"
      fi
    fi
    python scripts/evaluate_heldout.py --data "$DATA" --split "$split" --protocol "$proto"
  done
done
# repeated outer splits (Fig. 2, Supplement Section E)
python scripts/repeat_heldout.py --data "$DATA" ${SLIM_STRING_H5:+--slim-string "$SLIM_STRING_H5"}
python scripts/check_expected.py
