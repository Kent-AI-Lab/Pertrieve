#!/usr/bin/env bash
# Reproduce the other benchmarks from the released weights and compare with expected/. CPU only, about 2.5 h wall on a
# 32-core server (38 CPU-hours).
# Usage: bash scripts/reproduce_other.sh [data dir]
# Not included (see README): DE recovery (Supplement Table 20; needs replogle.h5ad and cell-eval, scripts/de_fewshot.py),
# SLIM rows of the few-shot table (needs SLIM, scripts/fewshot_baselines.py --slim-string), and scoring new
# CellSimBench predictions (scripts/run_cellsimbench.py, then the CellSimBench scorer).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${1:-data}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
# State few-shot (Table 3, Fig. 4); primary model = xline fused with the cross-cell-line linear model (xline+ae_xline, pooled)
for v in gene xline xonly xline0; do
  python scripts/predict_fewshot.py --data "$DATA" --variant $v
done
for v in xline xonly xline0; do
  python scripts/predict_fewshot.py --data "$DATA" --variant $v --val
done
python scripts/fewshot_baselines.py --data "$DATA"
for partner in ae_xline ridge_xline; do
  for rule in pooled per-line; do
    python scripts/fuse_fewshot.py --data "$DATA" --partner $partner --rule $rule
  done
done
for r in xline0 ridge_xline; do
  python scripts/fuse_fewshot.py --data "$DATA" --partner ae_xline --retrieval $r
done
python scripts/evaluate_fewshot.py --data "$DATA" --extra "State=$DATA/fewshot_bio/state_predictions/{line}.npz"
# few-shot performance by response breadth and Hallmark set recovery (Fig. 5, Supplement Section J)
python scripts/breadth_fewshot.py --data "$DATA"
python scripts/hallmark_fewshot.py --data "$DATA"
# cross-cell-line transfer of the learned similarity (Fig. 3, Supplement Section K)
for split in unseen cluster cluster_genept; do
  python scripts/transfer_heldout.py --data "$DATA" --split $split
done
# SLIM's protocol (Supplement Table 62)
python scripts/run_slim_protocol.py --data "$DATA"
# GEARS / scGenePT and Scouter formats (Supplement Section M)
for b in gears_norman scgenept_norman; do
  for v in plain additive split_joint split_double; do
    python scripts/run_gears_format.py --data "$DATA" --bench $b --variant $v
  done
done
python scripts/run_gears_format.py --data "$DATA" --bench scgenept_adamson --variant plain
for v in plain additive split_joint split_double; do
  for ds in norman wessels; do
    python scripts/run_scouter_format.py --data "$DATA" --dataset $ds --variant $v
  done
done
python scripts/run_scouter_format.py --data "$DATA" --dataset adamson --variant plain
python scripts/scouter_familiar.py --data "$DATA"
# Norman rows of Supplement Table 37: the variant chosen per run on validation conditions
python scripts/select_combo_variant.py
# fusion confirmation on the PT-RAG split and the CRISPRa screen (Table 4, Supplement Tables 21-22); the PT-RAG task
# also writes the fused model's test predictions that run_ptrag.py reports as "Pertrieve"
for task in ptrag kaden; do
  python scripts/confirm_fusion.py --data "$DATA" --task $task
done
# PerturbQA, PT-RAG comparison, CellSimBench statistics
python scripts/run_perturbqa.py --data "$DATA"
python scripts/run_ptrag.py --data "$DATA"
python scripts/cellsimbench_stats.py --data "$DATA"
# per-query magnitude (Section 3.3, Supplement Table 65); needs the held-out predictions of reproduce_heldout.sh
if [ -f predictions/heldout/gd/cluster_genept/pertrieve_s2_jurkat.npz ]; then
  python scripts/magnitude.py --data "$DATA"
fi
python scripts/check_expected.py
