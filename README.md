# Pertrieve

Pertrieve predicts the transcriptional response to an unmeasured genetic perturbation by retrieving measured
perturbations and combining their responses. A query gene is scored against every reference perturbation of
the target cell line with a learned similarity, a mixture of cosine similarities over frozen gene
representations (GoBERT, GenePT, ESM2, scGPT) plus a low-rank bilinear term, and the prediction is the
softmax-weighted average of the references' measured responses:

```
s(q, k) = (1/τ) Σ_m w_m cos(e_q^(m), e_k^(m)) + (Uᵀ e_q^(1))ᵀ (Vᵀ e_k^(1))
Δ̂_q    = γ_c Σ_k softmax_k s(q, k) · Δ_k
```

The trained part has 65,541 parameters (w, τ, U, V ∈ ℝ^{1024×32}); prediction is a matrix product and runs on
a CPU. Every prediction is a convex combination of measured responses, so it can be traced back to the
experiments it reuses.

**Naming.** In the paper, *Pertrieve* denotes the full model on tasks where the query gene has been measured in
other cell lines (State's few-shot task, the PT-RAG split, the CRISPRa screen): the retrieval above, run on the
gene's cross-cell-line expression profile, fused with a closed-form cross-cell-line linear model (Ahlmann-Eltze),
F = a · linear + (1 − a) · retrieval, with one weight a per seed chosen on validation genes. *Pertrieve-Retrieval*
(Pertrieve-R in figures) is the retrieval component alone, which is the model evaluated on every task without
cross-cell-line measurements (held-out-gene splits, SLIM, PerturbQA, GEARS, scGenePT, Scouter, CellSimBench). In this
code the class `Pertrieve` and the weight files `pertrieve_s<seed>` are the retrieval component; the few-shot fusion
is produced by `scripts/fuse_fewshot.py` (model name `xline+ae_xline`: retrieval with the cross-cell-line profile,
128,006 trained parameters per target line, U, V ∈ ℝ^{2000×32}, plus the linear model with 206,500 fitted numbers
at the selected K = 50).

This repository contains the **inference and evaluation code** and the **trained weights** used in the paper.
The training code is not included.

## Install

```bash
git clone https://github.com/Kent-AI-Lab/Pertrieve.git && cd Pertrieve
pip install -e .                 # numpy, pandas, scipy, h5py, scikit-learn
pip install -e ".[test]"         # + pytest
pip install -e ".[cell-eval]"    # optional: official cell-eval cross-check; needed for de_fewshot.py
pip install -e ".[slim]"         # optional: SLIM–STRING baselines
pytest -q tests                  # unit tests, no data needed
```

## Data

The scripts expect a `data/` directory (see [DATA.md](DATA.md)):

```
data/
  replogle_nadig_pseudobulk.npz       mean expression per (cell line, perturbation), 2,000 HVGs
  representations/{gobert,genept,esm2,scgpt,random,gobert_shuf}.npz
  fewshot_bio/                        State checkpoint predictions, real-cell DE tables, Hallmark sets (Fig. 5)
  annotations/                        only for scripts/analyze_retrieval.py
```

The `data/` directory and the predictions of the released models are attached to the
[v0.4.0 release](https://github.com/Kent-AI-Lab/Pertrieve/releases/tag/v0.4.0) (each file below 2 GB; `SHA256SUMS`
lists their checksums). Unpack them in the repository root:

```bash
sha256sum -c SHA256SUMS                                            # optional check
tar -xzf pertrieve_data.tar.gz                                     # -> data/
for f in pertrieve_predictions_*.tar.gz; do tar -xzf "$f"; done    # -> predictions/ (optional)
```

The predictions are optional: the scripts below write them again. `scripts/build_pseudobulk.py` rebuilds the
pseudobulk file from State's preprocessed Replogle/Nadig AnnData.

## Reproduce the held-out-gene results

```bash
bash scripts/reproduce_heldout.sh            # 30 min wall, 7 CPU-hours (shared 32-core server)
```

This predicts the held-out genes of the Replogle/Nadig screens (K562, HepG2, Jurkat; RPE1 responses are used
only for model selection) with every released model and the closed-form baselines, scores them, and compares the
results with `expected/`. As in every script of this repository, the outputs hold the models, comparisons and metrics
reported in the paper. Outputs go to `results/heldout/<protocol>/<split>/`:

| File | Content |
|---|---|
| `summary.csv` | mean response Pearson, cosine discrimination and MSE over the three test lines, as reported for each model |
| `per_line.csv` | the same per cell line |
| `paired.csv` | Pertrieve-Retrieval minus the comparators of the paper: line-weighted paired differences (mean within each line, averaged over lines) with gene-clustered bootstrap 95% intervals (B = 2,000) and Wilcoxon p (`pertrieve.stats.paired_lw`) |

| Paper item | Protocol / split | Models |
|---|---|---|
| Table 1; Supplement Tables 9–12 and 17 | `gd` / `unseen`, `cluster`, `cluster_genept` | `pertrieve` (5 seeds), `soft_knn_init`, GoBERT kNN, context mean, `genepert_ridge_gobert`, `ae_linear_{gobert,all4}`, `slim_string_{default,selected}` |
| Table 2; Supplement Tables 43–45 (representations) | `gd` / all three | `only_{gobert,genept,esm2,scgpt,random,gobert_shuf}`, `no_gobert` |
| Fig. 2; Supplement Tables 13–15 (repeated outer splits; `scripts/repeat_heldout.py`) | five outer splits per task, `splits/repeat/` | `pertrieve` (3 seeds), `soft_knn_init`, GoBERT kNN, `ae_linear_all4`, SLIM–STRING |
| Supplement Tables 16 and 46 (original split, components) | `full` / `unseen`, `cluster` | `pertrieve`, `ablation_{no_mixture,no_bilinear,temperature_only}` |

Splits: `unseen` = random genes, `cluster` = GoBERT functional clusters, `cluster_genept` = GenePT clusters.
Protocol `gd` (selection genes separate from evaluation genes) selects settings on 40% of the held-out genes (RPE1) and evaluates on the other
60% (`splits/heldout/<split>.{select,eval}.txt`); `full` uses all held-out genes. Held-out genes are excluded
from every reference library.

Baselines: `scripts/baselines_heldout.py` fits GenePert ridge regression and the linear model of Ahlmann-Eltze et
al. in closed form, with hyper-parameters chosen on the RPE1 selection genes. SLIM–STRING runs SLIM's released
code (`pip install -e ".[slim]"`) with SLIM's STRING embeddings (`data/gene_string_embeddings.v0.3.h5` in the SLIM
repository): `SLIM_STRING_H5=/path/to/gene_string_embeddings.v0.3.h5 bash scripts/reproduce_heldout.sh`. SLIM's PCA
uses an unseeded randomized SVD; the wrapper seeds it, and the paper's values (from an unseeded run) agree to about
0.001; the (K, λ) selected for the paper are stored in `splits/heldout/slim_string_selected.json`.

Single steps:

```bash
python scripts/predict_heldout.py   --split unseen --protocol gd --model pertrieve
python scripts/baselines_heldout.py --split unseen --protocol gd
python scripts/evaluate_heldout.py  --split unseen --protocol gd [--cell-eval]
```

Repeated outer splits (Fig. 2; Supplement Section E, Tables 13–15; `scripts/repeat_heldout.py`, run by
`reproduce_heldout.sh`): each task was split five times by the rule that produced it, with only the random seed
changed (`splits/repeat/<task>_s<k>.*`; outer split 0 is the split above). Each split has its own Pertrieve-Retrieval
models (three seeds) and untrained initialization (`weights/repeat/<task>_s<k>/`). GoBERT kNN and the Ahlmann-Eltze
linear model on the four representations are refitted and selected on the split's RPE1 selection genes; SLIM–STRING
is added when `SLIM_STRING_H5` is set, with the settings selected for the paper (`splits/repeat/selection.json`;
`--reselect` repeats the grid, about 4 min per split). Outputs `results/repeat/`: response Pearson per split, paired
differences with intervals per split, and their mean, SD and number of positive splits across the five splits.

## Other benchmarks

```bash
bash scripts/reproduce_other.sh              # 2.5 h wall, 38 CPU-hours (same server), then check_expected.py
```

Each script reads released weights from `weights/<benchmark>/`, predicts, and scores with the benchmark's metric.

| Paper item | Script | Data (`data/…`) |
|---|---|---|
| Table 3, Fig. 4; Supplement Tables 18–19 and 48, State rows of Table 7 (State few-shot; primary = `xline+ae_xline`) | `predict_fewshot.py --variant {gene,xline,xonly,xline0}` (and `--val`), `fewshot_baselines.py`, `fuse_fewshot.py --partner {ae_xline,ridge_xline} --rule {pooled,per-line}` (and `--retrieval {xline0,ridge_xline}`), then `evaluate_fewshot.py --extra State=…` | pseudobulk + representations; `fewshot_bio/state_predictions/` for the State row |
| Fig. 5; Supplement Tables 26–29 (performance by response breadth; Hallmark set recovery) | after the few-shot steps: `breadth_fewshot.py`, then `hallmark_fewshot.py` | `fewshot_bio/` |
| Fig. 3; Supplement Tables 34–35, Fig. S4 (cross-cell-line transfer of the learned similarity) | `transfer_heldout.py --split {unseen,cluster,cluster_genept}` | pseudobulk + representations; weights in `weights/heldout_line/` |
| Table 4; Supplement Tables 21–22 (fusion confirmation on the PT-RAG split and the CRISPRa screen; Table 22: Pearson, MSE, residual Pearson and cosine discrimination) | `confirm_fusion.py --task {ptrag,kaden}` | `ptrag/`, `fewshot_confirm/kaden/` |
| Supplement Table 20 (DE recovery after validation calibration) | after the few-shot steps: `de_fewshot.py --h5ad replogle.h5ad --step calibrate`, then `--step test` (variants `xline+ae_xline`, `xline`, `xonly`) | State's `replogle.h5ad` (cells); needs `cell-eval` |
| Supplement Table 62 (SLIM's official protocol) | `run_slim_protocol.py` | `slim_protocol/` |
| Supplement Section M, Fig. S6, Table 38 (GEARS / scGenePT combinations) | `run_gears_format.py --bench {gears_norman,scgenept_norman,scgenept_adamson} --variant {plain,additive,split_joint,split_double}` | `gears_format/<bench>/` |
| Supplement Tables 39–40 (Scouter) | `run_scouter_format.py --dataset {adamson,norman,wessels} --variant …`, then `scouter_familiar.py` | `scouter_format/<dataset>/` |
| Supplement Table 37, Norman rows (combination variant chosen per run on validation conditions) | after the GEARS-format and Scouter steps: `select_combo_variant.py` | `splits/combo_variant_selection.csv` |
| Supplement Section L, Table 37 (PerturbQA) | `run_perturbqa.py` | `perturbqa/` |
| Supplement Section N, Tables 41–42 (CellSimBench) | `run_cellsimbench.py --ds <dataset>`, then the CellSimBench scorer; statistics: `cellsimbench_stats.py` | `cellsimbench/<dataset>/` |
| Supplement Section I, Table 23, PT-RAG rows of Table 7 (PT-RAG comparison, PT-RAG's few-shot splits and gene space) | `run_ptrag.py` | `ptrag/` |
| Section 3.3; Supplement Section A.2, Table 65 (per-query magnitude) | after the few-shot steps and `reproduce_heldout.sh`: `magnitude.py` | predictions of both |

Few-shot: `--val` predicts each line's seeded 10% validation genes with models fitted without them
(`weights/fewshot/<line>/val/`); the fusions and the DE calibration choose their weight or gain on these genes only.
`fuse_fewshot.py` writes the fused test predictions and, with the same weight, the fused validation predictions that
`de_fewshot.py` calibrates the primary model on. The primary model is the pooled fusion with the Ahlmann-Eltze
partner (`xline+ae_xline`); the ridge partner and the per-line rule are exploratory rows. `evaluate_fewshot.py`
makes its line-weighted paired comparisons against the primary model. `fewshot_baselines.py` fits the closed-form
baselines (GenePert ridge on the four representations; ridge and Ahlmann-Eltze linear models on the cross-cell-line
profile); `--slim-string <h5>` adds SLIM–STRING (needs SLIM; its fits are not deterministic, so reruns differ by
about 3e-4). `evaluate_fewshot.py --extra NAME=PATTERN` scores predictions of other models with the same pseudobulk
scorer; `reproduce_other.sh` scores the State checkpoint this way (`data/fewshot_bio/state_predictions/<line>.npz`,
derived from State's released evaluation output): State's response is its predicted mean minus its own predicted
control mean, the convention of cell-eval; the row `State (measured control)` subtracts the measured control instead
(Supplement Table 7). Variant `xline0` is the untrained-retrieval control: the xline
model at its initialization (equal cosine weights on the profile and GoBERT, no bilinear term, output scale fitted by
leave-one-out least squares on the library; `pertrieve.untrained`, `Pertrieve.fit_scale`); `--retrieval xline0`
fuses it with the linear model ("ae + init." in Table 4); `--retrieval ridge_xline` fuses the two linear
models ("ae + ridge" in Table 4). `de_fewshot.py` turns mean predictions into
pseudo-cells and scores them with cell-eval (about 6–7 h for the three variants on four CPU threads; `--lines` runs
lines in parallel: about 1 h for one variant with one process per line).

Response breadth (`breadth_fewshot.py`): strata per line from the number of true DE genes of each test perturbation
(cell-eval Wilcoxon test on the real test cells, FDR < 0.05; fewer than 10 = "minimal", the rest at tertiles into
narrow / medium / broad), Pearson over all genes, sign agreement and MSE on the true DE genes, AUPRC of the DE genes
ranked by |predicted change|, with the sensitivity rules of the Supplement; `hallmark_fewshot.py` scores the 38
Hallmark sets with at least 15 genes in the 2,000-gene space. Both compare Pertrieve with State, the linear model, the
linear model fused with untrained retrieval and the retrieval component with line-weighted paired statistics.

Cross-cell-line transfer (`transfer_heldout.py`): `weights/heldout_line/<split>/transfer_hold<line>_s<seed>` were trained
on the `gd` libraries of the other three lines only (the held-out line is never a training query or key)
with the per-line output scales fitted afterward; `init_hold<line>` is the soft GoBERT kNN initialization with the
same scales. The script applies them to the evaluation genes of the held-out line with its own library and compares
them with the released four-cell-line model of the same seed.

Fusion confirmation (`confirm_fusion.py`): the fusion protocol fixed before testing, re-run on the PT-RAG few-shot
split (`weights/ptrag/`) and on the CRISPRa transcription-factor screen of Kaden et al. (fibroblasts and RPE1, five
folds per line, `weights/fewshot_confirm/kaden/`): Pertrieve-Retrieval, the linear model and their fusion (Pertrieve),
and the control fusions of the linear model with cross-cell-line ridge and with the untrained retrieval, each with one
pooled weight per seed chosen on the validation units.

SLIM's protocol: the SLIM–STRING row scores the predictions of SLIM's official pipeline (K = 10, λ = 0.1), provided
in `slim_protocol/slim_official/`, with the same code as Pertrieve.

CellSimBench: `run_cellsimbench.py` writes predictions in the format of CellSimBench's precomputed models; score them
with the unchanged `cellsimbench benchmark` of the CellSimBench release. `cellsimbench_stats.py` compares
Pertrieve-Retrieval (seed mean) with the best non-oracle model per metric (paired Wilcoxon over perturbations, Holm)
from the scorer output of the runs used in the paper (`cellsimbench/<ds>/scores.csv.gz`);
`--scores <ds>=<run>/detailed_metrics.csv` analyses a new run, in which the predictions must be registered as models
`retrieval_plain_s0..2` (single-gene datasets) / `retrieval_auto_s0..2` (Norman, Wessels).

PT-RAG: Pertrieve, Pertrieve-Retrieval with the cross-cell-line profile and the context mean on PT-RAG's own few-shot
splits, scored in PT-RAG's 2,000-gene output space against PT-RAG's released test predictions
(huggingface.co/difra00/PT-RAG), with line-weighted paired statistics (Holm over the four metrics of Supplement Table 23).
PT-RAG generates cells, so its response is the mean of its predicted cells minus the mean of its predicted control
cells, as for State; the row `PT-RAG (measured control)` subtracts the measured control instead (Supplement Table 7).

Per-query magnitude (`magnitude.py`): Spearman correlation between the norms of the predicted and the measured
responses across the test perturbations of each line (norms averaged over seeds, lines weighted equally), for
Pertrieve and its two components on the official few-shot test sets and for Pertrieve-Retrieval (seeds 0–2) on the
held-out-gene splits of Table 1. The bootstrap intervals of the two differences agree with Supplement Table 65 to
within 0.001 (the paper drew its resamples in a different order).

Combination variants (`pertrieve/combos.py`): `plain` retrieves whole training conditions; `additive` composes
measured or retrieved single-gene effects; `split_joint` / `split_double` first fit per-gene effects in closed form
from training singles and doubles (or doubles only) and retrieve only genes outside that fitted library. For the
Norman rows of Supplement Table 37, each run (split × seed) of GEARS-Norman, scGenePT-Norman and Scouter-Norman uses
the variant among `plain`, `additive` and `split_joint` with the highest validation score that the run logged during
training (its epoch-selection criterion); `splits/combo_variant_selection.csv` lists these scores and choices, and
`select_combo_variant.py` averages the test results of the chosen variants.

## What the model retrieves

```bash
python scripts/analyze_retrieval.py --annotations data/annotations
```

Compares the retrieval weights of the trained models with those of their untrained initialization (soft GoBERT
kNN) on annotations the model never uses (Complex Portal complexes, STRING physical interactions), after mapping the
expression data's gene symbols to current HGNC symbols, and lists the top references of example genes (Supplement
Section P, Tables 53–54 and 58; line-weighted means and paired differences). Download links for the annotation files
are in the script header.

## Use a released model

```python
import numpy as np
from pertrieve import Pertrieve, Pseudobulk, load_representations

model = Pertrieve.load("weights/heldout/gd/unseen/pertrieve_s0")
pb = Pseudobulk("data/replogle_nadig_pseudobulk.npz")
reps = load_representations("data/representations", model.representations)

line, queries = "k562", ["POLR1B", "MED7"]
library = [g for g in pb.genes(line) if g not in queries]          # measured references
Q = [r.matrix(queries) for r in reps]
K = [r.matrix(library) for r in reps]
delta = model.predict(Q, K, pb.delta(line, library), line)          # [2, 2000] response changes
weights = model.weights(Q, K)                                       # which references were used
print([library[i] for i in np.argsort(-weights[0])[:5]])
```

Queries need at least the first representation (GoBERT); a query without it receives the mean response of the
library.

## Statistics

`pertrieve.stats.paired_lw` is the paired comparison of the paper: the mean difference within each cell line,
averaged over lines with equal weight (the same estimand as the table means), with a 95% percentile bootstrap over
genes (all rows of a gene, in every line, resampled together; B = 2,000, seed 0) and a Wilcoxon test on the per-gene
mean differences. Its output also carries the pooled estimate (`pooled`), the per-line differences and whether every
line favors the first model. `pertrieve.stats.paired` is the pooled gene-clustered version (used where a single
population is compared, e.g. the retrieval analysis). `holm` adjusts p-values.

## Repository layout

```
pertrieve/       model.py (inference, kNN, untrained model, scale fit), data.py (incl. cross-cell-line profiles), combos.py,
                 compose.py, metrics.py, stats.py, baselines.py (closed-form GenePert / Ahlmann-Eltze baselines)
scripts/         predict_* / evaluate_* / run_* per benchmark, baselines and fusions, de_fewshot.py,
                 breadth_fewshot.py, hallmark_fewshot.py, transfer_heldout.py, repeat_heldout.py, confirm_fusion.py, magnitude.py,
                 cellsimbench_stats.py, select_combo_variant.py, analyze_retrieval.py, build_pseudobulk.py,
                 reproduce_heldout.sh, reproduce_other.sh, check_expected.py
weights/         trained models: <name>_s<seed>.npz (w, tau, U, V) + .json (scales, representations)
splits/          gene lists and splits of every benchmark (repeat/: the five outer splits per held-out-gene task),
                 validation-selected kNN k and baseline settings, combination variant chosen per run
expected/        reference values (paper numbers) for scripts/check_expected.py
tests/           unit tests
```

## Citation

Pertrieve: Learning to Reuse Measured Responses for Genetic Perturbation Prediction. Manuscript, 2026.

## License

MIT, see [LICENSE](LICENSE).
