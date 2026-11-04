# Data

## Replogle/Nadig pseudobulk

`replogle_nadig_pseudobulk.npz`: one row per (cell line, perturbation), fields `cell_lines`, `perts`, `n`
(cells), `mean_hvg` (mean of the 2,000 highly variable genes in log1p space, over all cells of the group,
including the `non-targeting` controls). Cell lines: K562 and RPE1 (Replogle et al. 2022), HepG2 and Jurkat
(Nadig et al. 2025).

Built from the preprocessed AnnData released with State (Adduri et al. 2026; `replogle.h5ad`, cells in
`obs['cell_line']` and `obs['gene']`, expression in `obsm['X_hvg']`):

```bash
python scripts/build_pseudobulk.py --h5ad replogle.h5ad --out data/replogle_nadig_pseudobulk.npz
```

Response change of perturbation k in line c: Δ_k = mean_hvg[c, k] − mean_hvg[c, non-targeting].

## Gene representations

`representations/<name>.npz`: fields `genes` (symbols as used in the screens) and `X` [n, d]. Vectors are
L2-normalized on load; an all-zero row means the gene has no representation.

| Name | d | Source |
|---|---|---|
| `gobert` | 1024 | GoBERT (Miao et al. 2025): the gene's GO term IDs as one sequence → last hidden layer → mean over GO-term tokens |
| `genept` | 1536 | GenePT v2 (Chen & Zou 2025): OpenAI text embedding of the NCBI/UniProt gene summary |
| `esm2` | 1280 | ESM2-650M (`esm2_t33_650M_UR50D`): mean residue embedding (last layer) of the longest reviewed UniProt protein; proteins over 1,022 aa: mean over chunks |
| `scgpt` | 512 | scGPT whole-human pretrained model: gene-token embedding (`encoder.embedding`) |
| `random` | 1024 | control: N(0, I) vector per gene (seed 0) |
| `gobert_shuf` | 1024 | control: GoBERT vectors with gene labels permuted (seed 0) among genes that have one |

Gene symbols were matched as given, then through HGNC previous and alias symbols.

SLIM–STRING (baseline only) uses SLIM's own STRING gene embeddings (`data/gene_string_embeddings.v0.3.h5` in
github.com/RasmussenLab/SLIM); they are not redistributed here.

## Cells (DE evaluation only)

`scripts/de_fewshot.py` (Supplement Table 20) reads cells from State's preprocessed `replogle.h5ad` (the file
`build_pseudobulk.py` uses): `obsm['X_hvg']` clipped to [0, 14]. It is not redistributed (22 GB); pass its path with
`--h5ad`. Validation gene lists are `splits/fewshot/<line>.val_s<seed>.txt` (the seeded 10% of each line's
library) and the validation-fit weights are in `weights/fewshot/<line>/val/`.

## Annotations (retrieval analysis only)

| File | Source |
|---|---|
| `9606.tsv` | Complex Portal, human complexes: https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv |
| `9606.protein.physical.links.v12.0.txt.gz` | STRING v12 physical links |
| `9606.protein.info.v12.0.txt.gz` | STRING v12 protein info |
| `goa_human.gaf.gz` | GO annotations (used only to map UniProt accessions to gene symbols) |
| `hgnc_complete_set.txt` | HGNC complete set (downloaded 2026-10-24), maps the older gene symbols of the expression data to current symbols before partners are matched |

## Other benchmarks

Per-condition means are the mean of the benchmark's own expression matrix (`adata.X`) over all cells of the
condition; `ctrl` is the control. Splits and folds are the benchmarks' own (copied into `splits/`).

| Directory | Source | Contents |
|---|---|---|
| `slim_protocol/<dataset>.npz` | GEARS `replogle_{k562,rpe1}_essential` (Harvard Dataverse files 7458695 / 7458694), as used by SLIM | `conds`, `mean`; splits = GEARS `simulation` split seeds 1–5 |
| `slim_protocol/slim_official/<dataset>_seed<s>.npz` | SLIM's official pipeline (github.com/RasmussenLab/SLIM, commit 5a7e9ade; STRING, K = 10, λ = 0.1) run on these splits; mean of its synthetic population per test condition | `genes`, `pred_mean`, `source` |
| `gears_format/<bench>/` | GEARS (Norman, small GO graph) and scGenePT (Norman, Adamson) data releases | `means.npz`, `representations/` (GoBERT, the benchmark model's own: GO-binary or GO_all-ada, GenePT_v2) |
| `scouter_format/<dataset>/` | Scouter's preprocessing of Adamson, Norman and Wessels | `means.npz`, `degs.npz` (top-20 non-dropout DE genes per condition), `representations/` |
| `perturbqa/` | PerturbQA release (`datasets/<line>-{de,dir}.csv`) | `datasets/`, `representations/{GoBERT,GenePT_v2}.npz` |
| `cellsimbench/<dataset>/` | CellSimBench processed datasets (`<ds>_processed_complete.h5ad`) | `data.npz` (`conds`, `mean`, `ctrl` = `uns['ctrl_baseline']`, `covariate`, `var_names`), `representations/` (GoBERT + the benchmark's own GenePT, ESM2 and scGPT embeddings) |
| `cellsimbench/<dataset>/scores.csv.gz` | CellSimBench scorer output (`detailed_metrics.csv`) of the runs used in the paper (run id in `scores_run.txt`) | 13 metrics per perturbation; Pertrieve (3 seeds) and every non-oracle model |
| `scouter_format/norman/scouter_published.csv` | Scouter reproducibility repository (MIT), `results/norman.csv`, rows with Method == "Scouter" | per-condition DE NormMSE and 1 − PCC, split, subgroup |
| `ptrag/pseudobulk.npz` | `replogle.h5ad` X (log1p, all cells per group) restricted to PT-RAG's 2,000 output genes (`genes`); equal to the means of PT-RAG's `adata_real` | `cell_lines`, `perts`, `n`, `mean_hvg`, `genes` |
| `ptrag/ptrag_predictions/<line>.npz` | PT-RAG's released test predictions (huggingface.co/difra00/PT-RAG, `<model>/eval_last.ckpt/adata_pred.h5ad`), mean over predicted cells per perturbation; the `non-targeting` row is PT-RAG's predicted control, from which its responses are taken | `perts`, `pred_mean`, `n_cells`, `source` |
| `fewshot_confirm/kaden/pseudobulk.npz` | CRISPRa transcription-factor screen of Kaden et al. 2025 (fibroblast and RPE1), as processed by CellSimBench (`kaden25fibroblast`, `kaden25rpe1`): the 1,836 activated factors measured in both lines, mean log-expression per (line, factor) over the 4,985 shared genes; `non-targeting` = control (used by `scripts/confirm_fusion.py --task kaden`; representations: `fewshot_confirm/kaden/representations/` (GoBERT plus CellSimBench's GenePT, ESM2, scGPT vectors of these genes); folds: `splits/fewshot_confirm/kaden/`) | `cell_lines`, `perts`, `n`, `mean_hvg`, `genes` |

Raw gene vectors are stored for these benchmarks; a condition's representation is built from them as described
in each script.

## Few-shot response-breadth analyses (`fewshot_bio/`; Fig. 5, Supplement Section J)

| File | Source | Contents |
|---|---|---|
| `state_predictions/<line>.npz` | State's released evaluation output on its official few-shot test set (`ST-HVG-Replogle` checkpoint, `<line>_pred_de.csv`: predicted mean and control mean per perturbation and HVG) | `genes` (test perturbations), `features` (the 2,000 HVGs), `delta` = log1p(predicted mean) − log1p(State's control mean), `ctrl_log1p` (State's control; gives the measured-control row of Supplement Table 7) |
| `real_de/<line>_test_de.csv.gz` | cell-eval Wilcoxon test of each test perturbation's real cells against the line's control cells (per-perturbation BH), rows with FDR < 0.05 only | `target`, `gene`, `log2_fold_change`, `fdr` |
| `real_de_counts.csv` | the same tests, counts per perturbation | `n_de` (FDR < 0.05), `n_de_fc` (FDR < 0.05 and |log2 FC| > log2 1.5), `n_de_global` (FDR < 0.05 after BH over the whole line), `n_cells`, `n_ctrl` |
| `cells_umi_per_pert.csv` | the preprocessed AnnData | cells and mean / median UMI per (line, perturbation) |
| `hvg_genes.txt` | the preprocessed AnnData (`var` of `X_hvg`) | the 2,000 HVGs in the column order of the pseudobulk file |
| `MSigDB_Hallmark_2020.txt` | MSigDB Hallmark 2020 (Enrichr library format; CC BY 4.0) | one set per line: name, description, genes |

## Held-out-line transfer weights (`weights/heldout_line/`; Fig. 3, Supplement Section K.4)

Not data but noted here: `<split>/transfer_hold<line>_s<seed>.{npz,json}` are similarity parameters trained on the
`gd` libraries of the three lines other than `<line>` (epoch selected on the RPE1 selection genes, K562's
when the held-out line is RPE1), with the four per-line output scales fitted afterward by leave-one-out least
squares on each line's library; `<split>/init_hold<line>` is the soft GoBERT kNN initialization with the same scales.
