# Auditing multisource crop-yield prediction on two public Kaggle datasets

Code, outputs and reproducibility material for the manuscript

> **When Yield-Prediction Benchmarks Carry No Learnable Signal: A Leakage-Safe, Uncertainty-Aware Audit of Multisource Machine Learning on Two Public Kaggle Datasets** (draft; [paper link / DOI to be added])

**Status:** research code accompanying a draft manuscript. Results reported here are from one Kaggle run dated 23 October 2026.

## What this repository shows

A written protocol asked whether fusing weather, soil, vegetation-index and engineered agro-climatic features improves yield prediction, and whether the models are robust, calibrated and interpretable. On the two public datasets analysed, **no learner meaningfully beat the training-mean predictor**:

| | D1 (AgriYield 2025) | D2 (Wheat Yield Prediction) |
|---|---|---|
| Rows | 10,000 | 200,000 |
| Non-trivial fits that beat the mean predictor (same fold and feature set) | 19.7% of 451 | 12.7% of 496 |
| Best single fit vs. mean predictor (relative RMSE) | 0.9985 | 0.99996 |
| Random forest / extra trees vs. mean predictor (median) | +1.8% / +1.9% | +0.4% / +0.4% |
| 90% conformal interval width as a share of the target range | about 90% | about 90% |

Feature-set effects were at most 0.4% of RMSE, with none significant after Holm correction (90 paired comparisons). The targets and predictors have moments consistent with independent uniform distributions. This is evidence of consistency only; **how the files were generated is not established** (see `docs/DATASETS.md`). The conclusions apply to these two files, not to crop-yield modelling in general.

## Repository layout

```
notebooks/agri_yield_experiment.ipynb   Kaggle notebook (the full pipeline, 19 cells)
src/agri_yield_experiment.py            The same code as one script (cells separated by "# %%")
scripts/summarize_results.py            Recomputes headline numbers and paper tables from saved CSVs
scripts/audit_moments.py                Moment-implied uniform bounds and correlation noise bounds
results/tables/                         All result tables (CSV/JSON), including per-fold metrics
results/figures/                        23 PNG figures (D1's empty Figure 10 panel is omitted)
results/audit/                          Per-dataset audit JSON, feature-block assignment, split availability
results/repro/                          Config, environment, split indices (.npz), feature sets, dataset sources
docs/                                   Dataset notes and protocol deviations
data/                                   Empty on purpose: raw data are not redistributed
```

## Reproduce the paper's numbers from the saved outputs (no raw data needed)

```bash
pip install pandas numpy
python scripts/summarize_results.py     # prints headline statistics, writes results/paper_tables.md
python scripts/audit_moments.py         # moment-implied bounds behind Table 2 of the paper
```

Fold indices in the printed output are 0-based (the paper counts from 1).

## Re-run the full experiment on Kaggle

1. Create a new Kaggle notebook and upload `notebooks/agri_yield_experiment.ipynb`.
2. **Add Input**, attaching:
   - competition **agriyield-2025** (accept the competition rules first) as D1
   - dataset **shaikasif89/wheat-yeild** as D2
   - dataset **abhinand05/crop-production-in-india** as D3
   - dataset **siddharthss/crop-recommendation-dataset** as D4
3. Settings: accelerator **GPU T4**.
4. Run the setup cells first. They print the resolved data paths, the feature-block assignment and a projected runtime. Set `PROFILE = "smoke"` for a short dry run, then `"full"`.
5. Outputs are written to `/kaggle/working/agri_outputs` and zipped to `/kaggle/working/agri_results.zip`.

Notes:

- Data discovery is dynamic. The run used the Kaggle layout `/kaggle/input/competitions/agriyield-2025` and `/kaggle/input/datasets/<owner>/<slug>`; the notebook searches `/kaggle/input` and prints what it finds. D2 is detected by its columns (Yield, NDVI, Latitude, Longitude, DayInSeason) and copied to `/kaggle/working/D2_wheat/train.csv`.
- Every fold is checkpointed. To resume after a crash, keep the session or attach the earlier run's output as an Input; finished folds are reused.
- A model that raises an error in a fold is skipped with a warning instead of aborting the run.
- Local runs: set `AGRI_INPUT` (folder with the datasets), `AGRI_OUT` and `AGRI_PROFILE` environment variables, then run `python src/agri_yield_experiment.py`. A GPU is optional; XGBoost, CatBoost and the MLP fall back to CPU.

## Environment used for the reported run

Python 3.13.15 on Kaggle with two Tesla T4 GPUs and 4 CPU cores; NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, XGBoost 3.4.1, LightGBM 4.6.0, CatBoost 1.2.10, Optuna 5.0.0, SHAP 0.52.0, PyTorch 2.11.0. Seed 42. Full record in `results/repro/environment.json` and `results/repro/config.json`. See `requirements.txt` for pinned versions.

## Method in brief

- **Feature sets:** E1 weather, E2 + soil, E3 + remote sensing, E4 + engineered agro-climatic features (E5 temporal summaries were not applicable).
- **Splits:** random (S1), group (S2, degenerate here because every field or coordinate occurs once), spatial cluster (S4, D2 only), extreme-climate holdout (S5). A temporal split (S3) was impossible.
- **Learners:** mean predictor, OLS, ridge, elastic net, random forest, extra trees, XGBoost, LightGBM, CatBoost, MLP, nested-Optuna-tuned XGBoost, and a non-negative-weight ensemble.
- **Uncertainty:** split conformal, ensemble-spread-normalised conformal, and conformalised quantile regression.
- **Robustness and explanation:** missing-data, noise, rainfall and temperature perturbations; TreeSHAP, interaction values and permutation importance.
- **Statistics:** pooled out-of-fold paired comparisons, bootstrap intervals, Wilcoxon tests, Holm correction.

## Limitations

Dataset provenance, units and licence terms were not available to the authors. D2 replaced the originally planned competition dataset (*kaggle-cup-2025-01*), which required an invitation. No label-permutation or shuffled-feature control experiments were run. The extreme-climate split is a single holdout. See the manuscript for the full list.

## Data and licences

Raw data are not included and must be obtained from the Kaggle sources in `docs/DATASETS.md`, subject to each source's terms and, for D1, the competition rules. Code in this repository is released under the MIT licence (see `LICENSE`). The result tables and figures were derived from the Kaggle data; confirm the terms of each source before reusing them.

## Citation

See `CITATION.cff` (placeholders to be completed once the manuscript has a DOI).

## Publishing this repository

```bash
cd agri-yield-audit
git init
git add .
git commit -m "Initial commit: code, results and reproducibility material"
git branch -M main
git remote add origin https://github.com/<your-account>/<repo-name>.git
git push -u origin main
```

Then archive a release on Zenodo (or similar) to obtain a DOI, and add the DOI to the manuscript's data and code availability statement.
