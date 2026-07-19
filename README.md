# 5G Base Station Energy Consumption Prediction

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [AI/ML for 5G-Energy Consumption Modelling by ITU AI/ML in 5G Challenge 🌾 - CHF 20 000 USD](https://zindi.africa/competitions/aiml-for-5g-energy-consumption-modelling)

Ranked in the TOP 51%
---

Predicting energy consumption for 5G base stations from network load, energy-saving mode states, hardware configuration (antenna count, RU type, frequency/bandwidth), and time-of-day patterns. Competition dataset (2023), tabular regression.

## Key Engineering Decisions

**Target construction via frequency-weighted energy disaggregation.** The raw data records total energy at the base station level. For multi-cell stations (two cells operating at different frequencies), no per-cell energy measurement exists. To enable cell-level modeling, energy is attributed to each cell using frequency-specific proportions derived from hardware characteristics — `AVG_365`, `AVG_426`, `AVG_155`, `AVG_189`. Predicting cell energy is only possible after this attribution step. Without it, the label itself is undefined at the modeling granularity.

```r
energy <- energy %>%
  mutate(
    perct = case_when(
      nb_cell == 2 & freq_cat == '365'    ~ AVG_365,
      nb_cell == 2 & freq_cat == '426.98' ~ AVG_426,
      nb_cell == 2 & freq_cat == '155.6'  ~ AVG_155,
      nb_cell == 2 & freq_cat == '189'    ~ AVG_189,
      TRUE ~ 1
    ),
    target = Energy * perct
  )
```

**Domain-derived indicator features from hardware rules.** Features like `es1_load_low`, `es3_load_high`, `tree_energy_pack_F` are explicit interaction rules between energy-saving mode states, load levels, antenna counts, and frequencies. These encode known hardware behavior (e.g. ES1 activates under very low load with large antenna arrays) as binary flags rather than relying on the model to rediscover the interaction.

**Per-variable encoding strategy.** Each categorical variable receives an encoding matched to its structure: Helmert contrasts for `day_of_week` (ordered, sequential), polynomial contrasts for `RUType` (ordered by hardware generation), backward difference for `type_anten`, WoE for binary indicator flags, M-estimator smoothed encoding for `load_hour`. A separate dataset (`05_build_lightgbm_dataset.R`) is built with these pre-encoded representations specifically for LightGBM.

**Two-level modeling: base station and cell.** Models are trained independently at both granularities. Base-level models use aggregated cell features; cell-level models use individual cell observations. Final submission blends predictions from both levels.

**Three-way train / freeze / test split.** A dedicated freeze set is held out for ensemble weight optimization and model comparison, separate from both training and final test.

## Project Structure

```
00_config.R                  # Global settings and utility functions
01_data_loading.R            # Load raw data, apply hardware fixes, aggregate base info
02_feature_engineering.R     # Encoding functions and domain feature constructors
03_build_base_dataset.R      # Base-level dataset with aggregated cell features
04_build_cell_dataset.R      # Cell-level dataset with frequency-weighted energy split
05_build_lightgbm_dataset.R  # Pre-encoded dataset variant for LightGBM
06_train_test_split.R        # Train / freeze / test splits (base and cell)
07_model_lightgbm.R          # LightGBM with grid search
08_model_xgboost.R           # XGBoost with Bayesian tuning via tidymodels
09_model_catboost.R          # CatBoost with K-fold CV averaging
10_model_caret.R             # RF, GLMNet, GAM via caret
11_model_h2o.R               # H2O AutoML
12_model_stacking.R          # Tidymodels stacking ensemble
13_model_sl3.R               # SuperLearner (sl3) with NNLS metalearner
14_clustering_analysis.R     # Cluster features from load/ES mode patterns
15_model_cell_level.R        # LightGBM and XGBoost at cell granularity
16_generate_submission.R     # Ensemble and format final predictions
17_eda_and_visualization.R   # EDA reports
18_evaluate_freeze.R         # Model comparison on freeze set
MAIN.R                       # Pipeline orchestration
```

## Running the Pipeline

```r
source("MAIN.R")
run_pipeline()          # All steps

run_data_prep()         # Steps 1-6 only
run_training()          # Steps 7-15 only
run_evaluation()        # Steps 16-18 only

run_pipeline(steps = c(1, 3, 7))  # Specific steps
```

## Data Files Required

| File | Contents |
|---|---|
| `imgs_202307101549519358.csv` | Submission template |
| `imgs_2023071012133740345.csv` | Base station energy measurements |
| `imgs_2023071012130978799.csv` | Cell-level load and ES mode states |
| `imgs_2023071012123392536.csv` | Base station hardware configuration |

## Configuration

Key settings in `00_config.R`:

| Variable | Purpose |
|---|---|
| `GLOBAL_SEED` | Reproducibility seed |
| `FILE_SUBMISSION` | Path to submission template |
| `FILE_ENERGY` | Path to energy data |
| `FILE_CELL` | Path to cell data |
| `FILE_BASE` | Path to base station data |

Edit `00_config.R` to modify:
- `GLOBAL_SEED`: Random seed for reproducibility (default: 1618)
- `CV_FOLDS`: Number of cross-validation folds (default: 5)
- `TRAIN_SAMPLE_SIZE`: Training set size (default: 75000)
- `FREEZE_SAMPLE_SIZE`: Holdout validation size (default: 10000)
- `AVG_*`: Energy split proportions for multi-cell bases

## Output Files

### Intermediate Data Files
- `data_base_cleaned.RData`: Cleaned base station data
- `df_base.RData`: Base-level feature dataset
- `df_cell.RData`: Cell-level feature dataset
- `df_ligh.RData`: LightGBM-optimized dataset
- `splits_base.RData`: Train/test/freeze splits (base level)
- `splits_cell.RData`: Train/test/freeze splits (cell level)

### Model Files
- `model_lgb_*.txt`: LightGBM models
- `model_xgb_*.model`: XGBoost models
- `model_catboost_*.cbm`: CatBoost models
- `model_*_caret.rds`: Caret models
- `model_stacks.rds`: Stacked ensemble
- `model_sl3.rds`: Super Learner model

### Prediction Files
- `freez_*_*.csv`: Freeze set predictions for each model
- `submission_*.csv`: Final submission files

### Analysis Files
- `model_comparison_results.csv`: Model performance comparison
- `eda_results.rds`: EDA analysis results
- `cluster_summary.csv`: Clustering analysis results

## Dependencies

### Core Packages
- data.table, dplyr, tidyr, purrr

### Machine Learning
- caret, tidymodels, recipes, stacks
- xgboost, lightgbm, catboost
- ranger, glmnet, mgcv
- h2o, sl3
- treesnip, bonsai

### Parallel Processing
- parallel, doParallel, future

### Visualization & Analysis
- ggplot2, rpart, rpart.plot
- effectsize, ltm

## Original File Mapping

| Original File       | Reorganized To                           |
|---------------------|------------------------------------------|
| First.R             | 01, 02, 03, 17                           |
| Second.R            | 04                                       |
| Build_DF.R          | 03, 04, 05                               |
| Encoding.R          | 02, 05                                   |
| Freez.R             | 06, 18                                   |
| Caret.R             | 10                                       |
| Catboost.R          | 09                                       |
| draft.R, draf_2.R   | 07, 08                                   |
| ligh_random*.R      | 07                                       |
| ligh_recipes.R      | 07, 15                                   |
| h2o.R               | 11                                       |
| Model_Base.R        | 07, 08, 10, 11, 12, 13                   |
| Model_Cell.R        | 15                                       |
| Stacking.R          | 12, 18                                   |
| Clustering.R        | 14                                       |
| Submit.R            | 16                                       |
| Graphs.R            | 17                                       |

## Best Practices Applied

1. **Modular Design**: Each script has a single responsibility
2. **Configuration Separation**: All constants in 00_config.R
3. **Consistent Naming**: Files numbered in execution order
4. **Documentation**: Detailed comments and headers
5. **Error Handling**: Graceful handling of missing files
6. **Reproducibility**: Seeds set for all random operations
7. **Reusable Functions**: Helper functions in 02_feature_engineering.R
8. **Pipeline Orchestration**: MAIN.R for easy execution

## Notes

- The freeze set is used for local validation before submission
- Cell-level predictions need aggregation to base level for submission
- Some models (H2O, SL3) may require additional installation steps
- Run times vary significantly by model (LightGBM ~5min, H2O AutoML ~30min)
