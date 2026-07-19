# ============================================================================
# 15_MODEL_CELL_LEVEL.R - Cell-Level Model Training
# ============================================================================
# Description: Trains models at the cell level for more granular predictions.
#              Cell-level predictions are then aggregated to base level.
# ============================================================================

source("00_config.R")
library(lightgbm)
library(tidymodels)
library(bonsai)

# Load cell splits
load("splits_cell.RData")

# ============================================================================
# 1. LIGHTGBM FOR CELL DATA
# ============================================================================

cat("Training LightGBM on cell data...\n")

# Prepare data
features <- tr_cell[, names(tr_cell) != "Energy"]
categorical_vars <- names(which(sapply(tr_cell, class) == "factor"))

# Create LightGBM dataset
dtrain <- lgb.Dataset(
  data = as.matrix(features),
  label = tr_cell$Energy,
  categorical_feature = categorical_vars
)

# Parameters
params <- list(
  objective = "regression",
  metric = "mae"
)

# Cross-validation
set.seed(GLOBAL_SEED)
cv_result <- lgb.cv(
  params = params,
  data = dtrain,
  nrounds = 10000,
  nfold = 3,
  early_stopping_rounds = 50,
  verbose = 0
)

# Train model
set.seed(GLOBAL_SEED)
model_cell_lgb <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_result$best_iter
)

# Predict on freeze set
freez_features <- freez_cell_unique[, names(freez_cell_unique) != "Energy"]
preds <- predict(model_cell_lgb, as.matrix(freez_features))
evaluate_model(freez_cell_unique$Energy, preds, "Cell LightGBM")

# Save predictions
cell_lgb_preds <- data.frame(pred = preds, real = freez_cell_unique$Energy)
write.csv(cell_lgb_preds, "freez_cell_lightgbm.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 2. LIGHTGBM WITH TIDYMODELS/BONSAI
# ============================================================================

cat("\nTraining LightGBM with tidymodels...\n")

# Create stratification variable
tr_cell$strata <- as.factor(tr_cell$anten_cat)

# Create folds
set.seed(GLOBAL_SEED)
cell_folds <- vfold_cv(tr_cell, v = 5, strata = strata)
tr_cell$strata <- NULL

# Define recipe
cell_recipe <- recipe(Energy ~ ., data = tr_cell)

# Define model
lgb_spec <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  mtry = tune()
) %>%
  set_engine("lightgbm", metric = "mape") %>%
  set_mode("regression")

# Create workflow
lgb_wf <- workflow() %>%
  add_recipe(cell_recipe) %>%
  add_model(lgb_spec)

# Define grid
lgb_grid <- expand.grid(
  trees = c(250, 300, 500),
  tree_depth = c(4, 5, 6),
  min_n = c(5, 20),
  loss_reduction = c(0.1, 0.2),
  mtry = c(4, 5, 6)
)

# Tune
cat("  Running grid search...\n")
lgb_res <- lgb_wf %>%
  tune_grid(
    resamples = cell_folds,
    grid = lgb_grid,
    metrics = metric_set(mape),
    control = control_grid(verbose = FALSE)
  )

# Select best
best_lgb <- lgb_res %>% select_best("mape")
cat("\nBest parameters:\n")
print(best_lgb)

# Finalize and fit
final_lgb <- lgb_wf %>%
  finalize_workflow(best_lgb) %>%
  fit(data = tr_cell)

# Evaluate
preds_tidy <- predict(final_lgb, freez_cell_unique)$.pred
evaluate_model(freez_cell_unique$Energy, preds_tidy, "Cell LightGBM (Tidymodels)")

# Save predictions
cell_tidy_preds <- data.frame(pred = preds_tidy, real = freez_cell_unique$Energy)
write.csv(cell_tidy_preds, "freez_cell_lightgbm_tidy.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 3. XGBOOST WITH TIDYMODELS
# ============================================================================

cat("\nTraining XGBoost with tidymodels...\n")

# Define model
xgb_spec <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune(),
  mtry = tune()
) %>%
  set_mode("regression") %>%
  set_engine("xgboost", nthread = 4)

# Create workflow
xgb_wf <- workflow() %>%
  add_recipe(cell_recipe) %>%
  add_model(xgb_spec)

# Set up parameters
xgb_params <- parameters(
  trees(),
  learn_rate(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), tr_cell)
)

xgb_params <- xgb_params %>%
  update(trees = trees(c(1000, 2000)))

# Bayesian tuning
tr_cell$strata <- as.factor(tr_cell$anten_cat)
set.seed(GLOBAL_SEED)
cell_folds <- vfold_cv(tr_cell, v = 5, strata = strata)
tr_cell$strata <- NULL

set.seed(GLOBAL_SEED)
xgb_tune <- xgb_wf %>%
  tune_bayes(
    resamples = cell_folds,
    param_info = xgb_params,
    iter = 5,
    metrics = metric_set(mape),
    control = control_bayes(no_improve = 50, save_pred = TRUE, verbose = TRUE)
  )

# Select best
best_xgb <- select_best(xgb_tune, "mape", maximize = FALSE)
cat("\nBest XGBoost parameters:\n")
print(best_xgb)

# Finalize and fit
final_xgb <- finalize_workflow(xgb_wf, best_xgb)
final_xgb_fit <- fit(final_xgb, data = tr_cell)

# Evaluate
preds_xgb <- predict(final_xgb_fit, freez_cell_unique)$.pred
evaluate_model(freez_cell_unique$Energy, preds_xgb, "Cell XGBoost (Tidymodels)")

# Save predictions
cell_xgb_preds <- data.frame(pred = preds_xgb, real = freez_cell_unique$Energy)
write.csv(cell_xgb_preds, "freez_cell_xgboost_tidy.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 4. SAVE MODELS
# ============================================================================

cat("\nSaving cell-level models...\n")

lgb.save(model_cell_lgb, "model_cell_lgb.txt")
saveRDS(final_lgb, "model_cell_lgb_tidy.rds")
saveRDS(final_xgb_fit, "model_cell_xgb_tidy.rds")

cat("Cell-level model training complete.\n")
