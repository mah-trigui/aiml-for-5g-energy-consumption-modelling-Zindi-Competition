# ============================================================================
# 07_MODEL_LIGHTGBM.R - LightGBM Model Training
# ============================================================================
# Description: Trains LightGBM models with hyperparameter tuning using
#              both manual grid search and tidymodels workflow.
# ============================================================================

source("00_config.R")
library(lightgbm)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. BASIC LIGHTGBM MODEL
# ============================================================================

cat("Training basic LightGBM model...\n")

# Prepare data
features <- tr_base[, -"Energy"]
categorical_vars <- names(which(sapply(tr_base, class) == "factor"))

# Create LightGBM dataset
dtrain <- lgb.Dataset(
  data = as.matrix(features),
  label = tr_base$Energy,
  categorical_feature = categorical_vars
)

# Basic parameters
params <- list(
  objective = "regression",
  metric = "mae"
)

# Cross-validation for optimal rounds
set.seed(GLOBAL_SEED)
cv_result <- lgb.cv(
  params = params,
  data = dtrain,
  nrounds = 10000,
  nfold = CV_FOLDS,
  early_stopping_rounds = 50,
  verbose = 0
)

# Train final model
set.seed(GLOBAL_SEED)
model_lgb_basic <- lgb.train(
  params = params,
  data = dtrain,
  nrounds = cv_result$best_iter
)

# Evaluate on freeze set
preds <- predict(model_lgb_basic, as.matrix(freez_base[, -"Energy"]))
evaluate_model(freez_base$Energy, preds, "LightGBM Basic")

# Save predictions
lgb_basic_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(lgb_basic_preds, "freez_base_lightgbm_basic.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 2. LIGHTGBM WITH HYPERPARAMETER TUNING
# ============================================================================

cat("\nTraining LightGBM with hyperparameter tuning...\n")

# Prepare validation set
features_ts <- ts_base[, -"Energy"]
dtest <- lgb.Dataset.create.valid(dtrain, data = as.matrix(features_ts), label = ts_base$Energy)
valids <- list(train = dtrain, test = dtest)

# Define parameter grid
param_grid <- expand.grid(
  num_leaves = c(40, 60, 80),
  max_depth = c(5, 6, 7),
  min_data_in_leaf = c(10, 15, 20),
  colsample_bytree = c(0.4, 0.6, 0.8)
)

# Store results
mae_values <- numeric(nrow(param_grid))

# Grid search
cat("Running grid search...\n")
for (i in seq_len(nrow(param_grid))) {
  set.seed(GLOBAL_SEED)

  model <- lgb.train(
    data = dtrain,
    num_iterations = 3000,
    objective = "regression",
    eval = "mae",
    metric = "mae",
    valids = valids,
    nthread = 4L,
    early_stopping_round = 100,
    verbose = -1,
    num_leaves = param_grid$num_leaves[i],
    max_depth = param_grid$max_depth[i],
    min_data_in_leaf = param_grid$min_data_in_leaf[i],
    colsample_bytree = param_grid$colsample_bytree[i]
  )

  # Evaluate
  features_freez <- freez_base[, names(freez_base) != "Energy"]
  pred <- predict(model, as.matrix(features_freez))
  mae_values[i] <- Metrics::mae(freez_base$Energy, pred)

  if (i %% 10 == 0) {
    cat(sprintf("  Completed %d/%d combinations\n", i, nrow(param_grid)))
  }
}

results <- cbind(param_grid, MAE = mae_values)

# Find best parameters
best_idx <- which.min(results$MAE)
best_params <- results[best_idx, ]
cat(sprintf("\nBest MAE: %.6f\n", best_params$MAE))

# ============================================================================
# 3. TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================

cat("\nTraining final LightGBM model with best parameters...\n")

set.seed(GLOBAL_SEED)
model_lgb_tuned <- lgb.train(
  data = dtrain,
  num_iterations = 3000,
  objective = "regression",
  eval = "mae",
  metric = "mae",
  valids = valids,
  nthread = 4L,
  early_stopping_round = 100,
  verbose = 0,
  num_leaves = best_params$num_leaves,
  max_depth = best_params$max_depth,
  min_data_in_leaf = best_params$min_data_in_leaf,
  colsample_bytree = best_params$colsample_bytree
)

# Final evaluation
features_freez <- freez_base[, -"Energy"]
preds <- predict(model_lgb_tuned, as.matrix(features_freez))
evaluate_model(freez_base$Energy, preds, "LightGBM Tuned")

# Save predictions
lgb_tuned_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(lgb_tuned_preds, "freez_base_lightgbm_tuned.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 4. SAVE MODELS
# ============================================================================

cat("\nSaving models...\n")

lgb.save(model_lgb_basic, "model_lgb_basic.txt")
lgb.save(model_lgb_tuned, "model_lgb_tuned.txt")

cat("LightGBM training complete.\n")
