# ============================================================================
# 08_MODEL_XGBOOST.R - XGBoost Model Training
# ============================================================================
# Description: Trains XGBoost models using both basic training and
#              tidymodels workflow with Bayesian hyperparameter tuning.
# ============================================================================

source("00_config.R")
library(xgboost)
library(tidymodels)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. BASIC XGBOOST MODEL
# ============================================================================

cat("Training basic XGBoost model...\n")

# Convert factors to numeric for XGBoost
tr_xgb <- as.data.frame(tr_base)
for (var in names(tr_xgb)) {
  if (is.factor(tr_xgb[[var]])) {
    tr_xgb[[var]] <- as.numeric(as.character(tr_xgb[[var]]))
  }
}

freez_xgb <- as.data.frame(freez_base)
for (var in names(freez_xgb)) {
  if (is.factor(freez_xgb[[var]])) {
    freez_xgb[[var]] <- as.numeric(as.character(freez_xgb[[var]]))
  }
}

# Prepare matrices
features <- tr_xgb[, names(tr_xgb) != "Energy"]
train_matrix <- xgb.DMatrix(data = as.matrix(features), label = tr_xgb$Energy)

# Parameters
params <- list(
  objective = "reg:squarederror",
  eval_metric = "mae"
)

# Train model
set.seed(GLOBAL_SEED)
model_xgb_basic <- xgb.train(params, train_matrix, nrounds = 1000)

# Evaluate
features_freez <- freez_xgb[, names(freez_xgb) != "Energy"]
preds <- predict(model_xgb_basic, as.matrix(features_freez))
evaluate_model(freez_base$Energy, preds, "XGBoost Basic")

# Save predictions
xgb_basic_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(xgb_basic_preds, "freez_base_xgboost_basic.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 2. XGBOOST WITH TIDYMODELS AND BAYESIAN TUNING
# ============================================================================

cat("\nTraining XGBoost with tidymodels...\n")

# Define model specification
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

# Create recipe
xgb_recipe <- recipe(Energy ~ ., data = tr_base)

# Create workflow
xgb_workflow <- workflow() %>%
  add_recipe(xgb_recipe) %>%
  add_model(xgb_spec)

# Set up parameter space
xgb_params <- parameters(
  trees(),
  learn_rate(),
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), tr_base)
)

xgb_params <- xgb_params %>%
  update(trees = trees(c(1000, 2000)))

# Create cross-validation folds
set.seed(GLOBAL_SEED)
cv_folds <- vfold_cv(tr_base, v = CV_FOLDS, strata = Energy)

# Run Bayesian tuning
cat("Running Bayesian hyperparameter tuning...\n")
set.seed(GLOBAL_SEED)
xgb_tune <- xgb_workflow %>%
  tune_bayes(
    resamples = cv_folds,
    param_info = xgb_params,
    iter = 10,
    metrics = metric_set(mape),
    control = control_bayes(no_improve = 50, save_pred = TRUE, verbose = TRUE)
  )

# Select best model
best_xgb <- select_best(xgb_tune, "mape", maximize = FALSE)
cat("\nBest parameters:\n")
print(best_xgb)

# Finalize workflow
final_xgb_workflow <- finalize_workflow(xgb_workflow, best_xgb)

# Fit final model
cat("\nFitting final XGBoost model...\n")
final_xgb_fit <- fit(final_xgb_workflow, data = tr_base)

# Evaluate
preds <- predict(final_xgb_fit, freez_base)$.pred
evaluate_model(freez_base$Energy, preds, "XGBoost Tuned (Tidymodels)")

# Save predictions
xgb_tuned_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(xgb_tuned_preds, "freez_base_xgboost_tuned.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 3. SAVE MODELS
# ============================================================================

cat("\nSaving models...\n")

xgb.save(model_xgb_basic, "model_xgb_basic.model")
saveRDS(final_xgb_fit, "model_xgb_tuned.rds")

cat("XGBoost training complete.\n")
