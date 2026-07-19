# ============================================================================
# 12_MODEL_STACKING.R - Model Stacking and Ensemble Methods
# ============================================================================
# Description: Implements various model stacking and ensemble strategies
#              using tidymodels/stacks and custom blending approaches.
# ============================================================================

source("00_config.R")
library(tidymodels)
library(stacks)
library(ranger)
library(kernlab)
library(kknn)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. PREPARE DATA FOR STACKING
# ============================================================================

cat("Preparing data for stacking...\n")

# Split data
set.seed(GLOBAL_SEED)
data_split <- initial_split(tr_base)
train_data <- training(data_split)
test_data <- testing(data_split)

# Create cross-validation folds
set.seed(GLOBAL_SEED)
folds <- vfold_cv(train_data, v = CV_FOLDS)

# Base recipe
base_recipe <- recipe(Energy ~ ., data = train_data)

# Metrics and controls
metric <- metric_set(mae)
ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

# ============================================================================
# 2. TRAIN BASE MODELS
# ============================================================================

cat("\nTraining base models for stacking...\n")

# -- K-Nearest Neighbors --
cat("  Training KNN...\n")

knn_spec <- nearest_neighbor(
  mode = "regression",
  neighbors = tune("k")
) %>%
  set_engine("kknn")

knn_rec <- base_recipe %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

knn_wflow <- workflow() %>%
  add_model(knn_spec) %>%
  add_recipe(knn_rec)

set.seed(GLOBAL_SEED)
knn_res <- tune_grid(
  knn_wflow,
  resamples = folds,
  metrics = metric,
  grid = 4,
  control = ctrl_grid
)

# -- Linear Regression --
cat("  Training Linear Regression...\n")

lin_reg_spec <- linear_reg() %>%
  set_engine("lm")

lin_reg_rec <- base_recipe %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

lin_reg_wflow <- workflow() %>%
  add_model(lin_reg_spec) %>%
  add_recipe(lin_reg_rec)

set.seed(GLOBAL_SEED)
lin_reg_res <- fit_resamples(
  lin_reg_wflow,
  resamples = folds,
  metrics = metric,
  control = ctrl_res
)

# -- Support Vector Machine --
cat("  Training SVM...\n")

svm_spec <- svm_rbf(
  cost = tune("cost"),
  rbf_sigma = tune("sigma")
) %>%
  set_engine("kernlab") %>%
  set_mode("regression")

svm_rec <- base_recipe %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_corr(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

svm_wflow <- workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(svm_rec)

set.seed(GLOBAL_SEED)
svm_res <- tune_grid(
  svm_wflow,
  resamples = folds,
  grid = 4,
  metrics = metric,
  control = ctrl_grid
)

# ============================================================================
# 3. CREATE STACKED ENSEMBLE
# ============================================================================

cat("\nCreating stacked ensemble...\n")

# Build stack
model_stack <- stacks() %>%
  add_candidates(knn_res) %>%
  add_candidates(lin_reg_res) %>%
  add_candidates(svm_res)

# Blend predictions
model_stack <- model_stack %>%
  blend_predictions()

# Fit members
model_stack <- model_stack %>%
  fit_members()

# ============================================================================
# 4. EVALUATE STACKED MODEL
# ============================================================================

cat("\nEvaluating stacked model...\n")

preds_stack <- predict(model_stack, freez_base)
evaluate_model(freez_base$Energy, preds_stack$.pred, "Stacked Ensemble")

# Save predictions
stack_preds <- data.frame(pred = preds_stack$.pred, real = freez_base$Energy)
write.csv(stack_preds, "freez_base_stacks.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 5. CUSTOM BLENDING FROM SAVED PREDICTIONS
# ============================================================================

cat("\nLoading saved predictions for custom blending...\n")

# Load all available prediction files
pred_files <- list.files(pattern = "^freez_base_.*\\.csv$")
cat(sprintf("  Found %d prediction files\n", length(pred_files)))

if (length(pred_files) > 0) {
  # Load predictions
  all_preds <- list()
  for (f in pred_files) {
    pred_data <- read.csv(f)
    model_name <- gsub("freez_base_|\\.csv", "", f)
    if ("pred" %in% names(pred_data)) {
      all_preds[[model_name]] <- pred_data$pred
    }
  }
  
  if (length(all_preds) > 1) {
    # Create prediction matrix
    pred_matrix <- as.data.frame(all_preds)
    actual <- freez_base$Energy
    
    # Simple averaging
    cat("\n  Simple Average Blend:\n")
    blend_avg <- rowMeans(pred_matrix)
    evaluate_model(actual, blend_avg, "Average Blend")
    
    # Median blend
    cat("\n  Median Blend:\n")
    blend_med <- apply(pred_matrix, 1, median)
    evaluate_model(actual, blend_med, "Median Blend")
    
    # Weighted average using linear regression
    cat("\n  Linear Stacking:\n")
    blend_data <- pred_matrix
    blend_data$actual <- actual
    
    set.seed(GLOBAL_SEED)
    meta_model <- lm(actual ~ ., data = blend_data)
    blend_lm <- predict(meta_model, pred_matrix)
    evaluate_model(actual, blend_lm, "Linear Stack")
    
    # Save meta model
    saveRDS(meta_model, "model_stack_meta.rds")
    
    # Save blend predictions
    blend_results <- data.frame(
      actual = actual,
      avg = blend_avg,
      median = blend_med,
      lm_stack = blend_lm
    )
    write.csv(blend_results, "freez_base_blend_results.csv", quote = FALSE, row.names = FALSE)
  }
}

# ============================================================================
# 6. SAVE STACKED MODEL
# ============================================================================

cat("\nSaving stacked model...\n")
saveRDS(model_stack, "model_stacks.rds")

cat("Stacking complete.\n")
