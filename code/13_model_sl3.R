# ============================================================================
# 13_MODEL_SL3.R - Super Learner Model Training
# ============================================================================
# Description: Trains models using the sl3 package (Super Learner 3)
#              for advanced ensemble learning.
# ============================================================================

source("00_config.R")
library(sl3)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. CREATE SL3 TASK
# ============================================================================

cat("Creating SL3 task...\n")

# Get feature names
feature_names <- names(tr_base)[names(tr_base) != "Energy"]

# Create task
task <- make_sl3_Task(
  data = tr_base,
  outcome = "Energy",
  covariates = feature_names
)

cat(sprintf("  Features: %d\n", length(feature_names)))
cat(sprintf("  Observations: %d\n", nrow(tr_base)))

# ============================================================================
# 2. DEFINE LEARNERS
# ============================================================================

cat("\nDefining learners...\n")

# Basic learners
lrn_glm <- Lrnr_glm$new()
lrn_mean <- Lrnr_mean$new()

# Penalized regressions
lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
lrn_lasso <- Lrnr_glmnet$new(alpha = 1)

# Tree-based methods
lrn_ranger <- Lrnr_ranger$new()
lrn_xgb <- Lrnr_xgboost$new()
lrn_gam <- Lrnr_gam$new()
lrn_bayesglm <- Lrnr_bayesglm$new()

# Create stack of learners
stack <- Stack$new(
  lrn_glm,
  lrn_mean,
  lrn_ridge,
  lrn_lasso,
  lrn_ranger,
  lrn_xgb,
  lrn_gam,
  lrn_bayesglm
)

cat("  Learners in stack:\n")
cat("    - GLM\n")
cat("    - Mean\n")
cat("    - Ridge\n")
cat("    - Lasso\n")
cat("    - Ranger (Random Forest)\n")
cat("    - XGBoost\n")
cat("    - GAM\n")
cat("    - Bayesian GLM\n")

# ============================================================================
# 3. CREATE SUPER LEARNER
# ============================================================================

cat("\nCreating Super Learner...\n")

# Use non-negative least squares as metalearner
sl <- Lrnr_sl$new(
  learners = stack,
  metalearner = Lrnr_nnls$new()
)

# ============================================================================
# 4. TRAIN SUPER LEARNER
# ============================================================================

cat("\nTraining Super Learner...\n")

start_time <- proc.time()

set.seed(GLOBAL_SEED)
sl_fit <- sl$train(task = task)

runtime <- proc.time() - start_time
cat(sprintf("  Training time: %.2f seconds\n", runtime["elapsed"]))

# ============================================================================
# 5. GET CROSS-VALIDATED PREDICTIONS
# ============================================================================

cat("\nGetting cross-validated predictions...\n")

# Full fit predictions
full_fit_preds <- sl_fit$fit_object$cv_fit$predict_fold(
  task = task,
  fold_number = "full"
)

# ============================================================================
# 6. EVALUATE ON FREEZE SET
# ============================================================================

cat("\nEvaluating on freeze set...\n")

# Create prediction task
prediction_task <- make_sl3_Task(
  data = freez_base,
  covariates = feature_names
)

# Get predictions
preds <- sl_fit$predict(task = prediction_task)
preds <- as.numeric(preds)

# Evaluate
evaluate_model(freez_base$Energy, preds, "Super Learner (SL3)")

# Save predictions
sl3_preds <- data.frame(pred = preds, real = freez_base$Energy)
write.csv(sl3_preds, "freez_base_sl3.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 7. LEARNER COEFFICIENTS
# ============================================================================

cat("\nSuper Learner Coefficients:\n")

# Get metalearner coefficients
coefficients <- sl_fit$coefficients
if (!is.null(coefficients)) {
  coef_df <- data.frame(
    Learner = names(coefficients),
    Coefficient = as.numeric(coefficients)
  )
  coef_df <- coef_df[order(-coef_df$Coefficient), ]
  print(coef_df)
}

# ============================================================================
# 8. SAVE MODEL
# ============================================================================

cat("\nSaving Super Learner model...\n")

saveRDS(sl_fit, "model_sl3.rds")

cat("Super Learner training complete.\n")
