# ============================================================================
# 11_MODEL_H2O.R - H2O AutoML Model Training
# ============================================================================
# Description: Uses H2O's AutoML for automatic model selection and
#              stacked ensemble creation.
# ============================================================================

source("00_config.R")
library(h2o)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. INITIALIZE H2O
# ============================================================================

cat("Initializing H2O...\n")
h2o.init()

# ============================================================================
# 2. PREPARE DATA
# ============================================================================

cat("Preparing data for H2O...\n")

# Convert to H2O frames
train_h2o <- as.h2o(tr_base)
test_h2o <- as.h2o(ts_base)
freez_h2o <- as.h2o(freez_base)

# Define predictors and response
y <- "Energy"
x <- setdiff(names(train_h2o), y)

cat(sprintf("  Predictors: %d\n", length(x)))
cat(sprintf("  Response: %s\n", y))

# ============================================================================
# 3. TRAIN INDIVIDUAL BASE MODELS
# ============================================================================

cat("\nTraining base models...\n")

nfolds <- CV_FOLDS

# -- Gradient Boosting Machine --
cat("  Training GBM...\n")
model_gbm <- h2o.gbm(
  x = x,
  y = y,
  training_frame = train_h2o,
  nfolds = nfolds,
  keep_cross_validation_predictions = TRUE,
  seed = GLOBAL_SEED
)

# -- Random Forest --
cat("  Training Random Forest...\n")
model_rf <- h2o.randomForest(
  x = x,
  y = y,
  training_frame = train_h2o,
  nfolds = nfolds,
  keep_cross_validation_predictions = TRUE,
  seed = GLOBAL_SEED
)

# -- Generalized Linear Model --
cat("  Training GLM...\n")
model_glm <- h2o.glm(
  x = x,
  y = y,
  training_frame = train_h2o,
  nfolds = nfolds,
  keep_cross_validation_predictions = TRUE,
  seed = GLOBAL_SEED
)

# ============================================================================
# 4. CREATE STACKED ENSEMBLE
# ============================================================================

cat("\nCreating stacked ensemble...\n")

ensemble <- h2o.stackedEnsemble(
  x = x,
  y = y,
  metalearner_algorithm = "glm",
  training_frame = train_h2o,
  base_models = list(model_gbm, model_rf, model_glm)
)

# ============================================================================
# 5. EVALUATE MODELS
# ============================================================================

cat("\nEvaluating models on freeze set...\n")

# Evaluate individual models
perf_gbm <- h2o.performance(model_gbm, newdata = freez_h2o)
perf_rf <- h2o.performance(model_rf, newdata = freez_h2o)
perf_glm <- h2o.performance(model_glm, newdata = freez_h2o)
perf_ensemble <- h2o.performance(ensemble, newdata = freez_h2o)

cat("\nBase Model MAE:\n")
cat(sprintf("  GBM: %.6f\n", h2o.mae(perf_gbm)))
cat(sprintf("  RF:  %.6f\n", h2o.mae(perf_rf)))
cat(sprintf("  GLM: %.6f\n", h2o.mae(perf_glm)))
cat(sprintf("  Ensemble: %.6f\n", h2o.mae(perf_ensemble)))

# Get ensemble predictions
preds_ensemble <- as.data.frame(h2o.predict(ensemble, freez_h2o))
names(preds_ensemble)[1] <- "pred"
preds_ensemble$real <- freez_base$Energy

evaluate_model(preds_ensemble$real, preds_ensemble$pred, "H2O Stacked Ensemble")

# Save predictions
write.csv(preds_ensemble, "freez_base_h2o_ensemble.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 6. H2O AUTOML
# ============================================================================

cat("\nRunning H2O AutoML...\n")

set.seed(GLOBAL_SEED)
aml <- h2o.automl(
  x = x,
  y = y,
  training_frame = train_h2o,
  max_models = 20,
  seed = GLOBAL_SEED,
  sort_metric = "MAE"
)

# Print leaderboard
cat("\nAutoML Leaderboard:\n")
print(aml@leaderboard)

# Get best model predictions
preds_automl <- as.data.frame(h2o.predict(aml@leader, freez_h2o))
names(preds_automl)[1] <- "pred"
preds_automl$real <- freez_base$Energy

evaluate_model(preds_automl$real, preds_automl$pred, "H2O AutoML Best")

# Save predictions
write.csv(preds_automl, "freez_base_h2o_automl.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 7. SAVE MODELS
# ============================================================================

cat("\nSaving H2O models...\n")

h2o.saveModel(ensemble, path = "./h2o_models", force = TRUE)
h2o.saveModel(aml@leader, path = "./h2o_models", force = TRUE)

cat("H2O training complete.\n")

# Note: Don't shutdown H2O if you plan to use it for predictions later
# h2o.shutdown(prompt = FALSE)
