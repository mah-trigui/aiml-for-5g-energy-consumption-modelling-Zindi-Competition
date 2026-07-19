# ============================================================================
# 10_MODEL_CARET.R - Caret-Based Model Training
# ============================================================================
# Description: Trains multiple models using the caret framework including
#              Random Forest, GLMNet, and GAM.
# ============================================================================

source("00_config.R")
library(caret)
library(ranger)
library(glmnet)
library(mgcv)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. SETUP
# ============================================================================

cat("Setting up caret training...\n")

# Set up parallel processing
cl <- setup_parallel(2)

# Define train control
fit_control <- trainControl(
  method = "cv",
  number = CV_FOLDS,
  allowParallel = TRUE,
  search = "random",
  summaryFunction = mapeSummary
)

# Prepare data
X_train <- tr_base[, names(tr_base) != "Energy"]
y_train <- tr_base$Energy
X_freez <- freez_base[, names(freez_base) != "Energy"]
y_freez <- freez_base$Energy

# ============================================================================
# 2. RANDOM FOREST
# ============================================================================

cat("\nTraining Random Forest...\n")

set.seed(GLOBAL_SEED)
model_rf <- caret::train(
  x = X_train,
  y = y_train,
  method = "ranger",
  trControl = fit_control,
  tuneLength = 5,
  maximize = FALSE,
  metric = "MAPE"
)

# Evaluate
preds_rf <- predict(model_rf, X_freez)
evaluate_model(y_freez, preds_rf, "Random Forest (Caret)")

# Save predictions
rf_preds <- data.frame(pred = preds_rf, real = y_freez)
write.csv(rf_preds, "freez_base_rf_caret.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 3. GLMNET (ELASTIC NET)
# ============================================================================

cat("\nTraining GLMNet...\n")

set.seed(GLOBAL_SEED)
model_glmnet <- caret::train(
  Energy ~ .,
  data = tr_base,
  method = "glmnet",
  trControl = fit_control,
  tuneLength = 5,
  metric = "MAPE"
)

# Evaluate
preds_glmnet <- predict(model_glmnet, X_freez)
evaluate_model(y_freez, preds_glmnet, "GLMNet (Caret)")

# Save predictions
glmnet_preds <- data.frame(pred = preds_glmnet, real = y_freez)
write.csv(glmnet_preds, "freez_base_glmnet_caret.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 4. GAM (GENERALIZED ADDITIVE MODEL)
# ============================================================================

cat("\nTraining GAM...\n")

set.seed(GLOBAL_SEED)
model_gam <- caret::train(
  Energy ~ .,
  data = tr_base,
  method = "gam",
  trControl = fit_control,
  tuneLength = 5,
  metric = "MAPE"
)

# Evaluate
preds_gam <- predict(model_gam, X_freez)
evaluate_model(y_freez, preds_gam, "GAM (Caret)")

# Save predictions
gam_preds <- data.frame(pred = preds_gam, real = y_freez)
write.csv(gam_preds, "freez_base_gam_caret.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 5. XGBOOST WITH CARET
# ============================================================================

cat("\nTraining XGBoost with caret...\n")

set.seed(GLOBAL_SEED)
model_xgb_caret <- caret::train(
  x = X_train,
  y = y_train,
  method = "xgbTree",
  trControl = fit_control,
  tuneLength = 5,
  nthread = 4,
  maximize = FALSE,
  metric = "MAPE"
)

# Evaluate
preds_xgb <- predict(model_xgb_caret, X_freez)
evaluate_model(y_freez, preds_xgb, "XGBoost (Caret)")

# Save predictions
xgb_caret_preds <- data.frame(pred = preds_xgb, real = y_freez)
write.csv(xgb_caret_preds, "freez_base_xgboost_caret.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 6. STOP PARALLEL PROCESSING
# ============================================================================

stop_parallel(cl)

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================

cat("\n============================================\n")
cat("Model Comparison Summary\n")
cat("============================================\n")

models <- list(
  "Random Forest" = preds_rf,
  "GLMNet" = preds_glmnet,
  "GAM" = preds_gam,
  "XGBoost" = preds_xgb
)

comparison <- data.frame(
  Model = names(models),
  MAE = sapply(models, function(p) Metrics::mae(y_freez, p)),
  MAPE = sapply(models, function(p) Metrics::mape(y_freez, p))
)

comparison <- comparison[order(comparison$MAE), ]
print(comparison)

cat("============================================\n")

# ============================================================================
# 8. SAVE MODELS
# ============================================================================

cat("\nSaving models...\n")

saveRDS(model_rf, "model_rf_caret.rds")
saveRDS(model_glmnet, "model_glmnet_caret.rds")
saveRDS(model_gam, "model_gam_caret.rds")
saveRDS(model_xgb_caret, "model_xgb_caret.rds")

cat("Caret training complete.\n")
