# ============================================================================
# 09_MODEL_CATBOOST.R - CatBoost Model Training
# ============================================================================
# Description: Trains CatBoost models with cross-validation fold averaging
#              for improved predictions.
# ============================================================================

source("00_config.R")
library(catboost)
library(caret)

# Load splits
load("splits_base.RData")

# ============================================================================
# 1. PREPARE DATA
# ============================================================================

cat("Preparing data for CatBoost...\n")

# Prepare training data
y_train <- unlist(tr_base$Energy)
X_train <- tr_base %>% select(-Energy)

# Prepare validation data
y_valid <- unlist(ts_base$Energy)
X_valid <- ts_base %>% select(-Energy)

# Prepare freeze data
y_freez <- unlist(freez_base$Energy)
X_freez <- freez_base %>% select(-Energy)

# Create CatBoost pools
train_pool <- catboost.load_pool(data = X_train, label = y_train)
valid_pool <- catboost.load_pool(data = X_valid, label = y_valid)
freez_pool <- catboost.load_pool(data = X_freez, label = y_freez)

# ============================================================================
# 2. BASIC CATBOOST MODEL
# ============================================================================

cat("\nTraining basic CatBoost model...\n")

# Parameters
params_basic <- list(
  iterations = 3000,
  learning_rate = 0.1,
  depth = 8,
  loss_function = 'MAPE',
  eval_metric = 'MAPE',
  metric_period = 200,
  logging_level = 'Silent'
)

# Train model
set.seed(GLOBAL_SEED)
model_cat_basic <- catboost.train(
  learn_pool = train_pool,
  test_pool = valid_pool,
  params = params_basic
)

# Evaluate
preds <- catboost.predict(model_cat_basic, pool = freez_pool)
evaluate_model(y_freez, preds, "CatBoost Basic")

# Save predictions
cat_basic_preds <- data.frame(pred = preds, real = y_freez)
write.csv(cat_basic_preds, "freez_base_catboost_basic.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 3. CATBOOST WITH K-FOLD CROSS-VALIDATION AVERAGING
# ============================================================================

cat("\nTraining CatBoost with K-Fold CV averaging...\n")

# Create folds
set.seed(GLOBAL_SEED)
folds <- createFolds(as.factor(y_train), k = CV_FOLDS)

# Initialize prediction accumulator
pred_accumulator <- rep(0, nrow(X_freez))

# Train on each fold
for (fold_idx in seq_along(folds)) {
  cat(sprintf("  Training fold %d/%d...\n", fold_idx, length(folds)))
  
  # Split indices
  valid_idx <- folds[[fold_idx]]
  train_idx <- setdiff(seq_len(length(y_train)), valid_idx)
  
  # Create pools for this fold
  dtrain <- catboost.load_pool(
    data = X_train[train_idx, ],
    label = y_train[train_idx]
  )
  dvalid <- catboost.load_pool(
    data = X_train[valid_idx, ],
    label = y_train[valid_idx]
  )
  
  # Parameters
  params_cv <- list(
    iterations = 3000,
    learning_rate = 0.1,
    depth = 8,
    loss_function = 'MAPE',
    eval_metric = 'MAPE',
    metric_period = 200,
    logging_level = 'Silent'
  )
  
  # Train model
  set.seed(GLOBAL_SEED)
  model_fold <- catboost.train(
    learn_pool = dtrain,
    test_pool = dvalid,
    params = params_cv
  )
  
  # Accumulate predictions
  pred_fold <- catboost.predict(model_fold, pool = freez_pool)
  pred_accumulator <- pred_accumulator + pred_fold
}

# Average predictions
preds_cv <- pred_accumulator / length(folds)

# Evaluate
evaluate_model(y_freez, preds_cv, "CatBoost K-Fold CV")

# Save predictions
cat_cv_preds <- data.frame(pred = preds_cv, real = y_freez)
write.csv(cat_cv_preds, "freez_base_catboost_cv.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 4. CATBOOST WITH CARET
# ============================================================================

cat("\nTraining CatBoost with caret...\n")

# Set up parallel processing
cl <- setup_parallel(2)

# Train control
fit_control <- trainControl(
  method = "cv",
  number = CV_FOLDS,
  allowParallel = TRUE,
  search = "random",
  summaryFunction = mapeSummary
)

# Train model
set.seed(GLOBAL_SEED)
model_cat_caret <- caret::train(
  x = X_train,
  y = y_train,
  method = catboost.caret,
  trControl = fit_control,
  logging_level = 'Silent',
  tuneLength = 5,
  maximize = FALSE,
  metric = "MAPE"
)

# Stop parallel processing
stop_parallel(cl)

# Evaluate
preds_caret <- predict(model_cat_caret, X_freez)
evaluate_model(y_freez, preds_caret, "CatBoost Caret")

# Save predictions
cat_caret_preds <- data.frame(pred = preds_caret, real = y_freez)
write.csv(cat_caret_preds, "freez_base_catboost_caret.csv", quote = FALSE, row.names = FALSE)

# ============================================================================
# 5. SAVE MODELS
# ============================================================================

cat("\nSaving models...\n")

catboost.save_model(model_cat_basic, "model_catboost_basic.cbm")
saveRDS(model_cat_caret, "model_catboost_caret.rds")

cat("CatBoost training complete.\n")
