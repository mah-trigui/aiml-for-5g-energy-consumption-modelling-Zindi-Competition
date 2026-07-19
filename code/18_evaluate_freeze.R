# ============================================================================
# 18_EVALUATE_FREEZE.R - Comprehensive Model Evaluation on Freeze Set
# ============================================================================
# Description: Loads all prediction files from the freeze set and performs
#              comprehensive model comparison and analysis.
# ============================================================================

source("00_config.R")
library(data.table)

# ============================================================================
# 1. LOAD FREEZE SET ACTUAL VALUES
# ============================================================================

cat("Loading freeze set data...\n")

# Load freeze info
freez_info <- read.csv2("freez_info.csv")
actual_values <- freez_info$Energy

cat(sprintf("  Freeze set size: %d\n", length(actual_values)))

# ============================================================================
# 2. LOAD ALL PREDICTION FILES
# ============================================================================

cat("\nLoading all prediction files...\n")

# Find all freeze prediction files
pred_files <- list.files(pattern = "^freez_.*\\.csv$")

# Load predictions
all_predictions <- list()
for (f in pred_files) {
  tryCatch(
    {
      pred_data <- read.csv(f)
      model_name <- gsub("freez_|\\.csv", "", f)

      # Handle different column names
      if ("pred" %in% names(pred_data)) {
        all_predictions[[model_name]] <- pred_data$pred
      } else if ("Energy" %in% names(pred_data)) {
        all_predictions[[model_name]] <- pred_data$Energy
      } else if (".pred" %in% names(pred_data)) {
        all_predictions[[model_name]] <- pred_data$.pred
      }

      cat(sprintf("  Loaded: %s\n", model_name))
    },
    error = function(e) {
      cat(sprintf("  Failed to load: %s (%s)\n", f, e$message))
    }
  )
}

cat(sprintf("\nTotal models loaded: %d\n", length(all_predictions)))

# ============================================================================
# 3. CALCULATE METRICS FOR ALL MODELS
# ============================================================================

cat("\n============================================\n")
cat("Model Performance Comparison\n")
cat("============================================\n")

ss_tot <- sum((actual_values - mean(actual_values))^2)

results <- do.call(rbind, lapply(names(all_predictions), function(model_name) {
  preds <- all_predictions[[model_name]]
  if (length(preds) != length(actual_values)) {
    cat(sprintf(
      "  Skipping %s: length mismatch (%d vs %d)\n",
      model_name, length(preds), length(actual_values)
    ))
    return(NULL)
  }
  ss_res <- sum((actual_values - preds)^2)
  data.frame(
    Model = model_name,
    MAE   = Metrics::mae(actual_values, preds),
    MAPE  = Metrics::mape(actual_values, preds),
    RMSE  = Metrics::rmse(actual_values, preds),
    R2    = 1 - ss_res / ss_tot
  )
}))

# Sort by MAE
results <- results[order(results$MAE), ]

# Print results
cat("\nResults sorted by MAE:\n")
print(results)

# ============================================================================
# 4. BEST MODEL ANALYSIS
# ============================================================================

cat("\n============================================\n")
cat("Best Model Analysis\n")
cat("============================================\n")

if (nrow(results) > 0) {
  best_model <- results$Model[1]
  best_preds <- all_predictions[[best_model]]

  cat(sprintf("Best Model: %s\n", best_model))
  cat(sprintf("  MAE:  %.6f\n", results$MAE[1]))
  cat(sprintf("  MAPE: %.6f\n", results$MAPE[1]))
  cat(sprintf("  RMSE: %.6f\n", results$RMSE[1]))
  cat(sprintf("  R2:   %.6f\n", results$R2[1]))

  # Error distribution
  errors <- actual_values - best_preds

  cat("\nError Distribution:\n")
  cat(sprintf("  Mean Error:   %.6f\n", mean(errors)))
  cat(sprintf("  Median Error: %.6f\n", median(errors)))
  cat(sprintf("  SD Error:     %.6f\n", sd(errors)))
  cat(sprintf("  Min Error:    %.6f\n", min(errors)))
  cat(sprintf("  Max Error:    %.6f\n", max(errors)))
}

# ============================================================================
# 5. ENSEMBLE COMPARISONS
# ============================================================================

cat("\n============================================\n")
cat("Ensemble Comparisons\n")
cat("============================================\n")

if (length(all_predictions) > 1) {
  pred_matrix <- as.data.frame(all_predictions)

  # Simple average
  avg_pred <- rowMeans(pred_matrix)
  mae_avg <- Metrics::mae(actual_values, avg_pred)
  mape_avg <- Metrics::mape(actual_values, avg_pred)

  cat(sprintf("Simple Average:\n"))
  cat(sprintf("  MAE:  %.6f\n", mae_avg))
  cat(sprintf("  MAPE: %.6f\n", mape_avg))

  # Median
  med_pred <- apply(pred_matrix, 1, median)
  mae_med <- Metrics::mae(actual_values, med_pred)
  mape_med <- Metrics::mape(actual_values, med_pred)

  cat(sprintf("\nMedian Ensemble:\n"))
  cat(sprintf("  MAE:  %.6f\n", mae_med))
  cat(sprintf("  MAPE: %.6f\n", mape_med))

  # Top-3 average
  if (nrow(results) >= 3) {
    top3_models <- results$Model[1:3]
    top3_preds <- pred_matrix[, top3_models]
    top3_avg <- rowMeans(top3_preds)
    mae_top3 <- Metrics::mae(actual_values, top3_avg)
    mape_top3 <- Metrics::mape(actual_values, top3_avg)

    cat(sprintf("\nTop-3 Average (%s):\n", paste(top3_models, collapse = ", ")))
    cat(sprintf("  MAE:  %.6f\n", mae_top3))
    cat(sprintf("  MAPE: %.6f\n", mape_top3))
  }

  # Weighted average (inverse MAE weights)
  weights <- 1 / results$MAE
  weights <- weights / sum(weights)
  names(weights) <- results$Model

  common <- intersect(names(weights), names(pred_matrix))
  w <- weights[common] / sum(weights[common])
  weighted_pred <- as.vector(as.matrix(pred_matrix[, common]) %*% w)

  mae_weighted <- Metrics::mae(actual_values, weighted_pred)
  mape_weighted <- Metrics::mape(actual_values, weighted_pred)

  cat(sprintf("\nWeighted Average (inverse MAE):\n"))
  cat(sprintf("  MAE:  %.6f\n", mae_weighted))
  cat(sprintf("  MAPE: %.6f\n", mape_weighted))
}

# ============================================================================
# 6. SAVE EVALUATION RESULTS
# ============================================================================

cat("\n============================================\n")
cat("Saving Results\n")
cat("============================================\n")

# Save full results
write.csv(results, "model_comparison_results.csv", quote = FALSE, row.names = FALSE)
cat("  Saved: model_comparison_results.csv\n")

# Save as RDS for easy loading
saveRDS(list(
  results = results,
  predictions = all_predictions,
  actual = actual_values
), "freeze_evaluation.rds")
cat("  Saved: freeze_evaluation.rds\n")

cat("\nEvaluation complete.\n")
