# ============================================================================
# 16_GENERATE_SUBMISSION.R - Generate Final Submission Files
# ============================================================================
# Description: Loads all trained models and generates final predictions
#              for submission. Implements various ensemble strategies.
# ============================================================================

source("00_config.R")
library(data.table)

# Load submission data
load("splits_base.RData")
load("splits_cell.RData")

# Load original submission format
submission <- read.csv(FILE_SUBMISSION)

# ============================================================================
# 1. LOAD ALL MODEL PREDICTIONS (BASE LEVEL)
# ============================================================================

cat("Loading base-level model predictions...\n")

# List of base-level prediction files
base_pred_files <- c(
  "base_lightgbm_tuned.csv",
  "base_xgboost_tuned.csv",
  "base_catboost_cv.csv",
  "base_catboost_caret.csv",
  "base_rf_caret.csv",
  "base_xgboost_caret.csv",
  "base_glmnet_caret.csv",
  "base_gam_caret.csv",
  "base_h2o_automl.csv",
  "base_sl3.csv",
  "base_stacks.csv"
)

# Load available predictions
base_preds <- list()
for (f in base_pred_files) {
  if (file.exists(f)) {
    pred_data <- read.csv(f)
    model_name <- gsub("base_|\\.csv", "", f)
    if ("Energy" %in% names(pred_data)) {
      base_preds[[model_name]] <- pred_data$Energy
    } else if ("pred" %in% names(pred_data)) {
      base_preds[[model_name]] <- pred_data$pred
    }
    cat(sprintf("  Loaded: %s\n", f))
  }
}

cat(sprintf("  Total base models: %d\n", length(base_preds)))

# ============================================================================
# 2. LOAD ALL MODEL PREDICTIONS (CELL LEVEL)
# ============================================================================

cat("\nLoading cell-level model predictions...\n")

cell_pred_files <- c(
  "cell_lightgbm.csv",
  "cell_lightgbm_tidy.csv",
  "cell_xgboost_tidy.csv"
)

# Load available predictions
cell_preds <- list()
for (f in cell_pred_files) {
  if (file.exists(f)) {
    pred_data <- read.csv(f)
    model_name <- gsub("cell_|\\.csv", "", f)
    if ("Energy" %in% names(pred_data)) {
      cell_preds[[model_name]] <- pred_data$Energy
    } else if ("pred" %in% names(pred_data)) {
      cell_preds[[model_name]] <- pred_data$pred
    }
    cat(sprintf("  Loaded: %s\n", f))
  }
}

cat(sprintf("  Total cell models: %d\n", length(cell_preds)))

# ============================================================================
# 3. CREATE ENSEMBLE PREDICTIONS
# ============================================================================

cat("\nCreating ensemble predictions...\n")

# -- Base-Level Ensembles --
if (length(base_preds) > 0) {
  base_pred_matrix <- as.data.frame(base_preds)
  
  # Simple average
  base_avg <- rowMeans(base_pred_matrix)
  
  # Median
  base_med <- apply(base_pred_matrix, 1, median)
  
  # Load meta model if exists
  if (file.exists("model_stack_meta.rds")) {
    meta_model <- readRDS("model_stack_meta.rds")
    base_stack <- predict(meta_model, base_pred_matrix)
  } else {
    base_stack <- base_avg
  }
}

# -- Cell-Level Ensembles --
if (length(cell_preds) > 0) {
  cell_pred_matrix <- as.data.frame(cell_preds)
  
  # Simple average
  cell_avg <- rowMeans(cell_pred_matrix)
  
  # Median
  cell_med <- apply(cell_pred_matrix, 1, median)
}

# -- Mixed Ensembles --
if (length(base_preds) > 0 && length(cell_preds) > 0) {
  # Combine base and cell predictions
  all_preds <- cbind(base_pred_matrix, cell_pred_matrix)
  
  # Mixed average
  mix_avg <- rowMeans(all_preds)
  
  # Mixed median
  mix_med <- apply(all_preds, 1, median)
}

# ============================================================================
# 4. FORMAT SUBMISSION FILES
# ============================================================================

cat("\nFormatting submission files...\n")

# Function to create submission dataframe
create_submission <- function(predictions, sub_data) {
  result <- data.frame(
    Time = paste0(sub_data$Time, "_", sub_data$BS),
    Energy = predictions
  )
  return(result)
}

# ============================================================================
# 5. GENERATE AND SAVE SUBMISSIONS
# ============================================================================

cat("\nGenerating submission files...\n")

# Base-level submissions
if (length(base_preds) > 0) {
  # Average
  sub_base_avg <- create_submission(base_avg, sub_base)
  write.csv(sub_base_avg, "submission_base_avg.csv", quote = FALSE, row.names = FALSE)
  cat("  Created: submission_base_avg.csv\n")
  
  # Median
  sub_base_med <- create_submission(base_med, sub_base)
  write.csv(sub_base_med, "submission_base_med.csv", quote = FALSE, row.names = FALSE)
  cat("  Created: submission_base_med.csv\n")
  
  # Stack
  sub_base_stack <- create_submission(base_stack, sub_base)
  write.csv(sub_base_stack, "submission_base_stack.csv", quote = FALSE, row.names = FALSE)
  cat("  Created: submission_base_stack.csv\n")
}

# Cell-level submissions (need to aggregate to base level)
if (length(cell_preds) > 0) {
  # Note: Cell predictions need to be aggregated by BS/Time
  # This is a simplified version - actual implementation may need adjustment
  cat("  Cell-level submissions require aggregation (see Submit.R for details)\n")
}

# Mixed submissions
if (length(base_preds) > 0 && length(cell_preds) > 0) {
  sub_mix_avg <- create_submission(mix_avg, sub_base)
  write.csv(sub_mix_avg, "submission_mix_avg.csv", quote = FALSE, row.names = FALSE)
  cat("  Created: submission_mix_avg.csv\n")
  
  sub_mix_med <- create_submission(mix_med, sub_base)
  write.csv(sub_mix_med, "submission_mix_med.csv", quote = FALSE, row.names = FALSE)
  cat("  Created: submission_mix_med.csv\n")
}

# ============================================================================
# 6. INDIVIDUAL MODEL SUBMISSIONS
# ============================================================================

cat("\nGenerating individual model submissions...\n")

for (model_name in names(base_preds)) {
  sub_individual <- create_submission(base_preds[[model_name]], sub_base)
  filename <- sprintf("submission_%s.csv", model_name)
  write.csv(sub_individual, filename, quote = FALSE, row.names = FALSE)
  cat(sprintf("  Created: %s\n", filename))
}

# ============================================================================
# 7. SUMMARY
# ============================================================================

cat("\n============================================\n")
cat("Submission Generation Summary\n")
cat("============================================\n")
cat(sprintf("Base-level models: %d\n", length(base_preds)))
cat(sprintf("Cell-level models: %d\n", length(cell_preds)))
cat("\nSubmission files created in working directory.\n")
cat("============================================\n\n")
