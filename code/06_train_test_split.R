# ============================================================================
# 06_TRAIN_TEST_SPLIT.R - Create Training and Test Sets
# ============================================================================
# Description: Creates train/test/freeze splits for model training and
#              evaluation. Freeze set is used for validation before submission.
# ============================================================================

source("00_config.R")

# ============================================================================
# 1. LOAD DATASETS
# ============================================================================

cat("Loading datasets...\n")

# Load LightGBM dataset for base-level modeling
if (file.exists("df_ligh.RData")) {
  load("df_ligh.RData")
  cat("  LightGBM dataset loaded\n")
}

# Load Cell dataset for cell-level modeling
if (file.exists("df_cell.RData")) {
  load("df_cell.RData")
  cat("  Cell dataset loaded\n")
}

# ============================================================================
# 2. CREATE BASE-LEVEL SPLITS
# ============================================================================

cat("\nCreating base-level train/test/freeze splits...\n")

if (exists("df_ligh")) {
  setDT(df_ligh)
  
  # Separate submission data (no Energy)
  sub_base <- df_ligh[is.na(df_ligh$Energy), ]
  df <- df_ligh[!is.na(df_ligh$Energy), ]
  
  # Create freeze set (holdout for final validation)
  set.seed(GLOBAL_SEED)
  freez_base <- df[sample(nrow(df), FREEZE_SAMPLE_SIZE), ]
  
  # Save freeze set info for later matching
  freez_info <- freez_base[, c("BS", "Time", "Energy")]
  write.csv2(freez_info, "freez_info.csv", quote = FALSE, row.names = FALSE)
  
  # Remove freeze samples from training data
  df <- df[!paste(df$BS, df$Time) %in% paste(freez_base$BS, freez_base$Time), ]
  
  # Remove BS and Time for modeling
  df <- df[, !(names(df) %in% c("BS", "Time")), with = FALSE]
  freez_base <- freez_base[, !(names(freez_base) %in% c("BS", "Time")), with = FALSE]
  
  # Create train/test split
  set.seed(GLOBAL_SEED)
  tr_base <- df[sample(nrow(df), TRAIN_SAMPLE_SIZE), ]
  ts_base <- df[!1:nrow(df) %in% sample(nrow(df), TRAIN_SAMPLE_SIZE), ]
  
  cat(sprintf("  Base training set: %d rows\n", nrow(tr_base)))
  cat(sprintf("  Base test set: %d rows\n", nrow(ts_base)))
  cat(sprintf("  Base freeze set: %d rows\n", nrow(freez_base)))
  cat(sprintf("  Base submission set: %d rows\n", nrow(sub_base)))
  
  # Save base splits
  save(tr_base, ts_base, freez_base, sub_base, file = "splits_base.RData")
}

# ============================================================================
# 3. CREATE CELL-LEVEL SPLITS
# ============================================================================

cat("\nCreating cell-level train/test/freeze splits...\n")

if (exists("df_cell")) {
  setDT(df_cell)
  
  # Load freeze info to match same samples
  freez_info <- read.csv2("freez_info.csv")
  
  # Separate submission data (no Energy)
  sub_cell <- df_cell[is.na(df_cell$Energy), ]
  df <- df_cell[!is.na(df_cell$Energy), ]
  
  # Create matching freeze set (same BS/Time as base freeze)
  freez_cell <- df[paste(df$BS, df$Time) %in% paste(freez_info$BS, freez_info$Time), ]
  
  # Remove freeze samples from training data
  df <- df[!paste(df$BS, df$Time) %in% paste(freez_info$BS, freez_info$Time), ]
  
  # Remove BS and Time for modeling (keep unique identifier)
  df_unique <- unique(df[, !(names(df) %in% c("BS", "Time")), with = FALSE])
  freez_cell_unique <- unique(freez_cell[, !(names(freez_cell) %in% c("BS", "Time")), with = FALSE])
  
  # Create train/test split
  set.seed(GLOBAL_SEED)
  tr_cell <- df_unique[sample(nrow(df_unique), min(77000, nrow(df_unique))), ]
  ts_cell <- df_unique[!1:nrow(df_unique) %in% sample(nrow(df_unique), min(77000, nrow(df_unique))), ]
  
  cat(sprintf("  Cell training set: %d rows\n", nrow(tr_cell)))
  cat(sprintf("  Cell test set: %d rows\n", nrow(ts_cell)))
  cat(sprintf("  Cell freeze set: %d rows\n", nrow(freez_cell)))
  cat(sprintf("  Cell submission set: %d rows\n", nrow(sub_cell)))
  
  # Save cell splits
  save(tr_cell, ts_cell, freez_cell, freez_cell_unique, sub_cell, file = "splits_cell.RData")
}

# ============================================================================
# 4. SUMMARY
# ============================================================================

cat("\n============================================\n")
cat("Train/Test/Freeze Split Summary\n")
cat("============================================\n")
cat(sprintf("Random Seed: %d\n", GLOBAL_SEED))
cat(sprintf("Train Sample Size: %d\n", TRAIN_SAMPLE_SIZE))
cat(sprintf("Freeze Sample Size: %d\n", FREEZE_SAMPLE_SIZE))
cat("============================================\n")
cat("Splits saved to:\n")
cat("  - splits_base.RData (tr_base, ts_base, freez_base, sub_base)\n")
cat("  - splits_cell.RData (tr_cell, ts_cell, freez_cell, sub_cell)\n")
cat("============================================\n\n")
