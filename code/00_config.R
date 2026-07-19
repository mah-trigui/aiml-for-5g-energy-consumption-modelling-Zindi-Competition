# ============================================================================
# 00_CONFIG.R - Configuration and Library Management
# ============================================================================
# Description: Central configuration file for the Energy Consumption Prediction
#              project. Contains all library imports and global settings.
# Updated: Comprehensive version with all dependencies from original scripts
# ============================================================================

# ============================================================================
# 1. REQUIRED LIBRARIES
# ============================================================================

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  Loading Required Packages for Energy Prediction Pipeline     \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

suppressPackageStartupMessages({
    # -- Data Manipulation --
    library(data.table)
    library(dplyr)
    library(tidyr)
    library(purrr)
    library(tibble)

    # -- Visualization & EDA --
    library(ggplot2)
    library(gwalkr) # Alternative: GWalkR
    library(explore)
    library(dataMaid) # For data reports
    library(SmartEDA) # For EDA reports
    library(dlookr) # For diagnostics

    # -- Factor Manipulation --
    library(forcats)

    # -- Statistical Analysis --
    library(MASS) # For boxcox, statistical tests
    library(ltm) # For biserial correlation
    library(broom) # Tidy model outputs
    library(effectsize) # Effect size calculations

    # -- Machine Learning - Core Frameworks --
    library(caret)
    library(tidymodels)
    library(recipes)
    library(stacks)
    library(workflows)
    library(tune)
    library(yardstick)
    library(rsample)
    library(parsnip)

    # -- Machine Learning - Tree-Based Algorithms --
    library(xgboost)
    library(lightgbm)
    library(catboost)
    library(ranger)
    library(rpart)
    library(rpart.plot)

    # -- Machine Learning - Advanced Frameworks --
    library(h2o)
    library(sl3) # Super Learner
    library(treesnip) # Additional tidymodels engines

    # -- Machine Learning - Linear/Statistical Models --
    library(glmnet) # Elastic net, lasso, ridge
    library(mgcv) # GAM models
    library(kernlab) # SVM
    library(kknn) # KNN

    # -- Clustering --
    library(cluster)
    library(FNN)

    # -- Feature Engineering & Encoding --
    library(vtreat)

    # -- Parallel Processing --
    library(parallel)
    library(doParallel)

    # -- Evaluation Metrics --
    library(Metrics)
})

cat("✓ All packages loaded successfully.\n\n")

# ============================================================================
# 2. GLOBAL SETTINGS
# ============================================================================

# -- Random Seed for Reproducibility --
GLOBAL_SEED <- 1618
set.seed(GLOBAL_SEED)

# -- File Paths --
DATA_PATH <- getwd() # Adjust as needed

# Input files (update these paths as needed)
FILE_SUBMISSION <- "imgs_202307101549519358.csv"
FILE_ENERGY <- "imgs_2023071012133740345.csv"
FILE_CELL <- "imgs_2023071012130978799.csv"
FILE_BASE <- "imgs_2023071012123392536.csv"

# -- Model Parameters --
CV_FOLDS <- 5
TRAIN_SAMPLE_SIZE <- 75000
FREEZE_SAMPLE_SIZE <- 10000

# -- Energy Split Proportions (for 2-cell bases) --
# Best performing proportions from analysis
AVG_189 <- 0.4328334
AVG_155 <- 0.5671666
AVG_426 <- 0.3533747
AVG_365 <- 0.6466253

# -- Global Options --
options(
    scipen = 999, # Disable scientific notation
    digits = 6,
    tidymodels.dark = TRUE, # Better console output for tidymodels
    warn = -1 # Suppress warnings (set to 0 to enable)
)

# ============================================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================================

if (!dir.exists("models")) dir.create("models")
if (!dir.exists("predictions")) dir.create("predictions")
if (!dir.exists("submissions")) dir.create("submissions")
if (!dir.exists("reports")) dir.create("reports")
if (!dir.exists("data")) dir.create("data")

# ============================================================================
# 4. HELPER FUNCTIONS
# ============================================================================

#' Set up parallel processing cluster
#' @param n_cores Number of cores to use (default: 2)
#' @return Cluster object
setup_parallel <- function(n_cores = 2) {
    cl <- makePSOCKcluster(n_cores)
    doParallel::registerDoParallel(cl)
    cat(sprintf("✓ Parallel cluster started with %d cores.\n", n_cores))
    return(cl)
}

#' Stop parallel processing cluster
#' @param cl Cluster object
stop_parallel <- function(cl) {
    if (!is.null(cl)) {
        stopCluster(cl)
        cat("✓ Parallel cluster stopped.\n")
    }
}

#' Custom MAPE summary function for caret
#' @param data Data frame with obs and pred columns
#' @param lev Levels (not used for regression)
#' @param model Model name (not used)
#' @return Named vector with MAPE value
mapeSummary <- function(data, lev = NULL, model = NULL) {
    mape <- mean(abs((data$obs - data$pred) / data$obs))
    c(MAPE = mape)
}

#' Print model evaluation metrics
#' @param actual Actual values
#' @param predicted Predicted values
#' @param model_name Name of the model for display
#' @return List with MAE and MAPE values
evaluate_model <- function(actual, predicted, model_name = "Model") {
    mae_val <- Metrics::mae(actual, predicted)
    mape_val <- Metrics::mape(actual, predicted)
    rmse_val <- sqrt(mean((actual - predicted)^2))

    cat(sprintf("\n%s Performance:\n", model_name))
    cat(sprintf("  MAE:  %.6f\n", mae_val))
    cat(sprintf("  MAPE: %.6f (%.2f%%)\n", mape_val, mape_val * 100))
    cat(sprintf("  RMSE: %.6f\n", rmse_val))

    return(list(MAE = mae_val, MAPE = mape_val, RMSE = rmse_val))
}

#' Save predictions to CSV file
#' @param predictions Data frame with predictions
#' @param filename Output filename
#' @param subfolder Subfolder within predictions/ (default: "")
save_predictions <- function(predictions, filename, subfolder = "") {
    output_dir <- file.path("predictions", subfolder)
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

    filepath <- file.path(output_dir, filename)
    write.csv(predictions, filepath, quote = FALSE, row.names = FALSE)
    cat(sprintf("✓ Predictions saved to: %s\n", filepath))
}

#' Create train/validation split with stratification
#' @param data Data frame
#' @param target_col Target column name
#' @param prop Proportion for training (default: 0.8)
#' @return List with train and valid data frames
create_split <- function(data, target_col = "Energy", prop = 0.8) {
    set.seed(GLOBAL_SEED)

    # Create stratification variable based on target quantiles
    data$strata_temp <- cut(data[[target_col]],
        breaks = quantile(data[[target_col]],
            probs = seq(0, 1, 0.2),
            na.rm = TRUE
        ),
        include.lowest = TRUE,
        labels = FALSE
    )

    split_idx <- createDataPartition(data$strata_temp, p = prop, list = FALSE)

    train_data <- data[split_idx, ]
    valid_data <- data[-split_idx, ]

    # Remove temporary stratification column
    train_data$strata_temp <- NULL
    valid_data$strata_temp <- NULL

    return(list(train = train_data, valid = valid_data))
}

# ============================================================================
# 5. PRINT CONFIGURATION SUMMARY
# ============================================================================

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("          Energy Consumption Prediction - Configuration        \n")
cat("═══════════════════════════════════════════════════════════════\n")
cat(sprintf("  Random Seed:           %d\n", GLOBAL_SEED))
cat(sprintf("  CV Folds:              %d\n", CV_FOLDS))
cat(sprintf("  Training Sample Size:  %d\n", TRAIN_SAMPLE_SIZE))
cat(sprintf("  Freeze Sample Size:    %d\n", FREEZE_SAMPLE_SIZE))
cat(sprintf("  Working Directory:     %s\n", getwd()))
cat("═══════════════════════════════════════════════════════════════\n\n")
