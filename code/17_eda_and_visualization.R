# ============================================================================
# 17_EDA_AND_VISUALIZATION.R - Exploratory Data Analysis and Visualization
# ============================================================================
# Description: Comprehensive EDA including correlation analysis, feature
#              importance, statistical tests, and visualizations.
# ============================================================================

source("00_config.R")
library(ggplot2)
library(rpart)
library(rpart.plot)
library(ltm)
library(effectsize)

# Load data
load("df_base.RData")

# Filter to training data
train_data <- df_base[!is.na(df_base$Energy), ]

# ============================================================================
# 1. BASIC DATA OVERVIEW
# ============================================================================

cat("============================================\n")
cat("Data Overview\n")
cat("============================================\n")
cat(sprintf("Total observations: %d\n", nrow(df_base)))
cat(sprintf("Training observations: %d\n", nrow(train_data)))
cat(sprintf("Test observations: %d\n", sum(is.na(df_base$Energy))))
cat(sprintf("Number of features: %d\n", ncol(df_base) - 2))  # Exclude BS and Energy

# Target variable summary
cat("\nEnergy (Target) Summary:\n")
print(summary(train_data$Energy))

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================

cat("\n============================================\n")
cat("Correlation Analysis\n")
cat("============================================\n")

# Select numeric columns
numeric_cols <- names(train_data)[sapply(train_data, is.numeric)]
numeric_cols <- numeric_cols[numeric_cols != "BS"]

# Calculate correlations with target
correlations <- sapply(numeric_cols, function(col) {
  if (col != "Energy") {
    cor(train_data[[col]], train_data$Energy, use = "complete.obs")
  } else {
    NA
  }
})

correlations <- correlations[!is.na(correlations)]
correlations <- sort(correlations, decreasing = TRUE)

cat("\nTop correlations with Energy:\n")
print(head(correlations, 10))

cat("\nBottom correlations with Energy:\n")
print(tail(correlations, 5))

# ============================================================================
# 3. ANOVA FOR CATEGORICAL VARIABLES
# ============================================================================

cat("\n============================================\n")
cat("ANOVA Analysis for Categorical Variables\n")
cat("============================================\n")

factor_cols <- names(train_data)[sapply(train_data, is.factor)]
factor_cols <- factor_cols[factor_cols != "BS"]

anova_results <- data.frame(
  Variable = character(),
  F_value = numeric(),
  P_value = numeric(),
  stringsAsFactors = FALSE
)

for (col in factor_cols) {
  if (length(unique(train_data[[col]])) > 1) {
    formula_str <- as.formula(paste("Energy ~", col))
    anova_fit <- aov(formula_str, data = train_data)
    summary_fit <- summary(anova_fit)
    
    f_val <- summary_fit[[1]][1, "F value"]
    p_val <- summary_fit[[1]][1, "Pr(>F)"]
    
    anova_results <- rbind(anova_results, data.frame(
      Variable = col,
      F_value = f_val,
      P_value = p_val
    ))
  }
}

anova_results <- anova_results[order(-anova_results$F_value), ]
cat("\nANOVA Results (sorted by F-value):\n")
print(anova_results)

# ============================================================================
# 4. EFFECT SIZE ANALYSIS
# ============================================================================

cat("\n============================================\n")
cat("Effect Size Analysis (Eta-squared)\n")
cat("============================================\n")

eta_results <- data.frame(
  Variable = character(),
  Eta_squared = numeric(),
  stringsAsFactors = FALSE
)

for (col in factor_cols[1:min(5, length(factor_cols))]) {
  if (length(unique(train_data[[col]])) > 1) {
    formula_str <- as.formula(paste("Energy ~", col))
    anova_fit <- aov(formula_str, data = train_data)
    eta_sq <- effectsize::eta_squared(anova_fit)
    
    eta_results <- rbind(eta_results, data.frame(
      Variable = col,
      Eta_squared = eta_sq$Eta2[1]
    ))
  }
}

eta_results <- eta_results[order(-eta_results$Eta_squared), ]
cat("\nEta-squared Results:\n")
print(eta_results)

# ============================================================================
# 5. POINT-BISERIAL CORRELATION
# ============================================================================

cat("\n============================================\n")
cat("Point-Biserial Correlations (Binary Variables)\n")
cat("============================================\n")

binary_vars <- c("es1_load_low", "es3_load_high", "es6_load_high", 
                 "freq_cell_anten_high", "freq_cell_anten_low",
                 "high_use", "low_use", "load_hour")

pb_correlations <- data.frame(
  Variable = character(),
  Correlation = numeric(),
  stringsAsFactors = FALSE
)

for (var in binary_vars) {
  if (var %in% names(train_data)) {
    binary_var <- as.numeric(as.character(train_data[[var]]))
    if (length(unique(binary_var)) == 2) {
      rpb <- ltm::biserial.cor(train_data$Energy, binary_var)
      pb_correlations <- rbind(pb_correlations, data.frame(
        Variable = var,
        Correlation = rpb
      ))
    }
  }
}

pb_correlations <- pb_correlations[order(-abs(pb_correlations$Correlation)), ]
cat("\nPoint-Biserial Correlations:\n")
print(pb_correlations)

# ============================================================================
# 6. DECISION TREE VISUALIZATION
# ============================================================================

cat("\n============================================\n")
cat("Decision Tree for Feature Importance\n")
cat("============================================\n")

# Fit decision tree
tree_data <- train_data[, !(names(train_data) %in% c("BS", "Time"))]
tree_fit <- rpart(Energy ~ ., data = tree_data, 
                  control = rpart.control(maxdepth = 4))

# Print variable importance
cat("\nVariable Importance (Top 10):\n")
var_imp <- tree_fit$variable.importance
var_imp <- sort(var_imp, decreasing = TRUE)
print(head(var_imp, 10))

# Save tree plot (if running interactively)
if (interactive()) {
  rpart.plot(tree_fit, cex = 0.5, main = "Energy Prediction Decision Tree")
}

# ============================================================================
# 7. SUMMARY STATISTICS BY KEY GROUPS
# ============================================================================

cat("\n============================================\n")
cat("Energy Statistics by Key Groups\n")
cat("============================================\n")

# By RUType
cat("\nBy RUType:\n")
print(train_data %>%
        group_by(RUType) %>%
        summarise(
          n = n(),
          mean = mean(Energy),
          median = median(Energy),
          sd = sd(Energy),
          .groups = "drop"
        ))

# By type_use
cat("\nBy type_use:\n")
print(train_data %>%
        group_by(type_use) %>%
        summarise(
          n = n(),
          mean = mean(Energy),
          median = median(Energy),
          sd = sd(Energy),
          .groups = "drop"
        ))

# By anten_cat
cat("\nBy anten_cat:\n")
print(train_data %>%
        group_by(anten_cat) %>%
        summarise(
          n = n(),
          mean = mean(Energy),
          median = median(Energy),
          sd = sd(Energy),
          .groups = "drop"
        ))

# ============================================================================
# 8. SAVE EDA RESULTS
# ============================================================================

cat("\nSaving EDA results...\n")

# Combine all results
eda_results <- list(
  correlations = correlations,
  anova_results = anova_results,
  eta_results = eta_results,
  pb_correlations = pb_correlations,
  var_importance = var_imp
)

saveRDS(eda_results, "eda_results.rds")

cat("EDA complete. Results saved to eda_results.rds\n")
