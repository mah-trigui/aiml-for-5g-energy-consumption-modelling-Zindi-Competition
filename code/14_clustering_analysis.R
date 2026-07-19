# ============================================================================
# 14_CLUSTERING_ANALYSIS.R - Clustering Analysis
# ============================================================================
# Description: Performs clustering analysis on the base dataset to identify
#              energy consumption patterns and create cluster-based features.
# ============================================================================

source("00_config.R")
library(rpart)
library(rpart.plot)

# Load base dataset
load("df_base.RData")

# ============================================================================
# 1. PREPARE DATA FOR CLUSTERING
# ============================================================================

cat("Preparing data for clustering...\n")

# Select relevant columns and exclude identifiers
cols_to_exclude <- c("BS", "Energy", "Time", "freq_max", "freq_min", 
                     "band_min", "tree_energy_pack_N", "tree_energy_rpart_N")
df_clus <- df_base[, !(names(df_base) %in% cols_to_exclude)]

# Remove any clustering results if they exist
if ("clus" %in% names(df_clus)) df_clus$clus <- NULL
if ("cluster_rules" %in% names(df_clus)) df_clus$cluster_rules <- NULL

# Convert factors to numeric for clustering
factor_columns <- sapply(df_clus, is.factor)
df_clus_num <- df_clus
df_clus_num[factor_columns] <- lapply(df_clus_num[factor_columns], function(x) as.numeric(as.character(x)))

# Scale numeric columns
numeric_columns <- sapply(df_clus_num, is.numeric)
df_clus_num[numeric_columns] <- scale(df_clus_num[numeric_columns])

cat(sprintf("  Clustering features: %d\n", ncol(df_clus_num)))

# ============================================================================
# 2. K-MEANS CLUSTERING
# ============================================================================

cat("\nPerforming K-Means clustering...\n")

# Try different k values
k_values <- 3:8
cluster_results <- list()

for (k in k_values) {
  set.seed(GLOBAL_SEED)
  kmeans_result <- kmeans(df_clus_num, k, iter.max = 10000, nstart = 21)
  
  # Calculate quality metrics
  between_ss <- kmeans_result$betweenss
  total_ss <- kmeans_result$totss
  ratio <- between_ss / total_ss
  
  cluster_results[[as.character(k)]] <- list(
    k = k,
    ratio = ratio,
    within_ss = sum(kmeans_result$withinss),
    clusters = kmeans_result$cluster
  )
  
  cat(sprintf("  k=%d: Between/Total SS ratio = %.4f\n", k, ratio))
}

# Select optimal k (e.g., k=6 based on original analysis)
optimal_k <- 6
set.seed(GLOBAL_SEED)
final_kmeans <- kmeans(df_clus_num, optimal_k, iter.max = 10000, nstart = 21)

cat(sprintf("\nSelected k=%d with ratio=%.4f\n", optimal_k, 
            final_kmeans$betweenss / final_kmeans$totss))

# Add cluster assignments to original data
df_base$clus <- as.factor(final_kmeans$cluster)

# ============================================================================
# 3. ANALYZE CLUSTERS
# ============================================================================

cat("\nAnalyzing clusters...\n")

# Cluster distribution
cat("\nCluster distribution:\n")
print(table(df_base$clus))

# Filter to training data only
train_data <- df_base[!is.na(df_base$Energy), ]

# Statistical significance of clusters
cat("\nANOVA for Energy by Cluster:\n")
anova_result <- aov(Energy ~ clus, data = train_data)
print(summary(anova_result))

# Effect size
if (requireNamespace("effectsize", quietly = TRUE)) {
  eta_sq <- effectsize::eta_squared(anova_result)
  cat("\nEta-squared (effect size):\n")
  print(eta_sq)
}

# Summary statistics by cluster
summary_stats <- train_data %>%
  group_by(clus) %>%
  summarise(
    n = n(),
    mean_energy = mean(Energy),
    median_energy = median(Energy),
    min_energy = min(Energy),
    max_energy = max(Energy),
    q25 = quantile(Energy, 0.25),
    q75 = quantile(Energy, 0.75),
    .groups = "drop"
  )

cat("\nEnergy statistics by cluster:\n")
print(summary_stats)

# ============================================================================
# 4. CREATE RULE-BASED CLUSTERS
# ============================================================================

cat("\nCreating rule-based clusters...\n")

# Based on decision tree analysis from original code
df_base <- df_base %>%
  mutate(
    cluster_rules = case_when(
      # Cluster 5: High complexity bases
      (band_nb == 1) |
        (band_nb == 0 & es6_load_high == 0 & ESMode1 < 0.41 & 
         ESMode6 < -0.0033 & anten_cat == 32) ~ 5,
      
      # Cluster 4: Medium-high load
      (band_nb == 0 & es6_load_high == 1 & es1_load_low == 0) ~ 4,
      
      # Cluster 1: Low energy usage
      (band_nb == 0 & es6_load_high == 0 & ESMode1 >= 0.41) ~ 1,
      
      # Cluster 3: Medium usage
      (band_nb == 0 & es6_load_high == 1 & es1_load_low == 1) |
        (band_nb == 0 & es6_load_high == 0 & ESMode1 < 0.41 & 
         ESMode6 < -0.0033 & anten_cat == 2) ~ 3,
      
      # Cluster 2: Default
      TRUE ~ 2
    )
  )

cat("\nRule-based cluster distribution:\n")
print(table(df_base$cluster_rules))

# ============================================================================
# 5. DECISION TREE FOR CLUSTER INTERPRETATION
# ============================================================================

cat("\nFitting decision tree for cluster interpretation...\n")

# Fit tree to predict clusters
tree_data <- df_base[, !(names(df_base) %in% c("BS", "Energy", "Time"))]
tree_model <- rpart(clus ~ ., data = tree_data, method = 'class')

# Print rules
cat("\nDecision tree rules:\n")
print(rpart.rules(tree_model))

# Save tree plot
if (interactive()) {
  rpart.plot(tree_model, cex = 0.7)
}

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================

cat("\nSaving clustering results...\n")

# Save updated dataset with clusters
save(df_base, file = "df_base_clustered.RData")
write.csv2(df_base, "df_base_clustered.csv", quote = FALSE, row.names = FALSE)

# Save cluster summary
write.csv(summary_stats, "cluster_summary.csv", quote = FALSE, row.names = FALSE)

# Save kmeans model
saveRDS(final_kmeans, "model_kmeans.rds")

cat("Clustering analysis complete.\n")
