# ============================================================================
# 01_data_loading.R - Data Loading and Initial Processing
# ============================================================================
# Description: Loads all raw data files and performs initial cleaning/merging
# Updated: Comprehensive version consolidating all data loading operations
# ============================================================================

cat("\n")
cat("═══════════════════════════════════════════════════════════════\n")
cat("  STEP 1: Loading and Processing Raw Data                      \n")
cat("═══════════════════════════════════════════════════════════════\n\n")

# ============================================================================
# 1. LOAD RAW DATA FILES
# ============================================================================

cat("Loading raw data files...\n")

# Load Energy data
energy <- fread(FILE_ENERGY, data.table = FALSE)
cat(sprintf("✓ Energy data loaded: %d rows\n", nrow(energy)))

# Load Cell configuration data
cell <- fread(FILE_CELL, data.table = FALSE)
cat(sprintf("✓ Cell data loaded: %d rows\n", nrow(cell)))

# Load Base Station data
base <- fread(FILE_BASE, data.table = FALSE)
cat(sprintf("✓ Base Station data loaded: %d rows\n", nrow(base)))

# Load Submission template
submission <- fread(FILE_SUBMISSION, data.table = FALSE)
cat(sprintf("✓ Submission template loaded: %d rows\n", nrow(submission)))

# ============================================================================
# 2. DATA QUALITY FIXES
# ============================================================================

cat("\nApplying data quality fixes...\n")

# Fix Time/BS column names
names(energy)[names(energy) == "Time "] <- "Time"
names(energy)[names(energy) == "BS "] <- "BS"
names(cell)[names(cell) == "BS "] <- "BS"
names(base)[names(base) == "BS "] <- "BS"

# Extract time and BS from submission template
submission <- submission %>%
    mutate(
        Time = as.numeric(gsub("_.*", "", Time)),
        BS = as.numeric(gsub(".*_", "", Time))
    )

# Convert data types
energy$Time <- as.numeric(energy$Time)
energy$BS <- as.numeric(energy$BS)
energy$Energy <- as.numeric(energy$Energy)

cell$BS <- as.numeric(cell$BS)
cell$Cell <- as.numeric(cell$Cell)
cell$Freq <- as.numeric(cell$Freq)
cell$RRUNumber <- as.numeric(cell$RRUNumber)

base$BS <- as.numeric(base$BS)
base$RUType <- as.factor(base$RUType)

cat("✓ Data types converted.\n")

# ============================================================================
# 3. MERGE DATASETS
# ============================================================================

cat("\nMerging datasets...\n")

# Create complete time-BS combinations for submission
time_bs_template <- expand.grid(
    Time = unique(c(energy$Time, submission$Time)),
    BS = unique(c(energy$BS, energy$BS, cell$BS, base$BS, submission$BS))
)

# Merge energy data with template
energy_full <- time_bs_template %>%
    left_join(energy, by = c("Time", "BS"))

# Add Cell information
energy_cell <- energy_full %>%
    left_join(cell, by = "BS")

# Add Base Station information
energy_base_cell <- energy_cell %>%
    left_join(base, by = "BS")

cat(sprintf("✓ Merged dataset created: %d rows\n", nrow(energy_base_cell)))

# ============================================================================
# 4. CALCULATE AGGREGATED FEATURES
# ============================================================================

cat("\nCalculating aggregated features...\n")

# Base station level aggregations
base_agg <- energy_base_cell %>%
    group_by(BS, RUType) %>%
    summarise(
        nb_cell = n_distinct(Cell, na.rm = TRUE),
        freq_nb = n_distinct(Freq, na.rm = TRUE),
        band_nb = n_distinct(Band, na.rm = TRUE),
        power_nb = n_distinct(Power, na.rm = TRUE),
        freq_avg = mean(Freq, na.rm = TRUE),
        band_avg = mean(Band, na.rm = TRUE),
        power_sum = sum(Power, na.rm = TRUE),
        freq_max = max(Freq, na.rm = TRUE),
        freq_min = min(Freq, na.rm = TRUE),
        band_max = max(Band, na.rm = TRUE),
        band_min = min(Band, na.rm = TRUE),
        .groups = "drop"
    )

# Add aggregations back to main dataset
data_with_agg <- energy_base_cell %>%
    left_join(base_agg, by = c("BS", "RUType"))

cat("✓ Aggregated features calculated.\n")

# ============================================================================
# 5. CREATE TIME-BASED FEATURES
# ============================================================================

cat("\nCreating time-based features...\n")

data_with_features <- data_with_agg %>%
    mutate(
        # Extract hour from Time (0-23)
        hour = Time %% 24,

        # Day of week (0-6, where 0 is Monday)
        day_of_week = as.integer((Time %/% 24) %% 7),

        # Hour category
        hour_ch = case_when(
            hour %in% c(2:6) ~ "L", # Low usage hours
            hour %in% c(0, 1, 7:11) ~ "M", # Medium usage hours
            TRUE ~ "H" # High usage hours (12-23)
        ),

        # Load hour indicator (low usage period: 3-7am)
        load_hour = ifelse(hour %in% c(3:7), 0, 1),

        # Weekend indicator
        is_weekend = ifelse(day_of_week %in% c(5, 6), 1, 0)
    )

cat("✓ Time-based features created.\n")

# ============================================================================
# 6. CONVERT TO APPROPRIATE DATA TYPES
# ============================================================================

cat("\nConverting data types...\n")

# Convert categorical variables to factors
categorical_vars <- c("RUType", "hour_ch", "day_of_week")
data_with_features <- data_with_features %>%
    mutate(across(all_of(categorical_vars), as.factor))

# Convert some numeric aggregations to factors (for modeling)
factor_vars <- c("freq_max", "freq_min", "band_max", "band_min")
data_with_features <- data_with_features %>%
    mutate(across(all_of(factor_vars), as.factor))

cat("✓ Data types converted.\n")

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================

cat("\nSaving processed data...\n")

# Save complete processed dataset
save(data_with_features, file = "data/01_data_loaded.RData")
fwrite(data_with_features, "data/01_data_loaded.csv")

# Create train/test split indicators
data_final <- data_with_features %>%
    mutate(
        is_train = !is.na(Energy),
        is_test = is.na(Energy)
    )

# Export for next steps
assign("raw_data", data_final, envir = .GlobalEnv)

cat(sprintf("✓ Data saved to: data/01_data_loaded.RData\n"))
cat(sprintf("✓ Total observations: %d\n", nrow(data_final)))
cat(sprintf("✓ Training observations: %d\n", sum(data_final$is_train)))
cat(sprintf("✓ Test observations: %d\n", sum(data_final$is_test)))

# ============================================================================
# 8. DATA SUMMARY
# ============================================================================

cat("\n")
cat("─────────────────────────────────────────────────────────────\n")
cat("Data Loading Summary:\n")
cat("─────────────────────────────────────────────────────────────\n")
cat(sprintf("  Unique Base Stations: %d\n", n_distinct(data_final$BS)))
cat(sprintf("  Unique Cells: %d\n", n_distinct(data_final$Cell, na.rm = TRUE)))
cat(sprintf("  Time Points: %d\n", n_distinct(data_final$Time)))
cat(sprintf("  RU Types: %d\n", n_distinct(data_final$RUType, na.rm = TRUE)))
cat(sprintf(
    "  Missing Energy Values: %d (%.2f%%)\n",
    sum(is.na(data_final$Energy)),
    100 * sum(is.na(data_final$Energy)) / nrow(data_final)
))
cat("─────────────────────────────────────────────────────────────\n\n")

cat("✓ Data loading complete!\n\n")
