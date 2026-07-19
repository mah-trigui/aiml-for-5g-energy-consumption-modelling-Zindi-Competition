# ============================================================================
# 01_DATA_LOADING.R - Data Import and Initial Cleaning
# ============================================================================
# Description: Loads raw data files and performs initial data quality fixes
#              including handling duplicates and known data errors.
# ============================================================================

source("00_config.R")

# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

cat("Loading raw data files...\n")

submission <- read.csv(FILE_SUBMISSION)
energy <- read.csv(FILE_ENERGY)
cell <- read.csv(FILE_CELL)
base <- read.csv(FILE_BASE)

cat(sprintf("  Submission: %d rows\n", nrow(submission)))
cat(sprintf("  Energy:     %d rows\n", nrow(energy)))
cat(sprintf("  Cell:       %d rows\n", nrow(cell)))
cat(sprintf("  Base:       %d rows\n", nrow(base)))

# ============================================================================
# 2. DATA QUALITY FIXES - BASE STATION DATA
# ============================================================================

cat("\nApplying data quality fixes to base station data...\n")

# -- Fix Antenna Inconsistencies --
# Replace the min antenna with the max value since it's a base characteristic
base$Antennas <- ave(
    base$Antennas,
    base$BS,
    FUN = function(x) if (length(unique(x)) > 1) max(x) else x
)

# -- Create TXpower Character Column --
base$TXpower_ch <- as.character(base$TXpower)

# -- Fix Bandwidth Issues --
# No 5MHz bandwidth exists - change to 10
base$Bandwidth[base$Bandwidth == 5] <- 10

# -- Fix Specific Station Errors --
base$Antennas[base$BS == "B_925"] <- 8
base$Bandwidth[base$Frequency == 364] <- 20
base$Frequency[base$Frequency == 364] <- 365
base$Antennas[base$RUType == "Type11"] <- 8

# -- Remove Invalid Records --
# Remove rows with invalid Frequency/Bandwidth combination
base <- base[!(base$Frequency == 426.98 & base$Bandwidth == 10), ]

# -- Create Factor Categories --
base$freq_cat <- as.factor(base$Frequency)
base$band_cat <- as.factor(base$Bandwidth)
base$anten_cat <- as.factor(base$Antennas)

# -- Remove CellName Column and Duplicates --
base <- base[, names(base) != "CellName"]
base <- unique(base)

# -- Create Cell Identifier --
base <- base %>%
    group_by(BS) %>%
    mutate(cellname = paste0("Cell", row_number() - 1)) %>%
    ungroup()

cat(sprintf("  Base stations after cleaning: %d unique records\n", nrow(base)))

# ============================================================================
# 3. CREATE BASE STATION AGGREGATED INFO
# ============================================================================

cat("Creating base station aggregated features...\n")

base_gp_info <- base %>%
    group_by(BS) %>%
    summarize(
        nb_cell   = n(),
        freq_nb   = n_distinct(Frequency),
        band_nb   = n_distinct(Bandwidth),
        power_nb  = n_distinct(TXpower),
        freq_avg  = mean(Frequency),
        freq_max  = max(Frequency),
        freq_min  = min(Frequency),
        band_avg  = mean(Bandwidth),
        band_max  = max(Bandwidth),
        band_min  = min(Bandwidth),
        power_sum = sum(TXpower),
        .groups   = "drop"
    )

cat(sprintf("  Aggregated info for %d base stations\n", nrow(base_gp_info)))

# ============================================================================
# 4. VERIFY DATA INTEGRITY
# ============================================================================

cat("\nData integrity checks:\n")

# Check unique counts
cat(sprintf("  Unique BS in base:   %d\n", length(unique(base$BS))))
cat(sprintf("  Unique BS in energy: %d\n", length(unique(energy$BS))))
cat(sprintf("  Unique BS in cell:   %d\n", length(unique(cell$BS))))

# Check RUType consistency (each base should have ONE RUType)
unique_rutypes <- sapply(split(base$RUType, base$BS), function(x) length(unique(x)))
if (all(unique_rutypes == 1)) {
    cat("  RUType consistency: OK (each base has one RUType)\n")
} else {
    cat("  WARNING: Some bases have multiple RUTypes!\n")
}

# Check Mode consistency
unique_modes <- sapply(split(base$Mode, base$BS), function(x) length(unique(x)))
if (all(unique_modes == 1)) {
    cat("  Mode consistency: OK (each base has one Mode)\n")
} else {
    cat("  WARNING: Some bases have multiple Modes!\n")
}

# ============================================================================
# 5. SAVE CLEANED DATA
# ============================================================================

cat("\nSaving cleaned base data...\n")

save(base, base_gp_info, file = "data_base_cleaned.RData")
write.csv2(base, "data_base_cleaned.csv", quote = FALSE, row.names = FALSE)

cat("Data loading complete.\n")
