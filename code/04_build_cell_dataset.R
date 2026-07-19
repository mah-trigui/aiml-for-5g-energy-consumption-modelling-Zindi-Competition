# ============================================================================
# 04_BUILD_CELL_DATASET.R - Build Cell-Level Dataset
# ============================================================================
# Description: Creates the cell-level dataset with all engineered features.
#              Handles energy splitting for multi-cell base stations.
# ============================================================================

source("02_feature_engineering.R")

# Load cleaned base data
load("data_base_cleaned.RData")
energy <- read.csv(FILE_ENERGY)
cell   <- read.csv(FILE_CELL)

# ============================================================================
# 1. PREPARE BASE DATA FOR CELL LEVEL
# ============================================================================

cat("Preparing base data for cell-level analysis...\n")

# Add cell count and CellName to base
base <- base %>%
  group_by(BS) %>%
  mutate(CellName = paste0("Cell", row_number() - 1)) %>%
  ungroup() %>%
  left_join(
    base %>% group_by(BS) %>% summarise(nb_cell = n(), .groups = "drop"),
    by = "BS"
  )

# ============================================================================
# 2. CALCULATE ENERGY SPLIT FOR MULTI-CELL BASES
# ============================================================================

cat("Calculating energy proportions for multi-cell bases...\n")

# Join frequency info to energy
energy <- energy %>%
  left_join(unique(base[, c("BS", "freq_cat", "CellName", "nb_cell")]), by = c("BS", "CellName"))

# Calculate energy proportion based on frequency
energy <- energy %>%
  mutate(
    perct = case_when(
      nb_cell == 2 & freq_cat == '365'    ~ AVG_365,
      nb_cell == 2 & freq_cat == '426.98' ~ AVG_426,
      nb_cell == 2 & freq_cat == '155.6'  ~ AVG_155,
      nb_cell == 2 & freq_cat == '189'    ~ AVG_189,
      TRUE ~ 1
    ),
    target = Energy * perct
  )

# Get actual cell count per time slot
cell_counts <- cell %>%
  group_by(Time, BS) %>%
  summarise(nb = n(), .groups = "drop")

# Calculate final cell energy
energy <- energy %>%
  left_join(cell_counts, by = c("Time", "BS")) %>%
  mutate(cell_energy = ifelse(nb == 1, Energy, target)) %>%
  select(BS, CellName, Time, cell_energy) %>%
  rename(Energy = cell_energy)

energy <- unique(energy)

# ============================================================================
# 3. BUILD CELL-LEVEL DATAFRAME
# ============================================================================

cat("Building cell-level dataframe...\n")

df_cell <- cell %>%
  left_join(energy, by = c("BS", "CellName", "Time"))

# ============================================================================
# 4. CREATE TIME FEATURES
# ============================================================================

cat("Creating time-based features...\n")

df_cell$Time <- strptime(df_cell$Time, format = "%m/%d/%Y %H:%M")
df_cell$hour <- as.numeric(format(df_cell$Time, format = "%H"))
df_cell$day_of_week <- as.integer(format(df_cell$Time, format = "%w"))
df_cell$day_of_week <- ifelse(df_cell$day_of_week == 0, 'Sun',
                               ifelse(df_cell$day_of_week == 6, 'Sat', 'week'))

# Store Time info for later use
df_cell_info <- df_cell[, c("BS", "CellName", "Time", "hour")]
df_cell$Time <- NULL

# ============================================================================
# 5. JOIN BASE STATION INFO
# ============================================================================

cat("Joining base station information...\n")

df_cell <- df_cell %>%
  left_join(
    unique(base[, c("BS", "RUType", "Frequency", "Bandwidth", "anten_cat", "nb_cell")]),
    by = "BS"
  )

# Filter out records with missing bandwidth
df_cell <- df_cell[!is.na(df_cell$Bandwidth), ]

# Convert anten_cat to numeric for calculations
df_cell$anten_cat <- as.numeric(as.character(df_cell$anten_cat))

# Remove unused columns
df_cell$CellName <- NULL
df_cell$ESMode2 <- NULL
df_cell$ESMode4 <- NULL
df_cell$ESMode5 <- NULL

# ============================================================================
# 6. CREATE DOMAIN-SPECIFIC FEATURES
# ============================================================================

cat("Creating domain-specific features...\n")

# -- Load-based hour indicator --
df_cell$load_hour <- create_load_hour(df_cell$hour)

# -- Tree-based energy categories --
df_cell$tree_energy_pack_F <- case_when(
  df_cell$nb_cell == 1 & df_cell$load <= 0.08 ~ 1,
  (df_cell$nb_cell == 1 & df_cell$load >= 0.6) |
    (df_cell$nb_cell == 2 & df_cell$load <= 0.37 & df_cell$anten_cat != 4) ~ 3,
  (df_cell$nb_cell == 2 & df_cell$load > 0.37 & df_cell$anten_cat != 4) |
    (df_cell$nb_cell == 2 & df_cell$load <= 0.37 & df_cell$anten_cat == 4) ~ 4,
  df_cell$nb_cell == 2 & df_cell$load > 0.37 & df_cell$anten_cat == 4 ~ 5,
  TRUE ~ 2
)

# -- ES Mode indicators --
df_cell$es1_load_low <- case_when(
  (df_cell$ESMode1 >= 0.47 & df_cell$load <= 0.044 & df_cell$anten_cat %in% c(8, 32, 64)) |
    (df_cell$nb_cell == 1 & df_cell$load <= 0.5) |
    (df_cell$nb_cell == 2 & df_cell$ESMode1 >= 0.46 & df_cell$load <= 0.044) ~ 1,
  TRUE ~ 0
)
df_cell$es3_load_high <- ifelse(df_cell$ESMode3 == 0 & df_cell$load >= 0.055, 1, 0)
df_cell$es6_load_high <- ifelse(df_cell$ESMode6 <= 0.13 & df_cell$load >= 0.055, 1, 0)

# -- Frequency-based indicators --
df_cell$freq_cell_anten_low <- case_when(
  (df_cell$Frequency == 426.98 & df_cell$nb_cell == 1 & df_cell$anten_cat == 1) |
    (df_cell$Frequency == 697.002 & df_cell$nb_cell == 2) ~ 1,
  TRUE ~ 0
)
df_cell$freq_cell_anten_high <- ifelse(
  df_cell$Frequency == 426.98 & df_cell$nb_cell == 2 & df_cell$anten_cat == 4, 1, 0
)

# -- Usage indicators --
df_cell$high_use <- case_when(
  (df_cell$RUType == 'Type1' & df_cell$anten_cat == 4) &
    ((df_cell$load <= 0.4 & df_cell$load_hour == 1) |
       (df_cell$hour %in% c(13, 14, 15, 16, 17, 18, 22)) |
       (df_cell$es3_load_high == 1)) ~ 1,
  TRUE ~ 0
)
df_cell$low_use <- case_when(
  (df_cell$ESMode1 >= 0.7 & df_cell$load >= 0.03) |
    (df_cell$Frequency == 697.002 & df_cell$nb_cell == 2) |
    (df_cell$Frequency == 426.98 & df_cell$nb_cell == 1 & df_cell$anten_cat == 1) ~ 1,
  TRUE ~ 0
)

# -- Type categorizations (numeric) --
df_cell$type_use <- case_when(
  df_cell$RUType %in% c('Type1', 'Type10', 'Type11') ~ 3,
  df_cell$RUType %in% c('Type12', 'Type7', 'Type8') ~ 2,
  df_cell$RUType %in% c('Type2', 'Type3', 'Type4', 'Type5', 'Type6', 'Type9') ~ 1
)

df_cell$hour_ch <- case_when(
  df_cell$hour %in% c(2:6) ~ 1,
  df_cell$hour %in% c(0, 1, 7:11) ~ 2,
  TRUE ~ 3
)

# ============================================================================
# 7. RECODE FACTOR LEVELS
# ============================================================================

cat("Recoding factor levels...\n")

df_cell$RUType <- as.factor(df_cell$RUType)
df_cell$RUType <- fct_recode(df_cell$RUType, 'Type6' = "Type9")
df_cell$RUType <- fct_recode(df_cell$RUType, 'Type7' = "Type8")
df_cell$RUType <- fct_recode(df_cell$RUType, 'Type10' = "Type11")
df_cell$RUType <- fct_recode(df_cell$RUType, 'Type10' = "Type12")
df_cell$RUType <- fct_recode(df_cell$RUType, 'Type4' = "Type2")
df_cell$RUType <- fct_recode(df_cell$RUType, 'Type4' = "Type6")

# Remove hour (keep hour_ch)
df_cell$hour <- NULL

# ============================================================================
# 8. APPLY ENCODINGS
# ============================================================================

cat("Applying categorical encodings...\n")

# Helmert encoding for day_of_week
d <- encode_helmert(df_cell, "day_of_week")
lookup <- d %>% distinct(day_of_week, day_of_week1)
df_cell <- merge(df_cell, lookup, by = "day_of_week", all.x = TRUE)
df_cell$day_of_week <- NULL

# Polynomial encoding for RUType
d <- encode_polynomial(df_cell, "RUType")
lookup <- d %>% distinct(RUType, RUType1)
df_cell <- merge(df_cell, lookup, by = "RUType", all.x = TRUE)
df_cell$RUType <- NULL

# Rename encoded columns
names(df_cell) <- sub("1$", "", names(df_cell))

# ============================================================================
# 9. CREATE FORMULA-BASED FEATURES
# ============================================================================

cat("Creating formula-based features...\n")

df_cell$formula <- (df_cell$Bandwidth / sqrt(df_cell$Frequency)) * log(df_cell$anten_cat)
df_cell$formula_2 <- exp(df_cell$Bandwidth^2 / df_cell$Frequency) * 
                     log(df_cell$anten_cat) * sqrt(df_cell$load)

# Transform Frequency
df_cell$Frequency <- poly(df_cell$Frequency, 2)[, 2]

# ============================================================================
# 10. SAVE CELL DATASET
# ============================================================================

cat("\nSaving cell dataset...\n")

save(df_cell, file = "df_cell.RData")
write.csv2(df_cell, "df_cell.csv", quote = FALSE, row.names = FALSE)

cat(sprintf("Cell dataset created: %d rows, %d columns\n", nrow(df_cell), ncol(df_cell)))
cat(sprintf("Training samples (with Energy): %d\n", sum(!is.na(df_cell$Energy))))
cat(sprintf("Test samples (without Energy): %d\n", sum(is.na(df_cell$Energy))))
