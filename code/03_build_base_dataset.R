# ============================================================================
# 03_BUILD_BASE_DATASET.R - Build Base-Level Dataset
# ============================================================================
# Description: Creates the base station level dataset with all engineered
#              features. Aggregates cell-level data to base station level.
# ============================================================================

source("02_feature_engineering.R")

# Load data
load("data_base_cleaned.RData")
energy <- read.csv(FILE_ENERGY)
cell   <- read.csv(FILE_CELL)

# ============================================================================
# 1. AGGREGATE CELL DATA TO BASE LEVEL
# ============================================================================

cat("Aggregating cell data to base station level...\n")

df_base <- cell %>%
  group_by(BS, Time) %>%
  summarise(
    load          = mean(load),
    load_diff     = max(load) - min(load),
    ESMode1       = mean(ESMode1),
    ESMode3       = mean(ESMode3),
    ESMode6       = mean(ESMode6),
    ESMode1_diff  = max(ESMode1) - min(ESMode1),
    ESMode3_diff  = max(ESMode3) - min(ESMode3),
    ESMode6_diff  = max(ESMode6) - min(ESMode6),
    .groups       = "drop"
  ) %>%
  left_join(energy, by = c("BS", "Time"))

# ============================================================================
# 2. CREATE TIME FEATURES
# ============================================================================

cat("Creating time-based features...\n")

df_base$Time <- strptime(df_base$Time, format = "%m/%d/%Y %H:%M")
df_base$hour <- as.numeric(format(df_base$Time, format = "%H"))
df_base$day_of_week <- as.integer(format(df_base$Time, format = "%w"))
df_base$day_of_week <- ifelse(df_base$day_of_week == 0, 'Sun',
                              ifelse(df_base$day_of_week == 6, 'Sat', 'week'))
df_base$Time <- NULL

# ============================================================================
# 3. JOIN BASE STATION INFO
# ============================================================================

cat("Joining base station information...\n")

df_base <- df_base %>%
  left_join(unique(base[, c("BS", "RUType", "anten_cat")]), by = "BS") %>%
  left_join(base_gp_info, by = "BS")

# Convert factor columns
df_base$freq_max <- as.factor(df_base$freq_max)
df_base$band_max <- as.factor(df_base$band_max)
df_base$freq_min <- as.factor(df_base$freq_min)
df_base$band_min <- as.factor(df_base$band_min)

# ============================================================================
# 4. CREATE DOMAIN-SPECIFIC FEATURES
# ============================================================================

cat("Creating domain-specific features...\n")

# -- Load-based hour indicator --
df_base$load_hour <- create_load_hour(df_base$hour)

# -- Tree-based energy categories --
df_base$tree_energy_pack_F <- create_tree_energy_F(
  df_base$nb_cell, df_base$load, as.numeric(as.character(df_base$anten_cat))
)
df_base$tree_energy_pack_N <- create_tree_energy_N(
  df_base$nb_cell, df_base$load, as.numeric(as.character(df_base$anten_cat))
)

# -- ES Mode indicators --
df_base$es1_load_low <- create_es1_load_low(
  df_base$ESMode1, df_base$load,
  as.numeric(as.character(df_base$anten_cat)), df_base$nb_cell
)
df_base$es3_load_high <- create_es3_load_high(df_base$ESMode3, df_base$load)
df_base$es6_load_high <- create_es6_load_high(df_base$ESMode6, df_base$load)

# -- Frequency-cell-antenna indicators --
df_base$freq_cell_anten_low <- create_freq_cell_anten_low(
  as.character(df_base$freq_max), df_base$nb_cell, as.character(df_base$anten_cat)
)
df_base$freq_cell_anten_high <- create_freq_cell_anten_high(
  as.character(df_base$freq_max), df_base$nb_cell, as.character(df_base$anten_cat)
)

# -- Usage indicators --
df_base$high_use <- create_high_use(
  df_base$RUType, as.numeric(as.character(df_base$anten_cat)),
  df_base$load, df_base$load_hour, df_base$hour, df_base$es3_load_high
)
df_base$low_use <- create_low_use(
  df_base$ESMode1, df_base$load, as.character(df_base$freq_max),
  df_base$nb_cell, as.character(df_base$anten_cat)
)

# -- Type categorizations --
df_base$type_anten <- create_type_anten(df_base$RUType)
df_base$type_use <- create_type_use(df_base$RUType)
df_base$hour_ch <- create_hour_ch(df_base$hour)

# ============================================================================
# 5. RECODE FACTOR LEVELS
# ============================================================================

cat("Recoding factor levels...\n")

df_base$anten_cat <- recode_anten_cat(df_base$anten_cat)
df_base$RUType <- recode_rutype(as.factor(df_base$RUType))
df_base$freq_max <- recode_freq(df_base$freq_max)
df_base$freq_min <- recode_freq(df_base$freq_min)
df_base$band_max <- recode_band(df_base$band_max)
df_base$band_min <- recode_band(df_base$band_min)

# Adjust band_nb
df_base$band_nb <- df_base$band_nb - 1

# ============================================================================
# 6. CONVERT TO FACTORS
# ============================================================================

cat("Converting variables to factors...\n")

factor_vars <- c("day_of_week", "hour_ch", "band_nb", "load_hour",
                 "es1_load_low", "es3_load_high", "es6_load_high",
                 "freq_cell_anten_high", "freq_cell_anten_low",
                 "high_use", "low_use", "tree_energy_pack_F",
                 "type_anten", "type_use")

for (var in factor_vars) {
  if (var %in% names(df_base)) {
    df_base[[var]] <- as.factor(df_base[[var]])
  }
}

# ============================================================================
# 7. CREATE DECISION TREE BASED FEATURES
# ============================================================================

cat("Creating decision tree-based features...\n")

df_base <- df_base %>%
  mutate(
    tree_energy_rpart_F = case_when(
      es1_load_low != 1 & type_use != 'low' & high_use != 0 & load >= 0.3 ~ 'VH',
      (es1_load_low != 1 & type_use != 'low' & high_use != 0 & load < 0.3) |
        (es1_load_low != 1 & type_use != 'low' & high_use == 0 & load >= 0.25 & power_sum >= 14) ~ 'H',
      (es1_load_low == 1 & load < 0.14 & !(RUType %in% c('Type1', 'Type10')) &
         ESMode1 < 0.012) ~ 'L',
      es1_load_low == 1 & load < 0.14 & !(RUType %in% c('Type1', 'Type10')) &
        ESMode1 >= 0.012 ~ 'VL',
      TRUE ~ 'M'
    ),
    tree_energy_rpart_N = case_when(
      es1_load_low != 1 & type_use != 'low' & high_use != 0 & load >= 0.3 ~ 5,
      (es1_load_low != 1 & type_use != 'low' & high_use != 0 & load < 0.3) |
        (es1_load_low != 1 & type_use != 'low' & high_use == 0 & load >= 0.25 & power_sum >= 14) ~ 4,
      (es1_load_low == 1 & load < 0.14 & !(RUType %in% c('Type1', 'Type10')) &
         ESMode1 < 0.012) ~ 2,
      es1_load_low == 1 & load < 0.14 & !(RUType %in% c('Type1', 'Type10')) &
        ESMode1 >= 0.012 ~ 1,
      TRUE ~ 3
    )
  )

df_base$tree_energy_rpart_F <- as.factor(df_base$tree_energy_rpart_F)

# ============================================================================
# 8. REMOVE UNNECESSARY COLUMNS
# ============================================================================

cat("Removing unnecessary columns...\n")

cols_to_remove <- c("nb_cell", "freq_nb", "power_nb", "hour")
df_base <- df_base[, !(names(df_base) %in% cols_to_remove)]

# ============================================================================
# 9. SAVE BASE DATASET
# ============================================================================

cat("\nSaving base dataset...\n")

save(df_base, file = "df_base.RData")
write.csv2(df_base, "df_base.csv", quote = FALSE, row.names = FALSE)

cat(sprintf("Base dataset created: %d rows, %d columns\n", nrow(df_base), ncol(df_base)))
cat(sprintf("Training samples (with Energy): %d\n", sum(!is.na(df_base$Energy))))
cat(sprintf("Test samples (without Energy): %d\n", sum(is.na(df_base$Energy))))
