# ============================================================================
# 05_BUILD_LIGHTGBM_DATASET.R - Build LightGBM-Optimized Dataset
# ============================================================================
# Description: Creates a specially encoded dataset optimized for LightGBM
#              with various categorical encodings applied.
# ============================================================================

source("02_feature_engineering.R")

# Load base dataset
load("df_base.RData")

# ============================================================================
# 1. SELECT AND PREPARE COLUMNS
# ============================================================================

cat("Preparing LightGBM-optimized dataset...\n")

# Remove columns not needed for LightGBM
cols_to_remove <- c("freq_max", "freq_min", "band_min", 
                    "tree_energy_pack_N", "tree_energy_rpart_N")
df_ligh <- df_base[, !(names(df_base) %in% cols_to_remove)]

# ============================================================================
# 2. APPLY HELMERT ENCODING TO DAY_OF_WEEK
# ============================================================================

cat("Applying Helmert encoding to day_of_week...\n")

d <- encode_helmert(df_ligh, "day_of_week")
lookup <- d %>% distinct(day_of_week, day_of_week1)
df_ligh <- merge(df_ligh, lookup, by = "day_of_week", all.x = TRUE)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("day_of_week"))]

# ============================================================================
# 3. APPLY POLYNOMIAL ENCODING TO RUTYPE
# ============================================================================

cat("Applying polynomial encoding to RUType...\n")

d <- encode_polynomial(df_ligh, "RUType")
lookup <- d %>% distinct(RUType, RUType1)
df_ligh <- merge(df_ligh, lookup, by = "RUType", all.x = TRUE)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("RUType"))]

# ============================================================================
# 4. APPLY HELMERT ENCODING TO ANTEN_CAT
# ============================================================================

cat("Applying Helmert encoding to anten_cat...\n")

d <- encode_helmert(df_ligh, "anten_cat")
lookup <- d %>% distinct(anten_cat, anten_cat1)
df_ligh <- merge(df_ligh, lookup, by = "anten_cat", all.x = TRUE)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("anten_cat"))]

# ============================================================================
# 5. APPLY TARGET ENCODING TO BAND_MAX
# ============================================================================

cat("Applying target encoding to band_max...\n")

df_ligh[["band_max1"]] <- encode_target(df_ligh[["band_max"]], df_ligh[["Energy"]])
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("band_max"))]

# ============================================================================
# 6. APPLY M-ESTIMATOR ENCODING TO LOAD_HOUR
# ============================================================================

cat("Applying M-estimator encoding to load_hour...\n")

df_train <- df_ligh[!is.na(df_ligh$Energy), ]
df_test <- df_ligh[is.na(df_ligh$Energy), ]

encoder <- encode_m_estimator(df_train[["load_hour"]], df_train[["Energy"]])
df_train[["load_hour1"]] <- encoder$encoded
df_test[["load_hour1"]] <- encoder$encode_new(df_test[["load_hour"]])

df_ligh <- rbind(df_train, df_test)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("load_hour"))]

# ============================================================================
# 7. APPLY WOE ENCODING TO TREE_ENERGY_PACK_F
# ============================================================================

cat("Applying WoE encoding to tree_energy_pack_F...\n")

df_train <- df_ligh[!is.na(df_ligh$Energy), ]
df_test <- df_ligh[is.na(df_ligh$Energy), ]

encoder <- encode_woe(df_train[["tree_energy_pack_F"]], df_train[["Energy"]], sigma = 0.05)
df_train[["tree_energy_pack_F1"]] <- encoder$encoded
df_test[["tree_energy_pack_F1"]] <- encoder$encode_new(df_test[["tree_energy_pack_F"]])

df_ligh <- rbind(df_train, df_test)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("tree_energy_pack_F"))]

# ============================================================================
# 8. APPLY WOE ENCODING TO ES1_LOAD_LOW
# ============================================================================

cat("Applying WoE encoding to es1_load_low...\n")

df_train <- df_ligh[!is.na(df_ligh$Energy), ]
df_test <- df_ligh[is.na(df_ligh$Energy), ]

encoder <- encode_woe(df_train[["es1_load_low"]], df_train[["Energy"]])
df_train[["es1_load_low1"]] <- encoder$encoded
df_test[["es1_load_low1"]] <- encoder$encode_new(df_test[["es1_load_low"]])

df_ligh <- rbind(df_train, df_test)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("es1_load_low"))]

# ============================================================================
# 9. APPLY BACKWARD DIFFERENCE ENCODING TO TYPE_ANTEN
# ============================================================================

cat("Applying backward difference encoding to type_anten...\n")

d <- encode_backward_difference(df_ligh, "type_anten")
lookup <- d %>% distinct(type_anten, type_anten1)
df_ligh <- merge(df_ligh, lookup, by = "type_anten", all.x = TRUE)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("type_anten"))]

# ============================================================================
# 10. APPLY M-ESTIMATOR ENCODING TO TYPE_USE
# ============================================================================

cat("Applying M-estimator encoding to type_use...\n")

df_train <- df_ligh[!is.na(df_ligh$Energy), ]
df_test <- df_ligh[is.na(df_ligh$Energy), ]

encoder <- encode_m_estimator(df_train[["type_use"]], df_train[["Energy"]], sigma = 0.05)
df_train[["type_use1"]] <- encoder$encoded
df_test[["type_use1"]] <- encoder$encode_new(df_test[["type_use"]])

df_ligh <- rbind(df_train, df_test)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("type_use"))]

# ============================================================================
# 11. APPLY M-ESTIMATOR ENCODING TO TREE_ENERGY_RPART_F
# ============================================================================

cat("Applying M-estimator encoding to tree_energy_rpart_F...\n")

df_train <- df_ligh[!is.na(df_ligh$Energy), ]
df_test <- df_ligh[is.na(df_ligh$Energy), ]

encoder <- encode_m_estimator(df_train[["tree_energy_rpart_F"]], df_train[["Energy"]])
df_train[["tree_energy_rpart_F1"]] <- encoder$encoded
df_test[["tree_energy_rpart_F1"]] <- encoder$encode_new(df_test[["tree_energy_rpart_F"]])

df_ligh <- rbind(df_train, df_test)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("tree_energy_rpart_F"))]

# ============================================================================
# 12. APPLY WOE ENCODING TO HOUR_CH
# ============================================================================

cat("Applying WoE encoding to hour_ch...\n")

df_train <- df_ligh[!is.na(df_ligh$Energy), ]
df_test <- df_ligh[is.na(df_ligh$Energy), ]

encoder <- encode_woe(df_train[["hour_ch"]], df_train[["Energy"]])
df_train[["hour_ch1"]] <- encoder$encoded
df_test[["hour_ch1"]] <- encoder$encode_new(df_test[["hour_ch"]])

df_ligh <- rbind(df_train, df_test)
df_ligh <- df_ligh[, !(names(df_ligh) %in% c("hour_ch"))]

# ============================================================================
# 13. CLEAN UP COLUMN NAMES
# ============================================================================

cat("Cleaning up column names...\n")

# Remove trailing "1" from encoded column names
names(df_ligh) <- sub("1$", "", names(df_ligh))

# ============================================================================
# 14. SAVE LIGHTGBM DATASET
# ============================================================================

cat("\nSaving LightGBM-optimized dataset...\n")

save(df_ligh, file = "df_ligh.RData")
write.csv2(df_ligh, "df_ligh.csv", quote = FALSE, row.names = FALSE)

cat(sprintf("LightGBM dataset created: %d rows, %d columns\n", nrow(df_ligh), ncol(df_ligh)))
cat(sprintf("Training samples: %d\n", sum(!is.na(df_ligh$Energy))))
cat(sprintf("Test samples: %d\n", sum(is.na(df_ligh$Energy))))
