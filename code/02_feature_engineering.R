# ============================================================================
# 02_FEATURE_ENGINEERING.R - Feature Engineering Pipeline
# ============================================================================
# Description: Creates engineered features for both base-level and cell-level
#              datasets including time features, domain-specific indicators,
#              and statistical transformations.
# ============================================================================

source("00_config.R")

# Load cleaned data if not in memory
if (!exists("base")) {
  load("data_base_cleaned.RData")
}

# ============================================================================
# 1. HELPER FUNCTIONS FOR ENCODING
# ============================================================================

#' Helmert Contrast Encoding
#' @param n Number of levels
helmert_matrix <- function(n) {
  m <- t((diag(seq(n - 1, 0)) - upper.tri(matrix(1, n, n)))[-n, ])
  t(apply(m, 1, rev))
}

#' Encode categorical variable using Helmert contrasts
#' @param df Data frame
#' @param var Variable name to encode
encode_helmert <- function(df, var) {
  x <- unique(df[[var]])
  n <- length(x)
  d <- as.data.frame(helmert_matrix(n))
  d[[var]] <- rev(x)
  names(d) <- c(paste0(var, 1:(n - 1)), var)
  return(d)
}

#' Polynomial Contrast Encoding
#' @param df Data frame
#' @param var Variable name to encode
encode_polynomial <- function(df, var) {
  x <- unique(df[[var]])
  n <- length(x)
  d <- as.data.frame(contr.poly(n))
  d[[var]] <- x
  names(d) <- c(paste0(var, 1:(n - 1)), var)
  return(d)
}

#' Backward Difference Encoding Matrix
#' @param n Number of levels
backward_difference_matrix <- function(n) {
  m <- matrix(0:(n - 1), nrow = n, ncol = n)
  m <- m + upper.tri(matrix(1, n, n))
  m2 <- matrix(-(n - 1), n, n)
  m2[upper.tri(m2)] <- 0
  m <- (m + m2) / n
  m <- (t(m))[, -n]
  return(m)
}

#' Encode categorical variable using backward difference contrasts
#' @param df Data frame
#' @param var Variable name to encode
encode_backward_difference <- function(df, var) {
  x <- unique(df[[var]])
  n <- length(x)
  d <- as.data.frame(backward_difference_matrix(n))
  d[[var]] <- x
  names(d) <- c(paste0(var, 1:(n - 1)), var)
  return(d)
}

#' Target Encoding (Mean Encoding)
#' @param x Categorical variable
#' @param y Target variable
#' @param sigma Optional noise for regularization
encode_target <- function(x, y, sigma = NULL) {
  d <- aggregate(y, list(factor(x, exclude = NULL)), mean, na.rm = TRUE)
  m <- d[is.na(as.character(d[, 1])), 2]
  l <- d[, 2]
  names(l) <- d[, 1]
  l <- l[x]
  l[is.na(l)] <- m

  if (!is.null(sigma)) {
    l <- l * rnorm(length(l), mean = 1, sd = sigma)
  }
  return(l)
}

#' M-Estimator Encoding (Smoothed Target Encoding)
#' @param x Categorical variable
#' @param y Target variable
#' @param m Smoothing parameter
#' @param sigma Optional noise for regularization
encode_m_estimator <- function(x, y, m = 1, sigma = NULL) {
  p_all <- mean(y, na.rm = TRUE)

  d <- aggregate(y, list(factor(x, exclude = NULL)), sum, na.rm = TRUE)
  d2 <- aggregate(y, list(factor(x, exclude = NULL)), length)
  g <- names(d)[1]
  d <- merge(d, d2, by = g, all = TRUE)
  d[, 4] <- (d[, 2] + p_all * m) / (d[, 3] + m)

  m_default <- d[is.na(as.character(d[, 1])), 4]
  l <- d[, 4]
  names(l) <- d[, 1]
  l <- l[x]
  l[is.na(l)] <- m_default

  if (!is.null(sigma)) {
    l <- l * rnorm(length(l), mean = 1, sd = sigma)
  }

  list(
    encoded = l,
    encode_new = function(new_x) {
      new_l <- l[new_x]
      new_l[is.na(new_l)] <- m_default
      if (!is.null(sigma)) {
        new_l <- new_l * rnorm(length(new_l), mean = 1, sd = sigma)
      }
      return(new_l)
    }
  )
}

#' Weight of Evidence Encoding
#' @param x Categorical variable
#' @param y Target variable
#' @param sigma Optional noise for regularization
encode_woe <- function(x, y, sigma = NULL) {
  d <- aggregate(y, list(factor(x, exclude = NULL)), mean, na.rm = TRUE)
  d[["woe"]] <- log(((1 / d[, 2]) - 1) * (sum(y, na.rm = TRUE) / sum(1 - y, na.rm = TRUE)))

  m <- d[is.na(as.character(d[, 1])), 3]
  l <- d[, 3]
  names(l) <- d[, 1]
  l <- l[x]
  l[is.na(l)] <- m

  if (!is.null(sigma)) {
    l <- l * rnorm(length(l), mean = 1, sd = sigma)
  }

  list(
    encoded = l,
    encode_new = function(new_x) {
      new_l <- l[new_x]
      new_l[is.na(new_l)] <- m
      if (!is.null(sigma)) {
        new_l <- new_l * rnorm(length(new_l), mean = 1, sd = sigma)
      }
      return(new_l)
    }
  )
}

# ============================================================================
# 2. DOMAIN-SPECIFIC FEATURE FUNCTIONS
# ============================================================================

#' Create time-based features
#' @param df Data frame with Time column
#' @param time_col Name of time column
create_time_features <- function(df, time_col = "Time") {
  df$Time <- strptime(df[[time_col]], format = "%m/%d/%Y %H:%M")
  df$hour <- as.numeric(format(df$Time, format = "%H"))
  df$day_of_week <- as.integer(format(df$Time, format = "%w"))
  df$day_of_week <- ifelse(df$day_of_week == 0, 'Sun',
                           ifelse(df$day_of_week == 6, 'Sat', 'week'))
  df$Time <- NULL
  return(df)
}

#' Create load-based hour indicator
#' @param hour Hour value (0-23)
create_load_hour <- function(hour) {
  ifelse(hour %in% c(3:7), 0, 1)
}

#' Create tree-based energy category (Factor version)
#' @param nb_cell Number of cells
#' @param load Load value
#' @param anten_cat Antenna category
create_tree_energy_F <- function(nb_cell, load, anten_cat) {
  case_when(
    nb_cell == 1 & load <= 0.08 ~ 'VL',
    (nb_cell == 1 & load >= 0.6) |
      (nb_cell == 2 & load <= 0.37 & anten_cat != 4) ~ 'M',
    (nb_cell == 2 & load > 0.37 & anten_cat != 4) |
      (nb_cell == 2 & load <= 0.37 & anten_cat == 4) ~ 'H',
    nb_cell == 2 & load > 0.37 & anten_cat == 4 ~ 'VH',
    TRUE ~ 'L'
  )
}

#' Create tree-based energy category (Numeric version)
#' @param nb_cell Number of cells
#' @param load Load value
#' @param anten_cat Antenna category
create_tree_energy_N <- function(nb_cell, load, anten_cat) {
  case_when(
    nb_cell == 1 & load <= 0.08 ~ 2,
    (nb_cell == 1 & load >= 0.6) |
      (nb_cell == 2 & load <= 0.37 & anten_cat != 4) ~ 3,
    (nb_cell == 2 & load > 0.37 & anten_cat != 4) |
      (nb_cell == 2 & load <= 0.37 & anten_cat == 4) ~ 4,
    nb_cell == 2 & load > 0.37 & anten_cat == 4 ~ 5,
    TRUE ~ 1
  )
}

#' Create ES1 low load indicator
create_es1_load_low <- function(ESMode1, load, anten_cat, nb_cell) {
  case_when(
    (ESMode1 >= 0.47 & load <= 0.044 & anten_cat %in% c(8, 32, 64)) |
      (nb_cell == 1 & load <= 0.5) |
      (nb_cell == 2 & ESMode1 >= 0.46 & load <= 0.044) ~ 1,
    TRUE ~ 0
  )
}

#' Create ES3 high load indicator
create_es3_load_high <- function(ESMode3, load) {
  ifelse(ESMode3 == 0 & load >= 0.055, 1, 0)
}

#' Create ES6 high load indicator
create_es6_load_high <- function(ESMode6, load) {
  ifelse(ESMode6 <= 0.13 & load >= 0.055, 1, 0)
}

#' Create frequency-cell-antenna based indicators
create_freq_cell_anten_low <- function(freq_max, nb_cell, anten_cat) {
  case_when(
    (freq_max == '426.98' & nb_cell == 1 & anten_cat == '1') |
      (freq_max == '697.002' & nb_cell == 2) ~ 1,
    TRUE ~ 0
  )
}

create_freq_cell_anten_high <- function(freq_max, nb_cell, anten_cat) {
  ifelse(freq_max == '426.98' & nb_cell == 2 & anten_cat == '4', 1, 0)
}

#' Create high/low use indicators
create_high_use <- function(RUType, anten_cat, load, load_hour, hour, es3_load_high) {
  case_when(
    (RUType == 'Type1' & anten_cat == 4) &
      ((load <= 0.4 & load_hour == 1) |
         (hour %in% c(13, 14, 15, 16, 17, 18, 22)) |
         (es3_load_high == 1)) ~ 1,
    TRUE ~ 0
  )
}

create_low_use <- function(ESMode1, load, freq_max, nb_cell, anten_cat) {
  case_when(
    (ESMode1 >= 0.7 & load >= 0.03) |
      (freq_max == '697.002' & nb_cell == 2) |
      (freq_max == '426.98' & nb_cell == 1 & anten_cat == '1') ~ 1,
    TRUE ~ 0
  )
}

#' Create type categorizations
create_type_anten <- function(RUType) {
  case_when(
    RUType %in% c('Type9', 'Type10', 'Type11', 'Type12') ~ 'anten_unique',
    RUType %in% c('Type1', 'Type2', 'Type3') ~ 'anten_mode2',
    RUType %in% c('Type4', 'Type5', 'Type6', 'Type7', 'Type8') ~ 'anten_two'
  )
}

create_type_use <- function(RUType) {
  case_when(
    RUType %in% c('Type1', 'Type10', 'Type11') ~ 'high',
    RUType %in% c('Type12', 'Type7', 'Type8') ~ 'medium',
    RUType %in% c('Type2', 'Type3', 'Type4', 'Type5', 'Type6', 'Type9') ~ 'low'
  )
}

#' Create hour category
create_hour_ch <- function(hour) {
  case_when(
    hour %in% c(2:6) ~ 'L',
    hour %in% c(0, 1, 7:11) ~ 'M',
    TRUE ~ 'H'
  )
}

#' Apply Box-Cox transformation safely
apply_boxcox <- function(x, y_for_fitting) {
  tryCatch({
    model <- boxcox(y_for_fitting ~ x, plotit = FALSE)
    lambda <- model$x[which.max(model$y)]

    if (abs(lambda) < .Machine$double.eps^0.25) {
      result <- log(x)
    } else {
      result <- (x^lambda - 1) / lambda
    }
    result[is.infinite(result) & result < 0] <- 0
    return(result)
  }, error = function(e) {
    return(x)
  })
}

# ============================================================================
# 3. FACTOR RECODING FUNCTIONS
# ============================================================================

#' Recode antenna categories
recode_anten_cat <- function(anten_cat) {
  anten_cat <- fct_recode(anten_cat, '2' = "8")
  anten_cat <- fct_recode(anten_cat, '32' = "64")
  return(anten_cat)
}

#' Recode RUType categories (merge similar types)
recode_rutype <- function(RUType) {
  RUType <- fct_recode(RUType, 'Type6' = "Type9")
  RUType <- fct_recode(RUType, 'Type7' = "Type8")
  RUType <- fct_recode(RUType, 'Type10' = "Type11")
  RUType <- fct_recode(RUType, 'Type10' = "Type12")
  RUType <- fct_recode(RUType, 'Type4' = "Type2")
  RUType <- fct_recode(RUType, 'Type4' = "Type6")
  return(RUType)
}

#' Recode frequency categories
recode_freq <- function(freq) {
  freq <- fct_recode(freq, '697.002' = "979.998")
  freq <- fct_recode(freq, '697.002' = "715.998")
  freq <- fct_recode(freq, '532' = "189")
  return(freq)
}

#' Recode bandwidth categories
recode_band <- function(band) {
  fct_recode(band, '10' = "8")
}

cat("Feature engineering functions loaded.\n")
