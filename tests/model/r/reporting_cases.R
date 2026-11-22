# Reference-date effects of a simulation's reporting delay through the
# pinned epinowcast's own model, for tests/model/test_reporting_effects.py.
#
# For a design and values (as the simulator draws them), builds
# epinowcast's reference module with the simulator engine's formulas and
# the engine's parameter values (random effects as standard normal draws,
# change terms fixed), runs epinowcast's Stan model with those fixed
# parameters and returns its transformed parameters (`refp_mean`,
# `refp_sd`, `refp_lh`, `refnp_lh`), with each design's column names and
# the parameters, so the Python port can take the same values.
#
# The covariates and formulas copy the engine's
# (`packages/nowcastfoundation/r/R/simulate.R`), with one change: the
# weekday effect within place is one factor (`dow_place`), as
# `(1 | day_of_week:place)` cannot run at the pin.

suppressMessages(library(data.table))

reporting_daily_walk <- function(days, walk) {
  x <- days / 7
  w <- floor(x)
  f <- x - w
  n <- length(walk)
  walk[pmin(w + 1, n)] * (1 - f) + walk[pmin(w + 2, n)] * f
}

reporting_window <- function(first, at, weeks, v) {
  start <- first + round(at * v$weeks * 7)
  c(start, start + 7L * weeks - 1L)
}

reporting_formula <- function(intercept, terms) {
  terms <- c(if (intercept) "1" else "0", terms)
  as.formula(paste("~", paste(terms, collapse = " + ")))
}

reporting_effect_sds <- function(rdesign, sds) {
  effects <- colnames(rdesign)[-1]
  array(as.numeric(unlist(sds[effects])))
}

reference_case <- function(input) {
  design <- input$design
  v <- input$values
  set.seed(input$seed)
  D <- as.integer(input$max_delay)
  places <- sprintf("P%02d", seq_len(input$n_places))
  ages <- sprintf("A%d", seq_len(input$n_ages))
  d0 <- as.IDate(input$first_date)
  obs <- CJ(
    place = places, age = ages, reference_date = d0 + 0:(input$n_days - 1),
    delay = 0:(D - 1)
  )
  obs[, report_date := reference_date + delay][, delay := NULL]
  obs <- obs[report_date <= max(reference_date)][, confirm := 0L]
  pobs <- suppressMessages(suppressWarnings(epinowcast::enw_preprocess_data(
    obs, by = c("place", "age"), max_delay = D
  )))
  meta <- pobs$metareference[[1]]
  n_weeks <- max(meta$week)
  day <- as.integer(meta$date - min(meta$date))
  set(meta, j = "trend", value = meta$week / n_weeks - 0.5)
  set(meta, j = "step", value = as.numeric(meta$week >= v$step_at * n_weeks))
  set(meta, j = "periodic", value = sin(2 * pi * meta$week / v$periodic_weeks))
  set(meta, j = "walk_daily", value = reporting_daily_walk(day, v$delay_walk))
  set(meta, j = "smooth", value = sin(
    2 * pi * day / v$smooth_period + v$smooth_phase
  ))
  backlog <- numeric(nrow(meta))
  for (k in seq_along(v$shock_places)) {
    w <- reporting_window(min(meta$date), v$shock_at[k], v$shock_weeks[k], v)
    on <- meta$place == sprintf("P%02d", v$shock_places[k]) &
      meta$date >= w[1] & meta$date <= w[2]
    backlog[on] <- backlog[on] + log(v$shock_mult[k])
  }
  set(meta, j = "backlog", value = backlog)
  slow <- reporting_window(min(meta$date), v$slow_at, v$slow_weeks, v)
  set(meta, j = "slowdown", value = as.numeric(
    meta$date >= slow[1] & meta$date <= slow[2]
  ))
  set(meta, j = "dow_place", value = paste(meta$day_of_week, meta$place, sep = "_"))
  multi_place <- input$n_places > 1
  multi_age <- input$n_ages > 1
  parametric <- design$delay_form != "non_parametric"
  hazard <- design$delay_form != "parametric"
  change_terms <- switch(design$delay_change,
    static = NULL, drift = "trend", step = "step", periodic = "periodic",
    walk = "walk_daily", smooth = "smooth", slowdown = "slowdown"
  )
  change_beta <- switch(design$delay_change,
    drift = v$drift, step = v$step, periodic = v$periodic, 0
  )
  if (design$shocks == "backlog") change_terms <- c(change_terms, "backlog")
  stratum_terms <- c(
    if (multi_place && design$delay_places != "shared") "(1 | place)",
    if (multi_age) "(1 | age)"
  )
  if (design$delay_dow == "yes") {
    stratum_terms <- c(stratum_terms, if (
      multi_place && design$delay_places == "independent"
    ) "(1 | dow_place)" else "(1 | day_of_week)")
  }
  family <- if (parametric) design$delay_family else "none"
  hazard_terms <- c("rw(delay)", "(1 | delay)")
  reference <- epinowcast::enw_reference(
    parametric = if (parametric) {
      reporting_formula(TRUE, c(change_terms, stratum_terms))
    } else {
      ~0
    },
    distribution = family,
    non_parametric = if (!hazard) {
      ~0
    } else if (parametric) {
      reporting_formula(FALSE, hazard_terms)
    } else {
      reporting_formula(TRUE, c(hazard_terms, change_terms, stratum_terms))
    },
    data = pobs
  )
  rd <- reference$data
  stratum_sd <- list(
    place = v$delay_place_sd, age = v$delay_age_sd,
    day_of_week = v$delay_dow_sd, dow_place = v$delay_dow_sd
  )
  p <- list()
  set_change <- function(prefix, cols) {
    beta <- rnorm(length(cols))
    beta[cols %in% c("trend", "step", "periodic")] <- change_beta
    direction <- if (prefix == "refp_mean") 1 else -1
    beta[cols == "backlog"] <- direction
    beta[cols == "slowdown"] <- log(v$slow_mult) * direction
    beta[cols == "walk_daily"] <- direction
    beta[cols == "smooth"] <- v$smooth_amp * direction
    p[[paste0(prefix, "_beta")]] <<- array(beta)
  }
  if (parametric) {
    fcols <- colnames(rd$refp_fdesign)
    p$refp_mean_int <- input$mean_int
    set_change("refp_mean", fcols)
    sds <- reporting_effect_sds(rd$refp_rdesign, stratum_sd)
    if (length(sds)) {
      p$refp_mean_beta_sd <- sds
      p$refp_sd_beta_sd <- sds / 2
    }
    if (family != "exponential") {
      p$refp_sd_int <- input$sd_int
      p$refp_sd_beta <- array(numeric(length(fcols)))
    }
  }
  if (hazard) {
    set_change("refnp", colnames(rd$refnp_fdesign))
    p$refnp_beta_sd <- reporting_effect_sds(
      rd$refnp_rdesign, c(list(delay = v$hazard_sd), stratum_sd)
    )
    if (!parametric) p$refnp_int <- qlogis(1 / (1 + v$delay_mean))
    p$refnp_arima_z <- matrix(
      rnorm(rd$refnp_arima_T * rd$refnp_arima_G),
      rd$refnp_arima_T, rd$refnp_arima_G
    )
    p$refnp_arima_sigma <- array(v$hazard_walk_sd)
  }
  lambda <- matrix(5, input$n_days, input$n_places * input$n_ages)
  expectation <- epinowcast::enw_expectation(
    r = ~1, generation_time = 1, observation = ~1, data = pobs
  )
  log_l <- log(lambda)
  p$expr_lelatent_int <- matrix(log_l[1, ], nrow = 1)
  capture <- function(data, init, ...) data.table(data = list(data), init = list(init))
  prepared <- epinowcast::epinowcast(
    data = pobs, reference = reference,
    report = epinowcast::enw_report(data = pobs), expectation = expectation,
    missing = epinowcast::enw_missing(formula = ~0, data = pobs),
    obs = epinowcast::enw_obs(family = "poisson", data = pobs),
    fit = epinowcast::enw_fit_opts(
      sampler = capture, nowcast = TRUE, pp = TRUE, likelihood = FALSE
    ),
    model = NULL
  )
  data <- epinowcast::enw_get_data(prepared, "data")
  data$expr_r_override <- 1L
  data$expr_r_override_value <- as.numeric(apply(log_l, 2, diff))
  init <- utils::modifyList(epinowcast::enw_get_data(prepared, "init")(), p)
  init <- epinowcast:::.map_centred_intercepts(init)
  model <- epinowcast::enw_model(threads = FALSE, verbose = FALSE)
  # Each variable in the model's declared shape and type (as the engine's
  # `write_stan_data`): integer data held as doubles in R are rounded.
  for (name in names(model$variables()$data)) {
    decl <- model$variables()$data[[name]]
    x <- data[[name]]
    if (length(x) == 1 && decl$dimensions == 1) x <- array(x, dim = 1)
    if (decl$type == "int" && !is.integer(x)) {
      x <- round(x)
      mode(x) <- "integer"
    }
    data[[name]] <- x
  }
  file <- tempfile(fileext = ".json")
  cmdstanr::write_stan_json(data, file, always_decimal = TRUE)
  fit <- suppressMessages(suppressWarnings(model$sample(
    data = file, fixed_param = TRUE, chains = 1, iter_sampling = 1,
    iter_warmup = 0, seed = 1, init = function() init, refresh = 0,
    sig_figs = 18,
    show_messages = FALSE, show_exceptions = FALSE
  )))
  draw <- function(name) {
    d <- fit$draws(name, format = "draws_matrix")
    if (ncol(d) == 0) NULL else as.numeric(d[1, ])
  }
  list(
    groups = meta[, .(.group, place, age)][!duplicated(.group)],
    dates = as.character(sort(unique(meta$date))),
    refp_cols = if (parametric) colnames(rd$refp_fdesign) else NULL,
    refnp_cols = if (hazard) colnames(rd$refnp_fdesign) else NULL,
    refp_findex = rd$refp_findex,
    # epinowcast's flag: with a parametric delay and no report-date model,
    # `refp_lh` holds log probabilities, not logit hazards.
    ref_as_p = as.integer(data$model_rep == 0 && data$model_refp > 0),
    parameters = p,
    refp_mean = if (parametric) draw("refp_mean") else NULL,
    refp_sd = if (parametric) draw("refp_sd") else NULL,
    # refp_lh is (D, refp_fnrow), column-major as a draws vector: logit
    # hazards, or log probabilities where `ref_as_p`.
    refp_lh = if (parametric) draw("refp_lh") else NULL,
    refnp_lh = if (hazard) draw("refnp_lh") else NULL
  )
}
