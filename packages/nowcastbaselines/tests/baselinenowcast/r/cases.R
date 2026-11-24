# The pinned baselinenowcast on the test cases, for the live checks in
# packages/nowcastbaselines/tests/baselinenowcast (through
# tests/model/r_live.py).
#
# `bnc_run(input)` returns, for one case, the named matrices the tests
# compare with (vectors as one-column matrices, NA where unobserved).
# Triangles come from the tests (toy, neg, zeros), from the package
# (example) or from a seed here (struct2, hub, recon).

pinned <- c(version = "0.2.2", sha = "999d1ad50e27921a4078175d6315bbd655878888")

suppressPackageStartupMessages({
  library(baselinenowcast)
  library(zoo)
})

found <- c(
  as.character(packageVersion("baselinenowcast")),
  c(packageDescription("baselinenowcast")$RemoteSha, "")[1]
)
if (!identical(unname(found), unname(pinned))) {
  stop("baselinenowcast ", toString(pinned), " is required, found ",
    toString(found))
}

as_matrix <- function(m) {
  m <- unclass(m)
  if (is.null(dim(m))) m <- matrix(m, ncol = 1)
  dimnames(m) <- NULL
  m
}

as_rt <- function(m) {
  m <- as_matrix(m)
  as_reporting_triangle(
    m,
    delays_unit = "days",
    reference_dates = as.Date("2026-01-01") + seq_len(nrow(m)) - 1
  )
}

roll <- function(k) function(x) zoo::rollsum(x, k = k, align = "right")

# The counts in the first half of the delays (tests' first_delays).
first_delays <- function(x) {
  rowSums(x[, seq_len(ceiling(ncol(x) / 2)), drop = FALSE], na.rm = TRUE)
}

quiet <- function(expr) suppressWarnings(suppressMessages(expr))

bnc_triangle <- function(input) {
  switch(input$case,
    example = as_matrix(baselinenowcast::example_reporting_triangle),
    struct2 = {
      set.seed(20260101)
      lambda <- rep(c(40, 20, 10, 6, 3, 1), each = 12)
      full <- matrix(rpois(12 * 6, lambda = lambda), 12, 6)
      as_matrix(apply_reporting_structure(as_rt(full), structure = 2))
    },
    hub = {
      set.seed(20260102)
      n_ref <- 262
      max_delay <- 84
      pmf <- diff(pgamma(0:(max_delay + 1), shape = 2.2, rate = 0.12))
      pmf <- pmf / sum(pmf)
      level <- 80 * exp(0.012 * seq_len(n_ref)) *
        (1 + 0.3 * sin(seq_len(n_ref) / 9))
      hub <- t(sapply(seq_len(n_ref), function(i) {
        rpois(max_delay + 1, level[i] * pmf)
      }))
      hub <- matrix(as.numeric(hub), n_ref, max_delay + 1)
      hub[3, 5] <- -2
      hub[40, 30] <- -3
      hub[40, 31] <- 1
      as_matrix(apply_reporting_structure(as_rt(hub), structure = 1))
    },
    input$tri
  )
}

# The retrospective observations and predictions the uncertainty model
# is fitted to, and the fitted sizes.
bnc_retro <- function(pre, n_hist, n_retro, ...) {
  cap <- NULL
  capture <- function(obs, pred) {
    cap <<- list(obs = obs, pred = pred)
    fit_by_horizon(obs, pred)
  }
  size <- quiet(estimate_uncertainty_retro(
    pre,
    n_history_delay = n_hist, n_retrospective_nowcasts = n_retro,
    uncertainty_model = capture, ...
  ))
  list(obs = cap$obs, pred = cap$pred, size = size)
}

# Mean and variance of the sampled nowcast at each reference time.
bnc_samples <- function(tri, draws, ...) {
  set.seed(1)
  smp <- quiet(baselinenowcast(
    tri, output_type = "samples", draws = draws,
    scale_factor = 3, prop_delay = 0.5, ...
  ))
  by_ref <- split(smp$pred_count, smp$reference_date)
  cbind(
    vapply(by_ref, mean, numeric(1)),
    vapply(by_ref, var, numeric(1))
  )
}

# Reference-time and delay aggregation: none, a rolling sum of `k`
# reference times, and that with the counts in the first delays.
bnc_modes <- function(k) {
  list(
    id = list(),
    agg = list(ref_time_aggregator = roll(k)),
    week = list(ref_time_aggregator = roll(k), delay_aggregator = first_delays)
  )
}

bnc_case <- function(input) {
  m <- bnc_triangle(input)
  tri <- quiet(as_rt(m))
  modes <- bnc_modes(input$k)
  if (isTRUE(input$samples)) {
    return(lapply(
      setNames(names(modes), paste0("samples_", names(modes))),
      function(mode) {
        do.call(bnc_samples, c(list(tri, input$draws), modes[[mode]]))
      }
    ))
  }
  n_hist <- input$n_hist
  n_retro <- input$n_retro
  out <- list(tri = m)

  pre <- quiet(preprocess_negative_values(tri))
  out$preprocess <- pre
  tr <- quiet(truncate_to_rows(tri, n = 3))
  for (i in seq_along(tr)) out[[paste0("trunc_", i)]] <- tr[[i]]
  st <- quiet(get_reporting_structure(tri))
  out$structure <- st
  out$applied_1 <- quiet(apply_reporting_structure(tr[[1]], structure = st))
  out$applied_struct1 <- quiet(
    apply_reporting_structure(tr[[2]], structure = 1)
  )

  grid <- expand.grid(
    scale_factor = c(3, 1, 2, 0.5),
    prop_delay = c(0.5, 0.3, 0.8),
    n_min = c(2, 0)
  )
  out$alloc <- t(vapply(seq_len(nrow(grid)), function(i) {
    a <- tryCatch(
      quiet(allocate_reference_times(
        tri,
        scale_factor = grid$scale_factor[i],
        prop_delay = grid$prop_delay[i],
        n_min_retro_nowcasts = grid$n_min[i]
      )),
      error = function(e) {
        list(n_history_delay = NA, n_retrospective_nowcasts = NA)
      }
    )
    c(grid$scale_factor[i], grid$prop_delay[i], grid$n_min[i],
      a$n_history_delay, a$n_retrospective_nowcasts)
  }, numeric(5)))

  out$pmf_raw <- quiet(estimate_delay(tri, n = n_hist))
  pmf <- quiet(estimate_delay(pre, n = n_hist))
  out$pmf <- pmf
  out$pmf_all <- quiet(estimate_delay(pre, n = nrow(pre)))
  out$apply <- quiet(apply_delay(pre, pmf))
  out$eaa <- quiet(estimate_and_apply_delay(pre, n = n_hist))
  out$point <- quiet(baselinenowcast(
    tri, output_type = "point", scale_factor = 3, prop_delay = 0.5
  ))$pred_count

  for (mode in names(modes)) {
    r <- do.call(bnc_retro, c(list(pre, n_hist, n_retro), modes[[mode]]))
    out[[paste0("retro_obs_", mode)]] <- r$obs
    out[[paste0("retro_pred_", mode)]] <- r$pred
    out[[paste0("size_", mode)]] <- r$size
  }
  out
}

bnc_zeros <- function(input) {
  list(apply = quiet(apply_delay(as_rt(input$tri), input$pmf)))
}

bnc_nb <- function(input) {
  out <- list()
  for (i in seq_along(input$x)) {
    out[[paste0("fit_out_", i)]] <- fit_nb(input$x[[i]], input$mu_fit[[i]])
  }
  # Moments of rnbinom for the sampler check.
  set.seed(2)
  out$sample_mom <- t(vapply(seq_along(input$mu), function(i) {
    x <- rnbinom(input$n, size = input$size[i], mu = input$mu[i])
    c(mean(x), var(x))
  }, numeric(2)))
  out
}

# A right-truncated daily triangle, its pmf from estimate_delay and the
# row totals of the chain ladder fill of its last `n_use` rows.
bnc_recon <- function(input) {
  set.seed(20261102)
  n_ref <- input$n_ref
  max_delay <- input$max_delay
  n_use <- input$n_use
  pmf <- diff(pgamma(0:(max_delay + 1), shape = 2, rate = 0.5))
  pmf <- pmf / sum(pmf)
  level <- 40 + 2 * seq_len(n_ref)
  tri <- t(sapply(seq_len(n_ref), function(i) {
    rpois(max_delay + 1, level[i] * pmf)
  }))
  for (i in seq_len(n_ref)) {
    for (d in 0:max_delay) if (i - 1 + d > n_ref - 1) tri[i, d + 1] <- NA
  }
  last <- tri[(n_ref - n_use + 1):n_ref, , drop = FALSE]
  filled <- baselinenowcast:::.chainladder_fill_triangle(last)
  list(
    tri = tri,
    pmf = quiet(estimate_delay(as_rt(tri), n = n_use)),
    row_totals = rowSums(filled)
  )
}

bnc_run <- function(input) {
  out <- switch(input$case,
    zeros = bnc_zeros(input),
    nb = bnc_nb(input),
    recon = bnc_recon(input),
    bnc_case(input)
  )
  lapply(out, as_matrix)
}
