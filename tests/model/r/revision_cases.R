# Revisions of reported counts: a copy of the simulator R engine's
# `revise_counts` (packages/nowcastfoundation/r/R/reporting.R), and its
# random draws, for tests/model/test_revisions.py.

suppressMessages(library(data.table))

engine_revise_counts <- function(pp, v) {
  last <- max(pp$report_date)
  first <- min(pp$reference_date)
  group <- if (".group" %in% names(pp)) pp[mean > 0, .group] else NULL
  base <- pp[mean > 0, .(draw, location, reference_date, report_date, mean)]
  n <- nrow(base)
  rate <- function(x) if (is.null(group)) rep(x[1], n) else x[group]
  keep <- pmax(1L, as.integer(round(rate(v$revise_keep))))
  later <- pmin(base$report_date + keep, last)
  err <- rpois(n, rate(v$revise_over) * base$mean)
  moved <- rbinom(n, base$mean, rate(v$revise_move))
  shift <- (2L * rbinom(n, 1, 0.5) - 1L) * (1L + rpois(n, 1))
  to <- pmin(pmax(base$reference_date + shift, first), base$report_date)
  at <- function(ref, rep, x) {
    data.table(
      draw = base$draw, location = base$location, reference_date = ref,
      report_date = rep, mean = x
    )
  }
  ref <- base$reference_date
  rep <- base$report_date
  out <- rbind(
    base,
    at(ref, rep, err), at(ref, later, -err),
    at(ref, rep, -moved), at(to, rep, moved),
    at(ref, later, moved), at(to, later, -moved)
  )
  out <- out[, .(mean = sum(mean)),
    by = .(draw, location, reference_date, report_date)
  ]
  out[mean != 0]
}

# Cells (`g`, `t`, `d`, 0-based, and `count`) revised with the engine's
# copy at `seed`; the draws it made, in its order, come back too.
revision_case <- function(input) {
  cells <- as.data.table(input$cells)
  v <- input$values
  d0 <- as.IDate("2026-01-03")
  pp <- cells[, .(
    draw = 1L, location = paste0("S", g + 1L), .group = g + 1L,
    reference_date = d0 + t, report_date = d0 + t + d, mean = count
  )]
  base <- pp[mean > 0]
  n <- nrow(base)
  rate <- function(x) x[base$.group]
  set.seed(input$seed)
  errors <- rpois(n, rate(v$revise_over) * base$mean)
  moved <- rbinom(n, base$mean, rate(v$revise_move))
  later <- rbinom(n, 1, 0.5)
  extra <- rpois(n, 1)
  set.seed(input$seed)
  out <- engine_revise_counts(pp, v)
  list(
    draws = list(errors = errors, moved = moved, later = later, extra = extra),
    out = out[, .(
      g = as.integer(sub("S", "", location)) - 1L,
      t = as.integer(reference_date - d0), r = as.integer(report_date - d0),
      count = mean
    )]
  )
}
