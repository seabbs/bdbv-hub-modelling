# Weekly AR(1) noise (autocorrelation 0.8) on `n` days.
weekly_noise <- function(n, sd) {
  weeks <- ceiling(n / 7)
  e <- numeric(weeks)
  for (w in seq_len(weeks)[-1]) e[w] <- 0.8 * e[w - 1] + rnorm(1, 0, sd)
  rep(e, each = 7)[seq_len(n)]
}
