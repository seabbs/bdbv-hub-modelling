# The infection-to-reference-event delay (incubation, or infection to
# test or admission) on 0, 1, ... days: a gamma or log-normal with CV `cv`
# and mean `mean`, double interval censored (primarycensored), to its
# 99.9% quantile.
incubation_pmf <- function(mean, cv = 0.5, family = "gamma") {
  if (identical(family, "lognormal")) {
    sdlog <- sqrt(log1p(cv^2))
    meanlog <- log(mean) - sdlog^2 / 2
    n <- max(1L, as.integer(ceiling(qlnorm(0.999, meanlog, sdlog))))
    return(primarycensored::dpcens(0:n, plnorm, D = n + 1,
      meanlog = meanlog, sdlog = sdlog))
  }
  shape <- 1 / cv^2
  n <- max(1L, as.integer(ceiling(qgamma(0.999, shape, shape / mean))))
  primarycensored::dpcens(0:n, pgamma, D = n + 1, shape = shape,
    rate = shape / mean)
}

# Each column of `x` (days by strata) convolved with its own pmf in
# `pmfs` on delays 0, 1, ... `before` is each column's constant value on
# the days before the first (0 if NULL), so the first rows hold the delayed
# part of that history too.
convolve_rows <- function(x, pmfs, before = NULL) {
  out <- matrix(0, nrow(x), ncol(x))
  for (j in seq_len(ncol(x))) {
    p <- pmfs[[j]]
    for (d in seq_along(p) - 1L) {
      if (d >= nrow(x)) break
      rows <- (d + 1L):nrow(x)
      out[rows, j] <- out[rows, j] + p[d + 1L] * x[rows - d, j]
    }
    if (!is.null(before) && before[j] != 0) {
      # Day t gets the history's infections d > t - 1 days before it.
      tail_mass <- rev(cumsum(rev(p)))[-1L]
      k <- seq_len(min(length(tail_mass), nrow(x)))
      out[k, j] <- out[k, j] + before[j] * tail_mass[k]
    }
  }
  out
}
