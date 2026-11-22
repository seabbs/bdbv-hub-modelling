# The delay's epinowcast parameters (`refp_mean_int`, `refp_sd_int`), its
# 99th centile and its probabilities on 0, ..., max_delay - 1 days:
# double interval censored (primarycensored, a uniform primary day) and
# truncated at max_delay, as epinowcast discretises it.
# Lognormal is meanlog and sdlog; gamma is shape exp(mean_int) and rate
# sd_int; the exponential rate is exp(-mean_int). With no parametric
# delay ("none"), the hazard's intercept is the constant hazard of a
# geometric delay with the drawn mean, already on whole days.
delay_parameters <- function(family, v, max_delay = NULL) {
  out <- switch(family,
    lognormal = list(
      mean_int = log(v$delay_median), sd_int = v$delay_sdlog,
      q99 = qlnorm(0.99, log(v$delay_median), v$delay_sdlog),
      pdist = list(plnorm, meanlog = log(v$delay_median),
        sdlog = v$delay_sdlog)
    ),
    gamma = {
      shape <- 1 / v$delay_cv^2
      rate <- shape / v$delay_mean
      list(
        mean_int = log(shape), sd_int = rate,
        q99 = qgamma(0.99, shape, rate),
        pdist = list(pgamma, shape = shape, rate = rate)
      )
    },
    exponential = list(
      mean_int = log(v$delay_mean),
      q99 = qexp(0.99, 1 / v$delay_mean),
      pdist = list(pexp, rate = 1 / v$delay_mean)
    ),
    none = {
      hazard <- 1 / (1 + v$delay_mean)
      list(hazard = hazard, q99 = qgeom(0.99, hazard))
    }
  )
  if (!is.null(max_delay)) {
    out$pmf <- if (family == "none") {
      pmf <- diff(pgeom(-1:(max_delay - 1), out$hazard))
      pmf / sum(pmf)
    } else {
      do.call(primarycensored::dpcens, c(
        list(0:(max_delay - 1), out$pdist[[1]], D = max_delay),
        out$pdist[-1]
      ))
    }
  }
  out
}

# The longest delay in days: the first of `options` (the grid's
# `max_delay_days`) at or above the delay's 99th centile, else the last.
max_delay_for <- function(q99, options) {
  options <- as.integer(options)
  options[min(which(c(options >= q99, TRUE))[1], length(options))]
}
