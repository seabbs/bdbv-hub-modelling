# The reporting delay's location and scale (`mean_int`, `sd_int`, as
# epinowcast's) and its 99th centile, which sets the longest delay
# (`max_delay_for`). Lognormal is meanlog and sdlog; gamma is shape
# exp(mean_int) and rate sd_int; the exponential rate is exp(-mean_int).
# With no parametric delay ("none"), the constant hazard of a geometric
# delay with the drawn mean.
delay_parameters <- function(family, v) {
  out <- switch(family,
    lognormal = list(
      mean_int = log(v$delay_median), sd_int = v$delay_sdlog,
      q99 = qlnorm(0.99, log(v$delay_median), v$delay_sdlog)
    ),
    gamma = {
      shape <- 1 / v$delay_cv^2
      rate <- shape / v$delay_mean
      list(
        mean_int = log(shape), sd_int = rate,
        q99 = qgamma(0.99, shape, rate)
      )
    },
    exponential = list(
      mean_int = log(v$delay_mean),
      q99 = qexp(0.99, 1 / v$delay_mean)
    ),
    none = {
      hazard <- 1 / (1 + v$delay_mean)
      list(hazard = hazard, q99 = qgeom(0.99, hazard))
    }
  )
  out
}

# The longest delay in days: the first of `options` (the grid's
# `max_delay_days`) at or above the delay's 99th centile, else the last.
max_delay_for <- function(q99, options) {
  options <- as.integer(options)
  options[min(which(c(options >= q99, TRUE))[1], length(options))]
}
