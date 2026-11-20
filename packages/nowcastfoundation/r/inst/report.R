# The simulator's epinowcast reporting step, called by
# nowcastfoundation.simulator (`RscriptReporting`) once per pass of a batch:
#
#   Rscript report.R <engine_dir> <job_dir>
#
# `job_dir/jobs.json` holds the grid file, the draws per simulation and
# the jobs (id, R seed, design, values with the latent step's additions,
# and each stratum's expected onsets, days by strata). Each job runs
# through `report_outbreak`; the step writes `counts.csv.gz` (id, draw,
# location, reference_date, report_date, count), `results.json` (each
# job's values, or its error) and `stack.json` (R, BLAS, LAPACK and the
# epinowcast fingerprint). ENW_LIB is the pinned epinowcast's library and
# ENW_CACHE its model cache; ENW_DEBUG shows Stan's messages.
args <- commandArgs(trailingOnly = TRUE)
engine <- args[[1]]
job_dir <- args[[2]]
lib <- Sys.getenv("ENW_LIB", "output/finetune/rlib")
.libPaths(c(lib, .libPaths()))
suppressPackageStartupMessages({
  library(epinowcast, lib.loc = lib)
  library(data.table)
})
pkgload::load_all(engine, quiet = TRUE)
options(cmdstanr_warn_inits = FALSE, mc.cores = 1)
enw_set_cache(
  Sys.getenv("ENW_CACHE", "output/finetune/enwcache"), type = "session"
)
input <- jsonlite::read_json(file.path(job_dir, "jobs.json"))
grid <- read_grid(input$grid_file)
debug <- nzchar(Sys.getenv("ENW_DEBUG"))
ctx <- list(
  model = suppressMessages(enw_model(threads = FALSE)),
  draws = as.integer(input$draws), debug = debug,
  expected_scale = grid$constants$expected_scale
)
# JSON arrays as R vectors: an empty array is a zero-length vector, a
# one-element array a scalar.
as_r <- function(x) {
  if (!is.list(x)) return(x)
  if (!length(x)) return(numeric(0))
  if (!is.null(names(x))) return(lapply(x, as_r))
  if (all(vapply(x, function(e) !is.list(e) && length(e) == 1L, TRUE))) {
    return(unlist(x))
  }
  lapply(x, as_r)
}
counts <- list()
results <- list()
for (job in input$jobs) {
  design <- as_r(job$design)
  v <- as_r(job$values)
  onsets <- do.call(rbind, lapply(job$onsets, function(r) {
    as.numeric(unlist(r))
  }))
  set.seed(job$seed)
  res <- tryCatch(
    {
      frame <- reporting_frame(design, v)
      # The onsets' columns must be epinowcast's strata, in its order.
      stopifnot(
        identical(unlist(job$places), frame$strata$place),
        identical(unlist(job$ages), frame$strata$age)
      )
      report_outbreak(grid, design, v, onsets, ctx, frame)
    },
    error = function(e) e
  )
  if (inherits(res, "error")) {
    results[[length(results) + 1L]] <- list(
      id = job$id, error = conditionMessage(res)
    )
    next
  }
  counts[[length(counts) + 1L]] <- res$counts[, .(
    id = job$id, draw, location, reference_date, report_date, count
  )]
  results[[length(results) + 1L]] <- list(id = job$id, values = res$values)
}
out <- if (length(counts)) {
  rbindlist(counts)
} else {
  data.table(
    id = character(), draw = integer(), location = character(),
    reference_date = character(), report_date = character(),
    count = integer()
  )
}
fwrite(out, file.path(job_dir, "counts.csv.gz"))
jsonlite::write_json(
  results, file.path(job_dir, "results.json"),
  auto_unbox = TRUE, digits = NA, null = "null", na = "null"
)
source("scripts/enw_fingerprint.R", local = TRUE)
info <- sessionInfo()
jsonlite::write_json(list(
  r_version = R.version.string, blas = info$BLAS, lapack = info$LAPACK,
  enw_fingerprint = enw_fingerprint(lib)
), file.path(job_dir, "stack.json"), auto_unbox = TRUE)
