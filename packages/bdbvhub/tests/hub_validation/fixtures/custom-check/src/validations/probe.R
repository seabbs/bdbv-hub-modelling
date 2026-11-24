# A custom check in the hub's style (hub-config/validations.yml, as the
# hub's forecast_date check since 2026-11-16): it passes only when the
# validator copied this script into the staged hub and gave it the
# original checkout (FORECAST_SOURCE_HUB) holding the new file, apart from
# the staged hub, and a base commit there (PR_BASE_SHA).
check_probe <- function(hub_path, file_path) {
  source_hub <- Sys.getenv("FORECAST_SOURCE_HUB")
  base_sha <- Sys.getenv("PR_BASE_SHA")
  ok <- nzchar(source_hub) && nzchar(base_sha) &&
    normalizePath(source_hub) != normalizePath(hub_path) &&
    file.exists(file.path(source_hub, "model-output", file_path)) &&
    system2("git", c("-C", shQuote(source_hub), "cat-file", "-e",
                     shQuote(paste0(base_sha, "^{commit}"))),
            stdout = FALSE, stderr = FALSE) == 0L
  hubValidations::capture_check_cnd(
    check = ok,
    file_path = file_path,
    msg_subject = "The staged forecast source",
    msg_attribute = "available."
  )
}
