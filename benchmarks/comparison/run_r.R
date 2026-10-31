# Run the R tools of the comparison on input.gpkg (columns ID, value, geometry).
# Usage: Rscript run_r.R WORKDIR REPEATS BUDGET_S
# A tool stops climbing its ladder once a run takes longer than BUDGET_S seconds.
# WORKDIR/r_ladders.json maps each tool to its iteration caps. The stopping
# tolerances are set so that they cannot be reached and every run stops at its cap.
# Writes one GeoPackage per (tool, cap) and r_times.json with the timings.
suppressMessages({
  library(sf)
  library(cartogram)
  library(cartogramR)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
workdir <- args[1]
repeats <- as.integer(args[2])
budget <- as.numeric(args[3])
ladders <- fromJSON(file.path(workdir, "r_ladders.json"))
input <- st_read(file.path(workdir, "input.gpkg"), quiet = TRUE)

tools <- list(
  "R cartogram" = function(cap) {
    cartogram_cont(input, "value", itermax = cap, maxSizeError = 1 + 1e-9, prepare = "none")
  },
  "cartogramR gsm" = function(cap) {
    cartogramR(input, "value", method = "gsm", options = list(relerror = 1e-9, maxit = cap))
  },
  "cartogramR gn" = function(cap) {
    cartogramR(input, "value", method = "gn", options = list(relerror = 1e-9, maxit = cap))
  }
)
geometry_of <- function(res) {
  if (inherits(res, "cartogramR")) st_geometry(as.sf(res)) else st_geometry(res)
}

# Warm-up: loads the compiled code and plans the FFTs outside the timed runs.
invisible(tools[["cartogramR gsm"]](2))

out <- list()
for (name in names(tools)) {
  for (cap in ladders[[name]]) {
    times <- numeric(repeats)
    for (i in seq_len(repeats)) {
      t0 <- Sys.time()
      res <- tools[[name]](cap)
      times[i] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    }
    file <- sprintf("r_%s_%s.gpkg", gsub(" ", "_", name), cap)
    st_write(st_sf(geometry = geometry_of(res)), file.path(workdir, file), quiet = TRUE, delete_dsn = TRUE)
    out[[length(out) + 1]] <- list(tool = name, max_iterations = cap, times_s = I(times), file = file)
    if (median(times) > budget) break
  }
}
write(toJSON(out, auto_unbox = TRUE, digits = NA), file.path(workdir, "r_times.json"))
