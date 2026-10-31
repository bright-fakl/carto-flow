# Run the R tools of the comparison on input.gpkg (columns ID, value, geometry).
# Usage: Rscript run_r.R WORKDIR REPEATS TOL [TOL ...]
# Writes one GeoPackage per (tool, tolerance) and r_times.json with the timings.
suppressMessages({
  library(sf)
  library(cartogram)
  library(cartogramR)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
workdir <- args[1]
repeats <- as.integer(args[2])
tolerances <- as.numeric(args[-(1:2)])
input <- st_read(file.path(workdir, "input.gpkg"), quiet = TRUE)

tools <- list(
  "R cartogram" = function(tol) {
    cartogram_cont(input, "value", itermax = 200, maxSizeError = 1 + tol, prepare = "adjust")
  },
  "cartogramR gsm" = function(tol) {
    cartogramR(input, "value", method = "gsm", options = list(relerror = tol, maxit = 200))
  },
  "cartogramR gn" = function(tol) {
    cartogramR(input, "value", method = "gn", options = list(relerror = tol, maxit = 200))
  }
)
geometry_of <- function(res) {
  if (inherits(res, "cartogramR")) st_geometry(as.sf(res)) else st_geometry(res)
}

# Warm-up: loads the compiled code and plans the FFTs outside the timed runs.
invisible(tools[["cartogramR gsm"]](0.05))

out <- list()
for (name in names(tools)) {
  for (tol in tolerances) {
    times <- numeric(repeats)
    for (i in seq_len(repeats)) {
      t0 <- Sys.time()
      res <- tools[[name]](tol)
      times[i] <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    }
    file <- sprintf("r_%s_%s.gpkg", gsub(" ", "_", name), tol)
    st_write(st_sf(geometry = geometry_of(res)), file.path(workdir, file), quiet = TRUE, delete_dsn = TRUE)
    out[[length(out) + 1]] <- list(tool = name, tolerance = tol, times_s = I(times), file = file)
  }
}
write(toJSON(out, auto_unbox = TRUE, digits = NA), file.path(workdir, "r_times.json"))
