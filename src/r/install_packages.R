# The rocker/r-ver:4.4.1 image points to a fixed-date CRAN snapshot,
# so package versions are pinned and reproducible.
install.packages("readr")

if (!requireNamespace("readr", quietly = TRUE)) {
  stop("Package 'readr' failed to install")
}
cat("Installed readr version:", as.character(packageVersion("readr")), "\n")