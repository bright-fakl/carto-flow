# Flow cartogram comparison

Compares carto-flow with other flow cartogram implementations on the US states, congressional districts and counties:

| Tool | Version |
|---|---|
| carto-flow | the checked-out source, with `validity` set to `check` (default), `repair-crossings` and `repair-quadtree` |
| go-cart (`cartogram-cpp`) | commit 9200b14 |
| R `cartogram` (`cartogram_cont`, Dougenik et al.) | 0.3.0 |
| R `cartogramR` (`gsm` and `gn`) | 1.5.1 |
| Python `cartogram` (Dougenik et al.) | 1.0.2 |

A tool does not decide when it is accurate enough. The iterative tools run with a stopping tolerance that cannot be reached and an increasing iteration cap, so every run stops at its cap. go-cart has no iteration cap; it runs with maximum area errors of 5, 2, 1, 0.5 and 0.2 %, which is also its own stopping rule. The accuracy of every output is then measured with the same metrics from the output geometries, so the results do not depend on how a tool reports its own error:

- area error: relative deviation of each region's share of the total area from its share of the total value (mean and maximum);
- validity: the number of output regions that are not valid polygons.

`time_to_target_s` in the results is the shortest median time among the runs whose measured maximum area error is within 5, 1 and 0.5 %. `null` means no run reached the target.

Wall-clock times of go-cart include reading the files and starting the process; the other tools are timed around the call. Times are the median of `--repeats` runs (default 3), after a warm-up that is not timed.

## Run it with Docker

From the repository root:

```bash
docker build -f benchmarks/comparison/Dockerfile -t carto-flow-comparison .
```

```bash
docker run --rm -v "$PWD:/repo" -v "$PWD/comparison_out:/out" carto-flow-comparison --dataset states --output /out/comparison_results.json
```

The image holds the tools and the Python dependencies; the repository is mounted, so changes to the source do not require a rebuild. The go-cart build takes about ten minutes and about 4 GB of memory (`--build-arg JOBS=...` changes the number of compile jobs). `--tools` selects tools, `--dataset` is `states`, `districts` or `counties`, and `--repeats` sets the number of timed runs.

Compare timings only within one run: they depend on the machine, and on macOS and Windows Docker runs in a virtual machine.

## Run it without Docker

Install the tools listed above, put the go-cart binary on the `PATH` (or pass `--go-cart`), make `Rscript` available, and run `python benchmarks/comparison/run_comparison.py`. go-cart needs longitude/latitude input and non-numeric region ids; the script handles both. The Python `cartogram` package must be installed in the same environment as carto-flow.
