# Performance

Runtime benchmarks of carto-flow, one page per cartogram type. The numbers come from the benchmark suites in `benchmarks/`, which the benchmark workflow runs on a CI runner and stores in `benchmark_results.json`. They are comparable within one run, not across machines.

- [Flow cartogram](flow-cartogram.ipynb): grid size, parallelism, single and multi-resolution, vertex count.
- [Voronoi and proportional cartograms](voronoi-and-proportional.ipynb): runtime on US states and congressional districts.
