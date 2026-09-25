# Releasing carto-flow

Maintainer procedure. Every step below is manual or manually triggered; nothing
releases on a merge to `main`.

## What the automation does

| Workflow | Trigger | Effect |
| --- | --- | --- |
| `release.yml` | manual, takes a version | bumps the version, commits, tags, creates the GitHub Release |
| `pypi-publish.yml` | a GitHub Release being published | builds and uploads to PyPI |
| `deploy-docs.yml` | push to `main`, or manual | publishes the documentation site |
| `benchmarks.yml` | manual only | runs the flow-cartogram benchmarks and commits the results file |

`release.yml` owns the version. Do not bump `pyproject.toml`,
`src/carto_flow/__init__.py` or `mkdocs.yml` by hand — pass the version to the
workflow and it writes all three through `scripts/bump_version.py`.

`release.yml` reads the release notes out of `CHANGELOG.md`. It looks for a
`## [<version>]` section and fails if that section is missing or empty, so the
changelog and the release cannot disagree.

## Before releasing

1. **Update `CHANGELOG.md`.** Add a `## [<version>] - <date>` section. This is
   required: the release fails without it, and its body becomes the release
   notes. Group breaking changes first, then features, then fixes.

2. **Check the migration guide** if the release breaks anything.
   `docs/migrating-to-2.0.md` is the template: one section per break, the old
   call and the new call as code, and what a caller sees if they do nothing —
   an exception, a warning, or silently different output.

3. **Run the benchmarks** and review the result.

       gh workflow run benchmarks.yml

   The workflow commits `docs/explanations/benchmark_results.json` to the branch
   it ran on. `docs/explanations/performance.ipynb` renders that file, so the
   published performance page shows whatever the last run produced. The
   benchmarks measure the flow cartogram only.

   The run takes about nine minutes. Check it succeeded — it commits nothing on
   failure, so a failed run leaves the previous results in place and the
   performance page silently keeps showing older numbers.

   Results are hardware-dependent. Numbers from a CI runner are not comparable
   with numbers measured on a workstation; the machine is recorded in
   `machine_info` inside the results file.

4. **Confirm CI is green on `main`** and that the documentation builds:

       make docs-test

5. **Read the built documentation.** `make docs` serves it locally. The
   changelog and migration pages are the ones most likely to be wrong, because
   nothing tests their content.

6. **Publish to TestPyPI first, when the release changes packaging.** Run
   `pypi-publish.yml` manually with the environment set to `testpypi`, then
   install from TestPyPI into an empty environment and import the package.

   Do this before releasing, not after: once `release.yml` has tagged and
   published a GitHub Release, `pypi-publish.yml` uploads to PyPI
   automatically, and a version number on PyPI cannot be reused.

   Worth doing whenever `pyproject.toml`'s build configuration changed, a
   bundled data file was added, moved or renamed, or a dependency was added or
   made required. Skip it for a release that only changes Python source.

## Releasing

    gh workflow run release.yml -f version=X.Y.Z

The version must be semver. The workflow bumps the three version files, commits
and pushes, tags `X.Y.Z`, and creates the GitHub Release with the changelog
section as its body. Publishing that release triggers `pypi-publish.yml`.

## After releasing

1. **Check the PyPI upload.** `pypi-publish.yml` runs on the release being
   published; confirm it succeeded and that the new version is on PyPI.
2. **Check the documentation site** picked up the new version.
3. **Install the published package in a clean environment** and import it. This
   catches packaging problems that a source checkout hides, such as a data file
   that was never added to the wheel.
