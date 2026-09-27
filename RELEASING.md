# Releasing carto-flow

Maintainer procedure. Every step below is manual or manually triggered; nothing
releases on a merge to `main`.

## What the automation does

| Workflow | Trigger | Effect |
| --- | --- | --- |
| `release.yml` | manual, takes a version | bumps the version, commits, tags, creates the GitHub Release |
| `pypi-publish.yml` | a GitHub Release being published | builds and uploads to PyPI, or to TestPyPI if the release is a pre-release |
| `deploy-docs.yml` | push to `main`, or manual | publishes the documentation site |
| `benchmarks.yml` | manual only | runs the flow-cartogram benchmarks and commits the results file |

`release.yml` owns the version. Do not bump `pyproject.toml`,
`src/carto_flow/__init__.py` or `mkdocs.yml` by hand — pass the version to the
workflow and it writes all three through `scripts/bump_version.py`.

`release.yml` reads the release notes out of `CHANGELOG.md`. It looks for a
`## [<version>]` section and fails if that section is missing or empty, so the
changelog and the release cannot disagree. For a pre-release version such as
`2.0.0-rc1`, it first looks for `## [2.0.0-rc1]`, and if that is absent falls
back to `## [2.0.0]` — an rc and the final version it leads to ship the same
intended content, so they share one CHANGELOG entry instead of requiring a
dated section per candidate.

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

6. **When the release changes packaging, cut a release candidate first**
   instead of publishing to TestPyPI directly. A direct TestPyPI publish
   before releasing cannot work: the version only becomes `X.Y.Z` inside
   `release.yml` (it bumps `pyproject.toml`, `mkdocs.yml` and
   `src/carto_flow/__init__.py` itself), so anything published beforehand
   still carries the old version and, if that version already exists on
   TestPyPI, fails with "file already exists".

   Worth doing whenever `pyproject.toml`'s build configuration changed, a
   bundled data file was added, moved or renamed, or a dependency was added or
   made required. Skip it for a release that only changes Python source — go
   straight to [Releasing](#releasing).

## Releasing

### Release candidate (packaging changes only)

1. **Cut the candidate:**

       gh workflow run release.yml -f version=X.Y.Z-rc1

   This bumps the three version files, commits, tags `X.Y.Z-rc1`, and creates
   a GitHub Release marked as a pre-release — which skips the documentation
   deploy and makes `pypi-publish.yml` upload to TestPyPI instead of PyPI.

2. **Verify by hand.** Install from TestPyPI into a clean venv, then import the
   package and load a bundled dataset. TestPyPI does not mirror PyPI, so
   dependencies have to come from the real index. Run this from a directory
   that is *not* the repo root — from inside the repo, the import can resolve
   to the local source tree instead of the installed wheel:

       cd /tmp && \
       uv venv /tmp/tpypi && \
         VIRTUAL_ENV=/tmp/tpypi uv pip install \
           --index-strategy unsafe-best-match \
           --index-url https://test.pypi.org/simple/ \
           --extra-index-url https://pypi.org/simple/ \
           "carto-flow==X.Y.ZrcN"
       VIRTUAL_ENV=/tmp/tpypi uv run --no-project python -c \
         "import carto_flow, carto_flow.data as d; print(carto_flow.__version__, len(d.load_world()))"

   Loading a bundled dataset is the point of that import: it is what catches a
   data file missing from the wheel.

   Three details in that command are load-bearing, each of which silently
   breaks the check if dropped:

   - **Pin the exact version**, written PEP 440 style: the tag `2.0.0-rc1`
     normalizes to the version `2.0.0rc1`. TestPyPI still carries older stable
     releases, so an unpinned install resolves to one of those and verifies the
     previous release while appearing to pass.
   - **`--index-strategy unsafe-best-match`** is required. By default uv takes
     every version of a package from the first index that offers it, so
     dependency resolution fails once TestPyPI and PyPI both carry part of the
     dependency tree.
   - **`--no-project`**, together with the leading `cd /tmp`. Without both,
     `uv run` picks up the repository's own environment and `import carto_flow`
     resolves against local source rather than the installed wheel, so the
     check passes without testing the artifact at all.

   If this fails, cut `X.Y.Z-rc2` (bump the rc number, do not delete the tag)
   and repeat. This is the reason to use a release candidate at all: a failed
   verification costs an rc number, never the final version, and never
   touches real PyPI.

3. **Release for real** once verification passes — see below.

### Final release

    gh workflow run release.yml -f version=X.Y.Z

The version must be semver. The workflow bumps the three version files, commits
and pushes, tags `X.Y.Z`, and creates the GitHub Release with the `## [X.Y.Z]`
changelog section as its body — the same section any of its rcs already fell
back to (see "What the automation does" above). Publishing that release
triggers `pypi-publish.yml`, which uploads to real PyPI since the release is
not a pre-release, and deploys the documentation site.

## After releasing

1. **Check the PyPI upload.** `pypi-publish.yml` runs on the release being
   published; confirm it succeeded and that the new version is on PyPI.
2. **Check the documentation site** picked up the new version.
3. **Install the published package in a clean environment** and import it:

       cd /tmp && \
       uv venv /tmp/pypi && \
         VIRTUAL_ENV=/tmp/pypi uv pip install carto-flow
       VIRTUAL_ENV=/tmp/pypi uv run python -c \
         "import carto_flow, carto_flow.data as d; print(carto_flow.__version__, len(d.load_world()))"

   This catches packaging problems a source checkout hides, such as a data file
   that was never added to the wheel.
