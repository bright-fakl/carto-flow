# Environment for the flow cartogram comparison (see README.md).
#
# Build from the repository root:
#   docker build -f benchmarks/comparison/Dockerfile -t carto-flow-comparison .
#
# The image holds the tools and the Python dependencies of carto-flow; the
# repository itself is mounted at run time, so editing the source does not
# require a rebuild.

# --- go-cart (cartogram-cpp), built from a pinned commit -------------------
FROM ubuntu:24.04 AS go-cart
ARG GO_CART_COMMIT=9200b14b983581b78992dd6944cceedf384bb02c
# Parallel compile jobs; 2 keeps peak memory near 4 GB.
ARG JOBS=2
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates python3-pip python3-venv pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build-venv && /opt/build-venv/bin/pip install --no-cache-dir conan cmake ninja
ENV PATH=/opt/build-venv/bin:$PATH
RUN git clone https://github.com/mgastner/cartogram-cpp.git /src \
    && git -C /src checkout ${GO_CART_COMMIT}
WORKDIR /src
RUN conan profile detect \
    && conan remote update conancenter --url=https://center2.conan.io \
    && conan install . --build=missing -s build_type=Release -s compiler.cppstd=20
RUN . build/Release/generators/conanbuild.sh \
    && cmake -B build/Release -S . \
        -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build/Release --target cartogram -j${JOBS}

# --- runtime: R packages, Python packages, go-cart binary ------------------
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        r-base r-base-dev r-cran-sf r-cran-jsonlite r-cran-rcpp r-cran-rcpparmadillo \
        r-cran-sp r-cran-remotes libfftw3-dev \
        libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
        build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned CRAN versions of the compared R packages.
ARG R_CARTOGRAM_VERSION=0.3.0
ARG R_CARTOGRAMR_VERSION=1.5.1
RUN Rscript -e 'options(repos = "https://cloud.r-project.org"); \
        remotes::install_version("cartogram", "'${R_CARTOGRAM_VERSION}'", upgrade = "never"); \
        remotes::install_version("cartogramR", "'${R_CARTOGRAMR_VERSION}'", upgrade = "never")'

COPY --from=go-cart /src/build/Release/cartogram /usr/local/bin/cartogram

# Python: the locked dependencies of carto-flow, plus the compared package.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --no-editable
ARG PY_CARTOGRAM_VERSION=1.0.2
RUN uv pip install --python /opt/venv/bin/python "cartogram==${PY_CARTOGRAM_VERSION}" pyarrow

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/repo/src
WORKDIR /repo
ENTRYPOINT ["python", "/repo/benchmarks/comparison/run_comparison.py"]
CMD ["--dataset", "states", "--output", "/out/comparison_results.json"]
