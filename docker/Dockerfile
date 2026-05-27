# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1: builder — solve conda env, install farmland-mpc + webui deps
# ============================================================================
FROM mambaorg/micromamba:1.5.10 AS builder

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /repo

# Build-time channel override. Defaults to upstream conda-forge; override via
#   docker build --build-arg CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
# when building from inside China. The runtime image is unaffected — clients
# just pull the resolved image.
ARG CONDA_CHANNEL=https://conda.anaconda.org/conda-forge

# Copy the whole repo first — environment.yml has a `pip: -e .` line that
# requires pyproject.toml to be present at solve time. The build-context size
# is kept small via .dockerignore.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /repo

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN printf "channels:\n  - %s\nshow_channel_urls: true\nremote_connect_timeout_secs: 30\nremote_max_retries: 5\nremote_backoff_factor: 2\n" "$CONDA_CHANNEL" > ~/.condarc \
    && micromamba install -y -n base -f /repo/environment.yml \
    && micromamba clean -a -y

# Webui-only extras (kept out of environment.yml so notebook/CLI users don't pay for them
# transitively). Pin loosely; FastAPI / uvicorn / python-multipart are all stable APIs.
RUN micromamba install -y -n base -c "$CONDA_CHANNEL" \
        jupyterlab=4 \
        fastapi \
        uvicorn-standard \
        python-multipart \
        jinja2 \
    && micromamba clean -a -y

# ============================================================================
# Stage 2: runtime — copy resolved env, drop build tools
# ============================================================================
FROM mambaorg/micromamba:1.5.10

LABEL org.opencontainers.image.title="farmland-mpc"
LABEL org.opencontainers.image.description="County-scale farmland consolidation via contrastive world-model + MPC. Pure Python, CPU."
LABEL org.opencontainers.image.source="https://github.com/zhouning/arcgis-farmland-mpc"
LABEL org.opencontainers.image.licenses="MIT"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make /work the canonical mount point for client data; permissions match $MAMBA_USER
RUN mkdir -p /work && chown $MAMBA_USER:$MAMBA_USER /work

USER $MAMBA_USER
WORKDIR /work

COPY --from=builder /opt/conda /opt/conda
COPY --from=builder --chown=$MAMBA_USER:$MAMBA_USER /repo /repo

# Activate the conda env on every shell (including non-interactive)
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    JUPYTER_ENABLE_LAB=yes \
    FARMLAND_MPC_WORK=/work

# Tini handles signals correctly (Ctrl-C in jupyter/uvicorn, OOM, k8s)
ENTRYPOINT ["/usr/bin/tini", "--", "/repo/docker/entrypoint.sh"]
CMD []

EXPOSE 8888 8000

# Sanity probe — both jupyter and webui expose /healthz; CLI mode skips healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${HEALTHCHECK_PORT:-8888}/api/status \
        || curl -fsS http://localhost:${HEALTHCHECK_PORT:-8000}/healthz \
        || exit 0
