FROM python:3.14-slim-bookworm@sha256:a70519002c49552ea0a853de47599cf40479b001bd7a624f1112eaf44dcaccc7

WORKDIR /opt/yolozu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE NOTICE COPYRIGHT ./
COPY requirements-locks ./requirements-locks
COPY tools/ci/install_with_hashes.py tools/ci/install_with_hashes.py
COPY yolozu ./yolozu

RUN python tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-runtime.lock --install-local-wheel

RUN useradd --create-home --uid 10001 yolozu && \
    chown -R yolozu:yolozu /opt/yolozu
USER yolozu

ENTRYPOINT ["yolozu"]
CMD ["--help"]
