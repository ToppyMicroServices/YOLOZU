FROM python:3.14-slim-bookworm@sha256:55e465cb7e50cd1d7217fcb5386aa87d0356ca2cd790872142ef68d9ef6812b4

WORKDIR /opt/yolozu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE NOTICE COPYRIGHT requirements-runtime.lock ./
COPY tools/ci/install_with_hashes.py tools/ci/install_with_hashes.py
COPY yolozu ./yolozu

RUN python tools/ci/install_with_hashes.py --requirements requirements-runtime.lock --install-local-wheel

RUN useradd --create-home --uid 10001 yolozu && \
    chown -R yolozu:yolozu /opt/yolozu
USER yolozu

ENTRYPOINT ["yolozu"]
CMD ["--help"]
