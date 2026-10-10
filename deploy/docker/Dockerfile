FROM python:3.14-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9

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
