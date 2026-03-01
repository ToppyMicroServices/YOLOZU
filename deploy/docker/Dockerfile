ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

WORKDIR /opt/yolozu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY pyproject.toml README.md LICENSE NOTICE COPYRIGHT ./
COPY yolozu ./yolozu

RUN python -m pip install .

RUN useradd --create-home --uid 10001 yolozu && \
    chown -R yolozu:yolozu /opt/yolozu
USER yolozu

ENTRYPOINT ["yolozu"]
CMD ["--help"]
