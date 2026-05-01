FROM nvcr.io/nvidia/tensorrt:26.04-py3@sha256:470a9179f700d684693ebbc552d62f2769725ad80fd499ca6f63ef8cf8fb4238

WORKDIR /workspace/YOLOZU

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/YOLOZU

# Minimal OS deps for common Python wheels (pycocotools/opencv) + git for run records.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# YOLOZU intentionally keeps torch out of requirements; this image is for tooling + TRT/ORT exporters.
RUN python3 tools/ci/install_with_hashes.py --requirements requirements-locks/requirements-trt-tools.lock
