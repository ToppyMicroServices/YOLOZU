FROM nvcr.io/nvidia/tensorrt:26.02-py3@sha256:bf56f269a505dfbba09ed146a428cc9f6c952f167bc08df53d5354d217eac0d3

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
RUN python3 tools/ci/install_with_hashes.py --requirements requirements-trt-tools.lock
