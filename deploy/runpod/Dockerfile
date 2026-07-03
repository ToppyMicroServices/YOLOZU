FROM nvcr.io/nvidia/tensorrt:26.06-py3@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3

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
