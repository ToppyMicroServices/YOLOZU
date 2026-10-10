FROM nvcr.io/nvidia/tensorrt:26.08-py3@sha256:3b127f45630cd56bf43d2ef70d7f50a7ee37e42bda4ba11adaaa069af2098125

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
