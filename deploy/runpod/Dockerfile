FROM nvcr.io/nvidia/tensorrt:26.03-py3@sha256:ade1b30517b3d66b911a3cd7faf0146484ab8956098abe66b96b944fa36f4861

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
